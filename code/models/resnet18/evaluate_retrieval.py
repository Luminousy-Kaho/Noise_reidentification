import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'resnet船型分类'))
from core import load_split, make_loaders  # noqa: E402


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def l2_normalize(x: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def extract_embeddings(model, loader, device):
    rows = []
    model.eval()
    with torch.no_grad():
        for x, y, wav_paths, true_class_names in loader:
            x = x.to(device, non_blocking=True)
            emb = model(x).detach().cpu().numpy()
            y_np = y.numpy()
            for i in range(len(wav_paths)):
                wavp = str(wav_paths[i])
                stem = Path(wavp).stem
                source_long_id = stem.split('__seg')[0] if '__seg' in stem else stem
                rows.append(
                    {
                        'wav_path': wavp,
                        'source_long_id': source_long_id,
                        'true_label': int(y_np[i]),
                        'true_class_name': str(true_class_names[i]),
                        'embedding': emb[i],
                    }
                )
    return rows


def build_prototypes(val_df: pd.DataFrame, val_emb: np.ndarray):
    ships = sorted(val_df['true_class_name'].unique().tolist())
    proto = []
    for ship in ships:
        idx = np.where(val_df['true_class_name'].to_numpy() == ship)[0]
        p = val_emb[idx].mean(axis=0)
        p = p / max(np.linalg.norm(p), 1e-12)
        proto.append(p)
    return np.stack(proto), ships


def rank_queries(query_ids, query_true_ships, query_emb, proto, proto_ships):
    rows = []
    ap_list = []
    top1 = 0
    top5 = 0

    for i in range(len(query_ids)):
        qid = query_ids[i]
        true_ship = query_true_ships[i]
        sims = np.dot(proto, query_emb[i])
        order = np.argsort(-sims)

        hits = []
        rank_true = None
        for r, idx in enumerate(order, start=1):
            pred_ship = proto_ships[idx]
            correct = int(pred_ship == true_ship)
            hits.append(correct)
            if correct and rank_true is None:
                rank_true = r
            rows.append(
                {
                    'query_id': qid,
                    'true_ship': true_ship,
                    'rank': r,
                    'gallery_ship': pred_ship,
                    'similarity': float(sims[idx]),
                    'is_relevant': correct,
                }
            )

        if rank_true == 1:
            top1 += 1
        if rank_true is not None and rank_true <= 5:
            top5 += 1

        # single relevant item case => AP = 1/rank_true
        ap = 0.0 if rank_true is None else (1.0 / rank_true)
        ap_list.append(ap)

    n = len(query_ids)
    metrics = {
        'query_count': n,
        'Recall@1': top1 / n if n else 0.0,
        'Recall@5': top5 / n if n else 0.0,
        'mAP': float(np.mean(ap_list)) if ap_list else 0.0,
    }
    return pd.DataFrame(rows), metrics


def run_one(setting_name: str, run_dir: Path, split_json: Path, out_dir: Path, seed: int = 42):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(split_json)
    ckpt = torch.load(run_dir / 'best_model.pt', map_location='cpu')
    class_names = ckpt['class_names']
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    for s in split:
        for i, it in enumerate(split[s]):
            split[s][i].label = class_to_idx[it.class_name]

    run_params_path = run_dir / 'run_params.json'
    run_params = json.loads(run_params_path.read_text(encoding='utf-8')) if run_params_path.exists() else {}
    shard_feature_root = Path(run_params.get('shard_feature_root', ''))

    feat_cfg = {
        'sample_rate': 16000,
        'clip_sec': 5.0,
        'n_fft': 1024,
        'win_length': 1024,
        'hop_length': 512,
        'n_mels': 128,
        'fmin': 20.0,
        'fmax': 8000.0,
    }

    _, val_dl, test_dl = make_loaders(
        split=split,
        batch_size=64,
        num_workers=0,
        feat_cfg=feat_cfg,
        feature_root=None,
        shard_feature_root=shard_feature_root,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)
    model.load_state_dict(ckpt['model'])
    model.fc = nn.Identity()
    model.eval()

    val_rows = extract_embeddings(model, val_dl, device)
    test_rows = extract_embeddings(model, test_dl, device)

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    val_emb = l2_normalize(np.stack(val_df['embedding'].to_numpy()))
    test_emb = l2_normalize(np.stack(test_df['embedding'].to_numpy()))

    val_df_meta = val_df.drop(columns=['embedding']).copy()
    test_df_meta = test_df.drop(columns=['embedding']).copy()
    val_df_meta.to_csv(out_dir / 'val_embedding_meta.csv', index=False, encoding='utf-8-sig')
    test_df_meta.to_csv(out_dir / 'test_embedding_meta.csv', index=False, encoding='utf-8-sig')

    proto, proto_ships = build_prototypes(val_df_meta, val_emb)
    np.save(out_dir / 'val_prototypes.npy', proto)
    pd.DataFrame({'ship_name': proto_ships}).to_csv(out_dir / 'val_prototype_meta.csv', index=False, encoding='utf-8-sig')

    # segment query retrieval
    seg_ids = test_df_meta['wav_path'].tolist()
    seg_true = test_df_meta['true_class_name'].tolist()
    seg_rank_df, seg_metrics = rank_queries(seg_ids, seg_true, test_emb, proto, proto_ships)
    seg_rank_df.to_csv(out_dir / 'segment_retrieval_rankings.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'Setting': setting_name, **seg_metrics}]).to_csv(out_dir / 'retrieval_segment_metrics.csv', index=False, encoding='utf-8-sig')

    # recording query retrieval: average test segments by source_long_id
    grp = test_df_meta.groupby(['source_long_id', 'true_class_name'], as_index=False).indices
    rec_ids, rec_true, rec_embs = [], [], []
    for (sid, ship), idx in grp.items():
        idx = list(idx)
        e = test_emb[idx].mean(axis=0)
        e = e / max(np.linalg.norm(e), 1e-12)
        rec_ids.append(sid)
        rec_true.append(ship)
        rec_embs.append(e)

    rec_emb = np.stack(rec_embs) if rec_embs else np.zeros((0, proto.shape[1]), dtype=np.float32)
    rec_rank_df, rec_metrics_base = rank_queries(rec_ids, rec_true, rec_emb, proto, proto_ships)
    rec_rank_df.to_csv(out_dir / 'recording_retrieval_rankings.csv', index=False, encoding='utf-8-sig')

    rec_metrics = {
        'query_count': rec_metrics_base['query_count'],
        'Top-1': rec_metrics_base['Recall@1'],
        'Top-5': rec_metrics_base['Recall@5'],
    }
    pd.DataFrame([{'Setting': setting_name, **rec_metrics}]).to_csv(out_dir / 'retrieval_recording_metrics.csv', index=False, encoding='utf-8-sig')

    md = []
    md.append(f"# {setting_name} Retrieval 报告")
    md.append('')
    md.append('## Segment Query (test segments -> val prototypes)')
    md.append(f"- Recall@1: {seg_metrics['Recall@1']:.6f}")
    md.append(f"- Recall@5: {seg_metrics['Recall@5']:.6f}")
    md.append(f"- mAP: {seg_metrics['mAP']:.6f}")
    md.append('')
    md.append('## Recording Query (test recordings -> val prototypes)')
    md.append(f"- Top-1: {rec_metrics['Top-1']:.6f}")
    md.append(f"- Top-5: {rec_metrics['Top-5']:.6f}")
    md.append('')
    md.append('## 说明')
    md.append('- gallery 固定为 val prototypes（每船一个）')
    md.append('- query 与 gallery 相似度为 cosine similarity')
    md.append('- mAP 在本单正样本设定下等价于 mean reciprocal rank')
    (out_dir / 'discussion_retrieval.md').write_text('\n'.join(md), encoding='utf-8')

    return {
        'Setting': setting_name,
        'Recall@1': seg_metrics['Recall@1'],
        'Recall@5': seg_metrics['Recall@5'],
        'mAP': seg_metrics['mAP'],
        'Top-1': rec_metrics['Top-1'],
        'Top-5': rec_metrics['Top-5'],
    }


def main():
    project_root = Path(__file__).resolve().parents[3]
    out_root = project_root / 'resnet18_embedding_ship_verification' / 'results_retrieval_query'
    out_root.mkdir(parents=True, exist_ok=True)

    split_json = project_root / 'split_protocols' / 'deepship' / 'ship_identity' / 'min5_median_split_20260511_163606' / 'split_detail.json'

    settings = [
        {
            'name': 'resnet18_ce',
            'run_dir': project_root / 'experiments' / 'resnet18' / 'deepship_ship_identity_25class' / 'imagenet_pretrained_closedset_runs' / 'runs' / 'ship_identity_min5_no_resample_median_split_20260511_203220',
        },
        {
            'name': 'resnet18_ce_os',
            'run_dir': project_root / 'experiments' / 'resnet18' / 'deepship_ship_identity_25class' / 'imagenet_pretrained_closedset_runs' / 'runs' / 'ship_identity_min5_resample_median_split_20260511_164138',
        },
    ]

    rows = []
    for s in settings:
        row = run_one(s['name'], s['run_dir'], split_json, out_root / s['name'], seed=42)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / 'summary_retrieval_metrics.csv', index=False, encoding='utf-8-sig')

    md = []
    md.append('# Retrieval Query 总览')
    md.append('')
    md.append('| Setting | Recall@1 | Recall@5 | mAP | Top-1 | Top-5 |')
    md.append('|---|---:|---:|---:|---:|---:|')
    for r in rows:
        md.append(f"| {r['Setting']} | {r['Recall@1']:.6f} | {r['Recall@5']:.6f} | {r['mAP']:.6f} | {r['Top-1']:.6f} | {r['Top-5']:.6f} |")
    best_r1 = max(rows, key=lambda x: x['Recall@1'])
    best_top1 = max(rows, key=lambda x: x['Top-1'])
    md.append('')
    md.append('## 结论')
    md.append(f"- Segment 检索最优 Recall@1: `{best_r1['Setting']}` ({best_r1['Recall@1']:.6f})")
    md.append(f"- Recording 检索最优 Top-1: `{best_top1['Setting']}` ({best_top1['Top-1']:.6f})")
    md.append('- 若 Top-5 明显高于 Top-1，说明模型通常能把真值放入候选前列，但第一名排序仍有提升空间。')
    (out_root / 'overview_retrieval.md').write_text('\n'.join(md), encoding='utf-8')

    print(out_root)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
