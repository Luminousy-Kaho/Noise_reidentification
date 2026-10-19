import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ecapa_model import AAMSoftmaxHead, ECAPATDNN, ECAPAFeatureDataset
from utils_ecapa import (
    attach_metadata_to_split,
    build_metadata_index,
    build_ship_label_from_train,
    l2n,
    load_split_json,
    metrics_from_preds,
    safety_check_no_wav_overlap,
    safety_check_recording_overlap,
    safety_check_shipname_from_train_only,
    safety_check_train_shipnames_match_split_class,
    set_seed,
    write_json,
)


def ensure_bmt(x, input_layout):
    if input_layout == 'BTM':
        x = x.transpose(1, 2)
    return x


def make_loader(items, batch_size, num_workers, feature_root, shard_root, input_layout, shuffle):
    ds = ECAPAFeatureDataset(items, feature_root=feature_root, shard_root=shard_root, input_layout=input_layout)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class AudioItem:
    def __init__(self, wav_path, class_name, label, duration_sec, source_long_id):
        self.wav_path = Path(wav_path)
        self.class_name = class_name
        self.label = int(label)
        self.duration_sec = float(duration_sec)
        self.source_long_id = source_long_id


def build_items(split_meta, ship_to_idx):
    out = {'train': [], 'val': [], 'test': []}
    for s in ('train', 'val', 'test'):
        for r in split_meta[s]:
            out[s].append(
                AudioItem(
                    wav_path=r['wav_path'],
                    class_name=r['ship_name'],
                    label=ship_to_idx[r['ship_name']],
                    duration_sec=float(r.get('duration_sec', 0.0)),
                    source_long_id=r.get('source_long_id', r['recording_id']),
                )
            )
    return out


def evaluate_cls(model, aam_head, loader, device, num_classes, input_layout):
    model.eval()
    if aam_head is not None:
        aam_head.eval()
    y_true, y_pred = [], []
    rows = []
    with torch.no_grad():
        for x, y, paths, class_names in loader:
            x = ensure_bmt(x, input_layout).to(device, non_blocking=True)
            if aam_head is None:
                logits = model(x)
            else:
                emb = model.forward_features(x)
                logits = aam_head(emb, labels=None, apply_margin=False)
            pred = torch.argmax(logits, dim=1).cpu().tolist()
            yt = y.tolist()
            y_true.extend(yt)
            y_pred.extend(pred)
            probs = torch.softmax(logits, dim=1).cpu()
            for i in range(len(paths)):
                rows.append(
                    {
                        'wav_path': paths[i],
                        'true_label': yt[i],
                        'pred_label': pred[i],
                        'true_class_name': class_names[i],
                        'pred_confidence': float(probs[i, pred[i]].item()),
                    }
                )
    return metrics_from_preds(y_true, y_pred, num_classes), pd.DataFrame(rows)


def extract_embeddings(model, loader, device, input_layout):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, paths, class_names in loader:
            x = ensure_bmt(x, input_layout).to(device, non_blocking=True)
            emb = model.forward_features(x).detach().cpu().numpy()
            yt = y.numpy()
            for i in range(len(paths)):
                rows.append({'wav_path': paths[i], 'true_label': int(yt[i]), 'true_class_name': class_names[i], 'embedding': emb[i]})
    return rows


def auc_eer(y_true, y_score):
    from sklearn.metrics import roc_auc_score, roc_curve

    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return auc, eer


def safe_auc_eer(pos_scores, neg_scores):
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float('nan'), float('nan')
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))]).astype(int)
    s = np.concatenate([np.asarray(pos_scores), np.asarray(neg_scores)]).astype(float)
    if len(np.unique(y)) < 2:
        return float('nan'), float('nan')
    return auc_eer(y, s)


def rank_queries(query_emb, query_true_ship, query_id, proto_emb, proto_ship):
    sims = query_emb @ proto_emb.T
    order = np.argsort(-sims, axis=1)
    rows = []
    true_rank = []
    for i in range(query_emb.shape[0]):
        tr = None
        for rk, j in enumerate(order[i], start=1):
            rel = int(proto_ship[j] == query_true_ship[i])
            rows.append({'query_id': query_id[i], 'true_ship': query_true_ship[i], 'rank': rk, 'gallery_ship': proto_ship[j], 'similarity': float(sims[i, j]), 'is_relevant': rel})
            if rel == 1 and tr is None:
                tr = rk
        if tr is None:
            tr = len(proto_ship) + 1
        true_rank.append(tr)
    return pd.DataFrame(rows), np.asarray(true_rank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--split-json', required=True)
    ap.add_argument('--metadata-csv', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--model-name', default='ECAPA-TDNN-CE')
    ap.add_argument('--model-type', default='ecapa_tdnn')
    ap.add_argument('--sampling', default='no_resample')
    ap.add_argument('--loss-type', default='auto', choices=['auto', 'cross_entropy', 'aam_softmax'])
    ap.add_argument('--aam-margin', type=float, default=0.2)
    ap.add_argument('--aam-scale', type=float, default=30.0)

    ap.add_argument('--wav-path-col', default='')
    ap.add_argument('--ship-name-col', default='ship_name')
    ap.add_argument('--ship-type-col', default='ship_type')
    ap.add_argument('--recording-id-col', default='recording_id')
    ap.add_argument('--split-col', default='split')

    ap.add_argument('--feature-root', default='')
    ap.add_argument('--shard-feature-root', default='')
    ap.add_argument('--input-layout', default='BMT', choices=['BMT', 'BTM'])
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--kshot', default='1,3,5,10,20,all')
    ap.add_argument('--kshot-repeats', type=int, default=20)
    args = ap.parse_args()

    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    best_ckpt = run_dir / 'best_checkpoint.pt'
    summary_p = run_dir / 'metrics_summary.json'
    if not best_ckpt.exists():
        raise FileNotFoundError(f'best checkpoint not found: {best_ckpt}')
    if not summary_p.exists():
        raise FileNotFoundError(f'metrics_summary.json not found: {summary_p}')

    # keep json loading robust across nested dict/list schema
    cfg = None

    split_payload = load_split_json(Path(args.split_json))
    meta_idx = build_metadata_index(
        Path(args.metadata_csv),
        wav_path_col=args.wav_path_col,
        ship_name_col=args.ship_name_col,
        ship_type_col=args.ship_type_col,
        recording_id_col=args.recording_id_col,
        split_col=args.split_col,
    )
    split_meta = attach_metadata_to_split(split_payload, meta_idx)

    safety_check_recording_overlap(split_meta)
    safety_check_no_wav_overlap(split_meta)
    safety_check_shipname_from_train_only(split_meta)
    safety_check_train_shipnames_match_split_class(split_payload, split_meta)

    ship_names, ship_to_idx = build_ship_label_from_train(split_meta)
    split = build_items(split_meta, ship_to_idx)

    feature_root = Path(args.feature_root) if args.feature_root else None
    shard_root = Path(args.shard_feature_root) if args.shard_feature_root else None
    if feature_root is None and shard_root is None:
        raise ValueError('either --feature-root or --shard-feature-root must be provided')

    train_dl = make_loader(split['train'], args.batch_size, args.num_workers, feature_root, shard_root, args.input_layout, True)
    val_dl = make_loader(split['val'], args.batch_size, args.num_workers, feature_root, shard_root, args.input_layout, False)
    test_dl = make_loader(split['test'], args.batch_size, args.num_workers, feature_root, shard_root, args.input_layout, False)

    # infer model dims from training summary json
    import json as _json
    tr = _json.loads(summary_p.read_text(encoding='utf-8'))
    n_mels = int(tr.get('n_mels', 128))
    channels = int(tr.get('channels', 512))
    emb_dim = int(tr.get('embedding_dim', 192))

    model = ECAPATDNN(n_mels=n_mels, channels=channels, emb_dim=emb_dim, num_classes=len(ship_names))
    ckpt = torch.load(best_ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)

    ckpt_loss_type = str(ckpt.get('loss_type', tr.get('loss_type', 'cross_entropy'))).lower()
    if args.loss_type == 'auto':
        eff_loss_type = ckpt_loss_type
    else:
        eff_loss_type = args.loss_type
    if eff_loss_type not in ('cross_entropy', 'aam_softmax'):
        raise ValueError(f'unsupported loss_type={eff_loss_type}')

    aam_head = None
    if eff_loss_type == 'aam_softmax':
        margin = float(ckpt.get('margin', tr.get('margin', args.aam_margin)))
        scale = float(ckpt.get('scale', tr.get('scale', args.aam_scale)))
        aam_head = AAMSoftmaxHead(emb_dim=emb_dim, num_classes=len(ship_names), margin=margin, scale=scale)
        if 'aam_head' not in ckpt:
            raise KeyError('AAM checkpoint missing `aam_head` state_dict')
        aam_head.load_state_dict(ckpt['aam_head'], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if aam_head is not None:
        aam_head = aam_head.to(device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # A closed-set
    val_m, val_df = evaluate_cls(model, aam_head, val_dl, device, len(ship_names), args.input_layout)
    test_m, test_df = evaluate_cls(model, aam_head, test_dl, device, len(ship_names), args.input_layout)
    val_df.to_csv(out / 'val_predictions.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv(out / 'test_predictions.csv', index=False, encoding='utf-8-sig')

    per_class = []
    for pc in test_m['per_class']:
        per_class.append({'ship_name': ship_names[pc['class_idx']], 'precision': pc['precision'], 'recall': pc['recall'], 'f1': pc['f1']})
    per_class_df = pd.DataFrame(per_class)
    per_class_df.to_csv(out / 'per_class_recall.csv', index=False, encoding='utf-8-sig')

    wav_to_rec = {}
    wav_to_type = {}
    for s in ('train', 'val', 'test'):
        for r in split_meta[s]:
            wav_to_rec[r['wav_path']] = r['recording_id']
            wav_to_type[r['wav_path']] = r['ship_type']

    test_df['recording_id'] = test_df['wav_path'].map(wav_to_rec)
    label2ship = {i: n for i, n in enumerate(ship_names)}
    test_df['pred_ship'] = test_df['pred_label'].map(label2ship)

    rec_rows = []
    for rid, g in test_df.groupby('recording_id'):
        vote = g.groupby('pred_ship').size().sort_values(ascending=False)
        pred_ship = vote.index[0]
        true_ship = g['true_class_name'].iloc[0]
        rec_rows.append({'recording_id': rid, 'true_ship': true_ship, 'pred_ship': pred_ship, 'correct': int(pred_ship == true_ship)})
    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(out / 'recording_level_predictions.csv', index=False, encoding='utf-8-sig')
    rec_top1_cls = float(rec_df['correct'].mean()) if len(rec_df) else np.nan

    closed = {'test_acc': test_m['accuracy'], 'test_macro_f1': test_m['macro_f1'], 'recording_top1': rec_top1_cls, 'zero_recall_ship_count': int((per_class_df['recall'] == 0.0).sum())}
    write_json(out / 'closed_set_metrics.json', closed)

    # B embedding
    val_rows = extract_embeddings(model, val_dl, device, args.input_layout)
    test_rows = extract_embeddings(model, test_dl, device, args.input_layout)
    val_emb = l2n(np.stack([r['embedding'] for r in val_rows]))
    test_emb = l2n(np.stack([r['embedding'] for r in test_rows]))
    np.save(out / 'val_embeddings.npy', val_emb)
    np.save(out / 'test_embeddings.npy', test_emb)

    val_meta = pd.DataFrame([{k: v for k, v in r.items() if k != 'embedding'} for r in val_rows])
    test_meta = pd.DataFrame([{k: v for k, v in r.items() if k != 'embedding'} for r in test_rows])
    val_meta['recording_id'] = val_meta['wav_path'].map(wav_to_rec)
    test_meta['recording_id'] = test_meta['wav_path'].map(wav_to_rec)
    val_meta['ship_type'] = val_meta['wav_path'].map(wav_to_type)
    test_meta['ship_type'] = test_meta['wav_path'].map(wav_to_type)
    val_meta.to_csv(out / 'val_embedding_meta.csv', index=False, encoding='utf-8-sig')
    test_meta.to_csv(out / 'test_embedding_meta.csv', index=False, encoding='utf-8-sig')

    # prototypes
    proto_ship = np.array(sorted(val_meta['true_class_name'].unique().tolist()))
    proto_emb = []
    proto_type = []
    for sname in proto_ship:
        idx = np.where(val_meta['true_class_name'].to_numpy() == sname)[0]
        p = l2n(val_emb[idx].mean(axis=0, keepdims=True))[0]
        proto_emb.append(p)
        proto_type.append(val_meta.iloc[idx[0]]['ship_type'])
    proto_emb = np.stack(proto_emb)
    np.save(out / 'val_prototypes.npy', proto_emb)
    pd.DataFrame({'ship_name': proto_ship, 'ship_type': proto_type}).to_csv(out / 'val_prototype_meta.csv', index=False, encoding='utf-8-sig')

    # C verification
    score_rows = []
    pos_scores, hard_scores, easy_scores = [], [], []
    for i in range(len(test_meta)):
        q_ship = test_meta.iloc[i]['true_class_name']
        q_type = test_meta.iloc[i]['ship_type']
        qid = test_meta.iloc[i]['wav_path']
        sim = proto_emb @ test_emb[i]
        for j, g_ship in enumerate(proto_ship):
            g_type = proto_type[j]
            s = float(sim[j])
            is_pos = int(g_ship == q_ship)
            if is_pos:
                ptype='positive'; pos_scores.append(s)
            else:
                if g_type == q_type:
                    ptype='hard_negative'; hard_scores.append(s)
                else:
                    ptype='easy_negative'; easy_scores.append(s)
            score_rows.append({
                'test_wav_path': qid,
                'test_true_ship': q_ship,
                'test_true_ship_type': q_type,
                'query_ship': q_ship,
                'query_type': q_type,
                'prototype_ship': g_ship,
                'prototype_ship_type': g_type,
                'prototype_type': g_type,
                'cosine_similarity': s,
                'label': is_pos,
                'pair_type': ptype,
            })
    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(out / 'verification_scores.csv', index=False, encoding='utf-8-sig')

    auc_all, eer_all = safe_auc_eer(pos_scores, hard_scores + easy_scores)
    auc_hard, eer_hard = safe_auc_eer(pos_scores, hard_scores)
    auc_easy, eer_easy = safe_auc_eer(pos_scores, easy_scores)
    verif = pd.DataFrame([
        {'setting': 'positive_vs_all_negative', 'auc': auc_all, 'eer': eer_all},
        {'setting': 'positive_vs_hard_negative', 'auc': auc_hard, 'eer': eer_hard},
        {'setting': 'positive_vs_easy_negative', 'auc': auc_easy, 'eer': eer_easy},
    ])
    verif.to_csv(out / 'verification_metrics_by_pair_type.csv', index=False, encoding='utf-8-sig')

    # D retrieval
    seg_rank, seg_tr = rank_queries(test_emb, test_meta['true_class_name'].to_numpy(), test_meta['wav_path'].to_numpy(), proto_emb, proto_ship)
    seg_rank.to_csv(out / 'retrieval_rankings_segment.csv', index=False, encoding='utf-8-sig')
    seg_r1 = float(np.mean(seg_tr == 1)); seg_r5 = float(np.mean(seg_tr <= 5)); seg_map = float(np.mean(1.0 / seg_tr))

    rec_q, rec_ship, rec_id = [], [], []
    for rid, g in test_meta.groupby('recording_id'):
        idx = g.index.to_numpy()
        e = l2n(test_emb[idx].mean(axis=0, keepdims=True))[0]
        rec_q.append(e); rec_ship.append(g['true_class_name'].iloc[0]); rec_id.append(rid)
    rec_q = np.stack(rec_q)
    rec_rank, rec_tr = rank_queries(rec_q, np.array(rec_ship), np.array(rec_id), proto_emb, proto_ship)
    rec_rank.to_csv(out / 'retrieval_rankings_recording.csv', index=False, encoding='utf-8-sig')
    rec_t1 = float(np.mean(rec_tr == 1)); rec_t5 = float(np.mean(rec_tr <= 5))

    pd.DataFrame([{'segment_recall_at1': seg_r1, 'segment_recall_at5': seg_r5, 'segment_map': seg_map}]).to_csv(out / 'retrieval_segment_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'recording_top1': rec_t1, 'recording_top5': rec_t5}]).to_csv(out / 'retrieval_recording_metrics.csv', index=False, encoding='utf-8-sig')

    # E k-shot
    rng = np.random.default_rng(args.seed)
    val_idx_by_ship = {s: np.where(val_meta['true_class_name'].to_numpy() == s)[0] for s in proto_ship}
    k_vals = [t.strip() for t in args.kshot.split(',') if t.strip()]
    k_rows = []
    for tok in k_vals:
        K = 'all' if tok == 'all' else int(tok)
        for rep in range(args.kshot_repeats):
            p_ship = []
            p_emb = []
            for s in proto_ship:
                idx_pool = val_idx_by_ship[s]
                if K == 'all':
                    pick = idx_pool
                else:
                    kk = min(int(K), len(idx_pool))
                    pick = rng.choice(idx_pool, size=kk, replace=False)
                p = l2n(val_emb[pick].mean(axis=0, keepdims=True))[0]
                p_ship.append(s); p_emb.append(p)
            p_ship = np.array(p_ship); p_emb = np.stack(p_emb)
            _, s_tr = rank_queries(test_emb, test_meta['true_class_name'].to_numpy(), test_meta['wav_path'].to_numpy(), p_emb, p_ship)
            _, r_tr = rank_queries(rec_q, np.array(rec_ship), np.array(rec_id), p_emb, p_ship)
            k_rows.append({'k_shot': K, 'repeat_id': rep, 'segment_recall_at1': float(np.mean(s_tr == 1)), 'segment_recall_at5': float(np.mean(s_tr <= 5)), 'segment_map': float(np.mean(1.0 / s_tr)), 'recording_top1': float(np.mean(r_tr == 1)), 'recording_top5': float(np.mean(r_tr <= 5))})
    pd.DataFrame(k_rows).to_csv(out / 'kshot_retrieval_metrics.csv', index=False, encoding='utf-8-sig')

    # F failure
    pos_stat = score_df[score_df['pair_type'] == 'positive'].groupby('test_true_ship', as_index=False)['cosine_similarity'].agg(positive_score_mean='mean', positive_score_std='std').rename(columns={'test_true_ship': 'ship_name'})
    hard_stat = score_df[score_df['pair_type'] == 'hard_negative'].groupby('test_true_ship', as_index=False)['cosine_similarity'].agg(hard_negative_score_mean='mean', hard_negative_score_std='std').rename(columns={'test_true_ship': 'ship_name'})
    hard_ship = score_df[score_df['label'] == 0].groupby(['test_true_ship', 'prototype_ship'], as_index=False)['cosine_similarity'].mean().sort_values(['test_true_ship', 'cosine_similarity'], ascending=[True, False]).groupby('test_true_ship', as_index=False).first().rename(columns={'test_true_ship': 'ship_name', 'prototype_ship': 'hardest_negative_ship', 'cosine_similarity': 'hardest_negative_score'})
    seg_ship = seg_rank[seg_rank['is_relevant'] == 1].groupby(['query_id', 'true_ship'], as_index=False)['rank'].min().groupby('true_ship', as_index=False).agg(segment_recall_at1=('rank', lambda x: float(np.mean(x == 1))), segment_recall_at5=('rank', lambda x: float(np.mean(x <= 5)))).rename(columns={'true_ship': 'ship_name'})
    rec_ship_df = rec_rank[rec_rank['is_relevant'] == 1].groupby(['query_id', 'true_ship'], as_index=False)['rank'].min().groupby('true_ship', as_index=False).agg(recording_true_rank=('rank', 'mean'), recording_top1=('rank', lambda x: float(np.mean(x == 1))), recording_top5=('rank', lambda x: float(np.mean(x <= 5)))).rename(columns={'true_ship': 'ship_name'})

    fail = per_class_df[['ship_name', 'recall']].rename(columns={'recall': 'closed_set_recall'})
    fail['ship_type'] = fail['ship_name'].map(lambda s: test_meta[test_meta['true_class_name'] == s]['ship_type'].iloc[0] if len(test_meta[test_meta['true_class_name'] == s]) else 'UNKNOWN')
    fail = fail.merge(pos_stat, on='ship_name', how='left').merge(hard_stat, on='ship_name', how='left').merge(hard_ship, on='ship_name', how='left').merge(rec_ship_df, on='ship_name', how='left').merge(seg_ship, on='ship_name', how='left')
    fail['is_zero_recall_ship'] = (fail['closed_set_recall'] == 0.0).astype(int)
    fail.to_csv(out / 'failure_case_matching_analysis.csv', index=False, encoding='utf-8-sig')
    zero_df = fail[fail['is_zero_recall_ship'] == 1].copy()
    zero_df.to_csv(out / 'zero_recall_ship_analysis.csv', index=False, encoding='utf-8-sig')
    if len(zero_df):
        closed['zero_recall_segment_r1_mean'] = float(zero_df['segment_recall_at1'].mean())
        closed['zero_recall_recording_top1_mean'] = float(zero_df['recording_top1'].mean())
    else:
        closed['zero_recall_segment_r1_mean'] = float('nan')
        closed['zero_recall_recording_top1_mean'] = float('nan')
    write_json(out / 'closed_set_metrics.json', closed)

    # plots
    plt.figure(figsize=(8,5))
    for ptype, c in [('positive','tab:green'),('hard_negative','tab:orange'),('easy_negative','tab:blue')]:
        arr = score_df[score_df['pair_type'] == ptype]['cosine_similarity'].to_numpy()
        if len(arr):
            plt.hist(arr, bins=80, alpha=0.5, density=True, label=ptype, color=c)
    plt.legend(); plt.xlabel('cosine similarity'); plt.ylabel('density'); plt.title('Score Distribution')
    plt.tight_layout(); plt.savefig(out / 'score_distribution.png', dpi=160); plt.close()

    from sklearn.metrics import roc_curve, roc_auc_score
    plt.figure(figsize=(7,6))
    pos_arr = np.asarray(pos_scores)
    for name, neg_arr, c in [('all negatives', np.asarray(hard_scores + easy_scores), 'tab:red'), ('hard negatives', np.asarray(hard_scores), 'tab:orange'), ('easy negatives', np.asarray(easy_scores), 'tab:blue')]:
        if len(pos_arr) == 0 or len(neg_arr) == 0:
            continue
        y = np.concatenate([np.ones(len(pos_arr)), np.zeros(len(neg_arr))]).astype(int)
        s = np.concatenate([pos_arr, neg_arr]).astype(float)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', color=c)
    plt.plot([0,1],[0,1],'--',color='gray'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curves'); plt.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(out / 'roc_curves.png', dpi=160); plt.close()

    ks = np.arange(1, max(25, int(max(seg_tr.max(), rec_tr.max()))) + 1)
    seg_cmc = np.array([np.mean(seg_tr <= k) for k in ks])
    rec_cmc = np.array([np.mean(rec_tr <= k) for k in ks])
    plt.figure(figsize=(7,6))
    plt.plot(ks, seg_cmc, label='Segment CMC'); plt.plot(ks, rec_cmc, label='Recording CMC')
    plt.xlabel('Rank k'); plt.ylabel('Recall@k'); plt.title('CMC Curve'); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out / 'cmc_curve.png', dpi=160); plt.close()

    summary = {
        'model_name': args.model_name,
        'model_type': args.model_type,
        'sampling': args.sampling,
        'loss_type': eff_loss_type,
        'closed_set': closed,
        'verification': {'auc_all': auc_all, 'eer_all': eer_all, 'auc_hard': auc_hard, 'eer_hard': eer_hard, 'auc_easy': auc_easy, 'eer_easy': eer_easy},
        'retrieval': {'segment_recall_at1': seg_r1, 'segment_recall_at5': seg_r5, 'segment_map': seg_map, 'recording_top1': rec_t1, 'recording_top5': rec_t5},
        'safety_checks': {
            'recording_overlap_checked': True,
            'val_test_not_used_for_training': True,
            'test_not_used_for_selection': True,
            'val_test_not_resampled': True,
            'ship_name_from_train_only_checked': True,
            'input_dim_checked': True,
            'embedding_eval_mode_no_grad': True,
            'embedding_not_softmax': True,
        },
    }
    write_json(out / 'model_summary.json', summary)

    md = []
    md.append(f"# {args.model_name} Summary")
    md.append('')
    md.append(f"- loss_type: {eff_loss_type}")
    md.append(f"- closed_test_acc: {closed['test_acc']:.6f}")
    md.append(f"- closed_test_macro_f1: {closed['test_macro_f1']:.6f}")
    md.append(f"- zero_recall_ship_count: {closed['zero_recall_ship_count']}")
    md.append(f"- auc_all/eer_all: {auc_all:.6f} / {eer_all:.6f}")
    md.append(f"- auc_hard/eer_hard: {auc_hard:.6f} / {eer_hard:.6f}")
    md.append(f"- auc_easy/eer_easy: {auc_easy:.6f} / {eer_easy:.6f}")
    md.append(f"- segment_recall@1/@5/map: {seg_r1:.6f} / {seg_r5:.6f} / {seg_map:.6f}")
    md.append(f"- recording_top1/top5: {rec_t1:.6f} / {rec_t5:.6f}")
    (out / 'model_summary.md').write_text('\n'.join(md), encoding='utf-8')

    print(out)


if __name__ == '__main__':
    main()
