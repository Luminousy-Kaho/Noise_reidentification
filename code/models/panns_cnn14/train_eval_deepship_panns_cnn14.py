import argparse
import csv
import importlib.util
import json
import random
import sys
import types
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = Path(__file__).resolve().parent
SITE_PACKAGES = Path(sys.executable).resolve().parent / "Lib" / "site-packages"
PANNS_DIR = SITE_PACKAGES / "panns_inference"


def load_panns_cnn14_class():
    package_name = "_local_panns_inference"
    pkg = types.ModuleType(package_name)
    pkg.__path__ = [str(PANNS_DIR)]
    sys.modules[package_name] = pkg

    util_spec = importlib.util.spec_from_file_location(
        f"{package_name}.pytorch_utils", PANNS_DIR / "pytorch_utils.py"
    )
    util_mod = importlib.util.module_from_spec(util_spec)
    sys.modules[util_spec.name] = util_mod
    util_spec.loader.exec_module(util_mod)

    model_spec = importlib.util.spec_from_file_location(f"{package_name}.models", PANNS_DIR / "models.py")
    model_mod = importlib.util.module_from_spec(model_spec)
    sys.modules[model_spec.name] = model_mod
    model_spec.loader.exec_module(model_mod)
    return model_mod.Cnn14


Cnn14 = load_panns_cnn14_class()


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def localize_path(path_text: str, data_root: Path) -> Path:
    p = Path(path_text)
    if p.exists():
        return p
    markers = ["数据集", "鏁版嵁闆"]
    text = str(p)
    for marker in markers:
        if marker in text:
            rel = text.split(marker, 1)[1].lstrip("\\/")
            return data_root / rel


def read_wav(path: Path, target_sr: int, target_len: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, data = wavfile.read(str(path))
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), abs(info.max))
        data = data.astype(np.float32) / float(scale)
    else:
        data = data.astype(np.float32)
    if sr != target_sr:
        data = scipy.signal.resample_poly(data, target_sr, sr).astype(np.float32)
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)))
    elif len(data) > target_len:
        data = data[:target_len]
    return data


def load_split(split_json: Path, data_root: Path):
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    rows = []
    for split_name in ("train", "val", "test"):
        for row in payload[split_name]:
            item = dict(row)
            item["split"] = split_name
            item["local_wav_path"] = str(localize_path(item["wav_path"], data_root))
            rows.append(item)
    return pd.DataFrame(rows)


class DeepShipWaveDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, class_to_idx: dict, sample_rate=32000, clip_sec=5.0):
        self.rows = rows.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * clip_sec)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        wav = read_wav(Path(row["local_wav_path"]), self.sample_rate, self.target_len)
        y = torch.tensor(self.class_to_idx[str(row["class_name"])], dtype=torch.long)
        return (
            torch.from_numpy(wav).float(),
            y,
            str(row["wav_path"]),
            str(row["class_name"]),
            str(row["source_long_id"]),
        )


def make_loaders(rows, class_to_idx, batch_size, num_workers):
    loaders = []
    for split_name in ("train", "val", "test"):
        ds = DeepShipWaveDataset(rows[rows["split"] == split_name], class_to_idx)
        loaders.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                pin_memory=True,
            )
        )
    return loaders


class Cnn14Classifier(nn.Module):
    def __init__(self, classes_num: int):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        self.classifier = nn.Linear(2048, classes_num)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def load_audioset_checkpoint(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        return missing, unexpected

    def forward(self, x):
        out = self.backbone(x)
        emb = out["embedding"]
        logits = self.classifier(emb)
        return logits, emb


def l2_normalize(x, axis=1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def evaluate_closed_set(model, loader, device, class_names, out_csv=None):
    model.eval()
    rows, y_true, y_pred = [], [], []
    with torch.no_grad():
        for x, y, wav_paths, true_class_names, source_ids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            for i in range(y.size(0)):
                pred_i = int(pred[i].detach().cpu().item())
                true_i = int(y[i].detach().cpu().item())
                rows.append(
                    {
                        "wav_path": wav_paths[i],
                        "source_long_id": source_ids[i],
                        "true_label": true_i,
                        "pred_label": pred_i,
                        "true_class_name": true_class_names[i],
                        "pred_class_name": class_names[pred_i],
                        "pred_confidence": float(probs[i, pred_i].detach().cpu().item()),
                    }
                )
                y_true.append(true_i)
                y_pred.append(pred_i)
    if out_csv is not None:
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class_idx": list(range(len(class_names))),
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "zero_recall_ship_count": int((per_class["recall"] == 0).sum()),
    }, per_class


def extract_embeddings(model, loader, device):
    model.eval()
    rows, embs = [], []
    with torch.no_grad():
        for x, y, wav_paths, true_class_names, source_ids in loader:
            x = x.to(device, non_blocking=True)
            _, emb = model(x)
            emb = emb.detach().cpu().numpy()
            for i in range(len(wav_paths)):
                rows.append(
                    {
                        "wav_path": wav_paths[i],
                        "source_long_id": source_ids[i],
                        "true_label": int(y[i].item()),
                        "true_class_name": true_class_names[i],
                    }
                )
                embs.append(emb[i])
    return pd.DataFrame(rows), l2_normalize(np.asarray(embs, dtype=np.float32))


def load_ship_to_big(split_csv: Path):
    df = pd.read_csv(split_csv)
    out = {}
    for _, row in df.iterrows():
        ship = str(row["class_name"])
        if ship in out:
            continue
        rel = str(row.get("source_wav_relpath", ""))
        out[ship] = rel.replace("\\", "/").split("/")[0] if rel and rel != "nan" else "UNKNOWN"
    return out


def build_prototypes(val_meta, val_emb, ship_to_big):
    rows, protos = [], []
    for ship in sorted(val_meta["true_class_name"].unique()):
        idx = np.flatnonzero(val_meta["true_class_name"].to_numpy() == ship)
        proto = val_emb[idx].mean(axis=0)
        proto = proto / max(np.linalg.norm(proto), 1e-12)
        rows.append({"ship_name": ship, "big_class": ship_to_big.get(ship, "UNKNOWN"), "val_segment_count": int(len(idx))})
        protos.append(proto)
    return pd.DataFrame(rows), np.stack(protos)


def pairwise_scores(test_meta, test_emb, proto_meta, proto_emb):
    rows = []
    ships = proto_meta["ship_name"].tolist()
    bigs = dict(zip(proto_meta["ship_name"], proto_meta["big_class"]))
    for i, row in test_meta.iterrows():
        q_ship = str(row["true_class_name"])
        q_big = bigs.get(q_ship, "UNKNOWN")
        sims = np.dot(proto_emb, test_emb[i])
        for j, p_ship in enumerate(ships):
            label = int(p_ship == q_ship)
            if label:
                pair_type = "positive"
            elif bigs.get(p_ship, "UNKNOWN") == q_big:
                pair_type = "hard_negative"
            else:
                pair_type = "easy_negative"
            rows.append(
                {
                    "query_id": row["wav_path"],
                    "source_long_id": row["source_long_id"],
                    "test_true_ship": q_ship,
                    "test_true_big_class": q_big,
                    "prototype_ship": p_ship,
                    "prototype_big_class": bigs.get(p_ship, "UNKNOWN"),
                    "cosine_similarity": float(sims[j]),
                    "label": label,
                    "pair_type": pair_type,
                }
            )
    return pd.DataFrame(rows)


def verification_metrics(score_df):
    pair_sets = {"all": ["positive", "hard_negative", "easy_negative"], "hard": ["positive", "hard_negative"], "easy": ["positive", "easy_negative"]}
    pos_mean = float(score_df.loc[score_df["pair_type"] == "positive", "cosine_similarity"].mean())
    hard_mean = float(score_df.loc[score_df["pair_type"] == "hard_negative", "cosine_similarity"].mean())
    easy_mean = float(score_df.loc[score_df["pair_type"] == "easy_negative", "cosine_similarity"].mean())
    rows = []
    for name, allowed in pair_sets.items():
        sub = score_df[score_df["pair_type"].isin(allowed)]
        y_true = sub["label"].to_numpy(dtype=int)
        y_score = sub["cosine_similarity"].to_numpy(dtype=float)
        rows.append(
            {
                "pair_set": name,
                "num_pairs": int(len(sub)),
                "AUC": float(roc_auc_score(y_true, y_score)),
                "EER": compute_eer(y_true, y_score),
                "positive_mean_similarity": pos_mean,
                "hard_negative_mean_similarity": hard_mean,
                "easy_negative_mean_similarity": easy_mean,
                "hard_negative_gap": pos_mean - hard_mean,
                "easy_negative_gap": pos_mean - easy_mean,
            }
        )
    return pd.DataFrame(rows)


def retrieval(query_meta, query_emb, proto_meta, proto_emb, query_type):
    ships = proto_meta["ship_name"].tolist()
    rankings, ranks = [], []
    for i, row in query_meta.reset_index(drop=True).iterrows():
        true_ship = str(row["true_class_name"])
        qid = str(row["source_long_id"] if query_type == "recording" else row["wav_path"])
        sims = np.dot(proto_emb, query_emb[i])
        order = np.argsort(-sims)
        true_rank = len(order) + 1
        for rank, idx in enumerate(order, start=1):
            ok = int(ships[idx] == true_ship)
            if ok and true_rank == len(order) + 1:
                true_rank = rank
            rankings.append({"query_type": query_type, "query_id": qid, "true_ship": true_ship, "rank": rank, "prototype_ship": ships[idx], "cosine_similarity": float(sims[idx]), "is_correct_match": ok})
        ranks.append(true_rank)
    ranks = np.asarray(ranks, dtype=float)
    return pd.DataFrame(rankings), {
        "query_type": query_type,
        "query_count": int(len(ranks)),
        "Recall@1": float(np.mean(ranks <= 1)),
        "Recall@5": float(np.mean(ranks <= 5)),
        "mAP": float(np.mean(1.0 / ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
        "mean_true_rank": float(np.mean(ranks)),
    }


def aggregate_recordings(test_meta, test_emb):
    rows, embs = [], []
    for (source_id, ship), idx in test_meta.groupby(["source_long_id", "true_class_name"]).indices.items():
        idx = list(idx)
        emb = test_emb[idx].mean(axis=0)
        emb = emb / max(np.linalg.norm(emb), 1e-12)
        rows.append({"source_long_id": source_id, "true_class_name": ship, "segment_count": len(idx)})
        embs.append(emb)
    return pd.DataFrame(rows), np.stack(embs)


def plot_scores(score_df, out_png):
    plt.figure(figsize=(8, 5))
    for pair_type, color in [("positive", "tab:green"), ("hard_negative", "tab:orange"), ("easy_negative", "tab:blue")]:
        vals = score_df.loc[score_df["pair_type"] == pair_type, "cosine_similarity"].to_numpy()
        if len(vals):
            plt.hist(vals, bins=80, density=True, alpha=0.45, label=pair_type, color=color)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def run_eval(model, val_dl, test_dl, device, class_names, split_csv, exp_dir, summary_extra):
    val_m, val_pc = evaluate_closed_set(model, val_dl, device, class_names, exp_dir / "val_predictions.csv")
    test_m, test_pc = evaluate_closed_set(model, test_dl, device, class_names, exp_dir / "test_predictions.csv")
    val_pc.to_csv(exp_dir / "val_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    test_pc.to_csv(exp_dir / "test_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    val_meta, val_emb = extract_embeddings(model, val_dl, device)
    test_meta, test_emb = extract_embeddings(model, test_dl, device)
    np.save(exp_dir / "val_embeddings.npy", val_emb)
    np.save(exp_dir / "test_embeddings.npy", test_emb)
    val_meta.to_csv(exp_dir / "val_embedding_meta.csv", index=False, encoding="utf-8-sig")
    test_meta.to_csv(exp_dir / "test_embedding_meta.csv", index=False, encoding="utf-8-sig")
    proto_meta, proto_emb = build_prototypes(val_meta, val_emb, load_ship_to_big(split_csv))
    np.save(exp_dir / "val_prototypes.npy", proto_emb)
    proto_meta.to_csv(exp_dir / "val_prototype_meta.csv", index=False, encoding="utf-8-sig")
    score_df = pairwise_scores(test_meta, test_emb, proto_meta, proto_emb)
    score_df.to_csv(exp_dir / "verification_scores.csv", index=False, encoding="utf-8-sig")
    verif_df = verification_metrics(score_df)
    verif_df.to_csv(exp_dir / "verification_metrics_by_pair_type.csv", index=False, encoding="utf-8-sig")
    plot_scores(score_df, exp_dir / "score_distribution.png")
    seg_rank, seg_m = retrieval(test_meta, test_emb, proto_meta, proto_emb, "segment")
    rec_meta, rec_emb = aggregate_recordings(test_meta, test_emb)
    rec_rank, rec_m = retrieval(rec_meta, rec_emb, proto_meta, proto_emb, "recording")
    seg_rank.to_csv(exp_dir / "retrieval_segment_rankings.csv", index=False, encoding="utf-8-sig")
    rec_rank.to_csv(exp_dir / "retrieval_recording_rankings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([seg_m]).to_csv(exp_dir / "retrieval_segment_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([rec_m]).to_csv(exp_dir / "retrieval_recording_metrics.csv", index=False, encoding="utf-8-sig")
    verif = verif_df.set_index("pair_set").to_dict(orient="index")
    summary = {
        **summary_extra,
        "closed_set_val_accuracy": val_m["accuracy"],
        "closed_set_val_macro_F1": val_m["macro_f1"],
        "closed_set_test_accuracy": test_m["accuracy"],
        "closed_set_test_macro_F1": test_m["macro_f1"],
        "zero_recall_ship_count": test_m["zero_recall_ship_count"],
        "zero_recall_ship_names": ";".join(test_pc.loc[test_pc["recall"] == 0, "class_name"].tolist()),
        "AUC_all": verif["all"]["AUC"],
        "AUC_hard": verif["hard"]["AUC"],
        "AUC_easy": verif["easy"]["AUC"],
        "EER_all": verif["all"]["EER"],
        "EER_hard": verif["hard"]["EER"],
        "EER_easy": verif["easy"]["EER"],
        "positive_mean_similarity": verif["all"]["positive_mean_similarity"],
        "hard_negative_mean_similarity": verif["all"]["hard_negative_mean_similarity"],
        "easy_negative_mean_similarity": verif["all"]["easy_negative_mean_similarity"],
        "hard_negative_gap": verif["all"]["hard_negative_gap"],
        "easy_negative_gap": verif["all"]["easy_negative_gap"],
        "Segment_Recall@1": seg_m["Recall@1"],
        "Segment_Recall@5": seg_m["Recall@5"],
        "Segment_mAP": seg_m["mAP"],
        "Recording_Top1": rec_m["Recall@1"],
        "Recording_Top5": rec_m["Recall@5"],
        "Recording_mAP_or_MRR": rec_m["MRR"],
    }
    pd.DataFrame([summary]).to_csv(exp_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    (exp_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        f"# {summary['model_name']} DeepShip Evaluation",
        "",
        "- Split: DeepShip 25-class min5 median recording-wise split.",
        "- Training loss: CrossEntropyLoss.",
        "- Downstream matching: validation prototypes, test queries.",
        "",
        f"- Test macro-F1: {summary['closed_set_test_macro_F1']:.6f}",
        f"- AUC-hard: {summary['AUC_hard']:.6f}",
        f"- Segment mAP/MRR: {summary['Segment_mAP']:.6f}",
        f"- Recording Top-1: {summary['Recording_Top1']:.6f}",
        f"- Recording Top-5: {summary['Recording_Top5']:.6f}",
    ]
    (exp_dir / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def train(args):
    set_seed(args.seed)
    split_json = Path(args.split_json)
    rows = load_split(split_json, Path(args.data_root))
    class_names = sorted(rows["class_name"].unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    train_dl, val_dl, test_dl = make_loaders(rows, class_to_idx, args.batch_size, args.num_workers)
    experiment_name = {
        "pretrained_audioset": "ce_audioset_pretrained",
        "random_init": "ce_random_init",
    }[args.init_type]
    out_root = Path(args.output_root) / experiment_name / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    exp_dir = out_root / now_str()
    exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cnn14Classifier(len(class_names))
    pretrained_note = "none"
    if args.init_type == "pretrained_audioset":
        missing, unexpected = model.load_audioset_checkpoint(Path(args.pretrained_checkpoint))
        pretrained_note = str(Path(args.pretrained_checkpoint).resolve())
        (exp_dir / "pretrained_load_report.json").write_text(
            json.dumps({"missing": missing, "unexpected": unexpected}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))
    best_val_f1 = -1.0
    best_path = exp_dir / "best_model.pt"
    train_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for bi, (x, y, *_rest) in enumerate(train_dl, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu().item()) * y.size(0)
            n += y.size(0)
            if bi == 1 or bi % args.print_every_n_batch == 0 or bi == len(train_dl):
                print(f"[{args.init_type}] epoch {epoch:03d}/{args.epochs:03d} batch {bi}/{len(train_dl)} loss={loss_sum/max(1,n):.6f}", flush=True)
        scheduler.step()
        val_m, _ = evaluate_closed_set(model, val_dl, device, class_names)
        row = {"epoch": epoch, "train_loss": loss_sum / max(1, n), "val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"]}
        train_rows.append(row)
        pd.DataFrame(train_rows).to_csv(exp_dir / "train_log.csv", index=False, encoding="utf-8-sig")
        torch.save(
            {
                "model": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "init_type": args.init_type,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "val_macro_f1": val_m["macro_f1"],
                "best_val_macro_f1": best_val_f1,
            },
            exp_dir / "latest_model.pt",
        )
        print(f"[{args.init_type}] epoch {epoch:03d} val_macro_f1={val_m['macro_f1']:.6f}", flush=True)
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_names": class_names,
                    "best_val_macro_f1": best_val_f1,
                    "init_type": args.init_type,
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                best_path,
            )
            print(f"saved best: {best_path}", flush=True)
    (exp_dir / "run_params.json").write_text(json.dumps({**vars(args), "pretrained_checkpoint_resolved": pretrained_note, "best_val_macro_f1": best_val_f1}, ensure_ascii=False, indent=2), encoding="utf-8")
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    run_eval(
        model,
        val_dl,
        test_dl,
        device,
        class_names,
        Path(args.split_csv),
        exp_dir,
        {
            "model_name": "PANNs-CNN14-AudioSet" if args.init_type == "pretrained_audioset" else "PANNs-CNN14-random-init",
            "init_type": args.init_type,
            "pretrained_checkpoint": pretrained_note,
            "best_val_macro_f1": best_val_f1,
        },
    )
    print(exp_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-type", choices=["pretrained_audioset", "random_init"], required=True)
    ap.add_argument(
        "--split-json",
        default=str(PROJECT_ROOT / "split_protocols" / "deepship" / "ship_identity" / "min5_median_split_20260511_163606" / "split_detail.json"),
    )
    ap.add_argument(
        "--split-csv",
        default=str(PROJECT_ROOT / "split_protocols" / "deepship" / "ship_identity" / "min5_median_split_20260511_163606" / "split_detail.csv"),
    )
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pretrained-checkpoint", default=str(TASK_ROOT / "pretrained_weights" / "Cnn14_mAP=0.431.pth"))
    ap.add_argument("--output-root", default=str(TASK_ROOT / "recordingwise_min5_median"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--print-every-n-batch", type=int, default=50)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()

