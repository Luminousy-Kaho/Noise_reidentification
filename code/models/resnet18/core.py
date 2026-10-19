import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchaudio


@dataclass
class AudioItem:
    wav_path: Path
    class_name: str
    label: int
    duration_sec: float
    source_long_id: str


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_dataset(dataset_root: Path) -> Tuple[List[AudioItem], List[str], Dict[str, int]]:
    class_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    items: List[AudioItem] = []
    for cdir in class_dirs:
        cname = cdir.name
        label = class_to_idx[cname]
        for wav in sorted(cdir.glob("*.wav")):
            info = torchaudio.info(str(wav))
            dur = info.num_frames / float(info.sample_rate)
            stem = wav.stem
            source_long_id = stem.split("__seg")[0] if "__seg" in stem else stem
            items.append(
                AudioItem(
                    wav_path=wav,
                    class_name=cname,
                    label=label,
                    duration_sec=dur,
                    source_long_id=source_long_id,
                )
            )
    return items, class_names, class_to_idx


def _assign_by_duration(items: List[AudioItem], ratios=(0.8, 0.1, 0.1), seed=42):
    rng = random.Random(seed)
    by_class: Dict[str, List[AudioItem]] = {}
    for x in items:
        by_class.setdefault(x.class_name, []).append(x)
    split = {"train": [], "val": [], "test": []}
    for cname, arr in by_class.items():
        rng.shuffle(arr)
        total = sum(a.duration_sec for a in arr)
        tgt_train = total * ratios[0]
        tgt_val = total * ratios[1]
        cur_train, cur_val = 0.0, 0.0
        for a in arr:
            if cur_train < tgt_train:
                split["train"].append(a)
                cur_train += a.duration_sec
            elif cur_val < tgt_val:
                split["val"].append(a)
                cur_val += a.duration_sec
            else:
                split["test"].append(a)
    return split


def split_random_clip(items: List[AudioItem], ratios=(0.8, 0.1, 0.1), seed=42):
    return _assign_by_duration(items, ratios=ratios, seed=seed)


def split_grouped_by_longwav(items: List[AudioItem], ratios=(0.8, 0.1, 0.1), seed=42):
    rng = random.Random(seed)
    by_class: Dict[str, List[AudioItem]] = {}
    for x in items:
        by_class.setdefault(x.class_name, []).append(x)

    split = {"train": [], "val": [], "test": []}
    for cname, arr in by_class.items():
        groups: Dict[str, List[AudioItem]] = {}
        for a in arr:
            groups.setdefault(a.source_long_id, []).append(a)
        group_list = list(groups.items())
        rng.shuffle(group_list)
        group_list.sort(key=lambda kv: -sum(x.duration_sec for x in kv[1]))

        total = sum(a.duration_sec for a in arr)
        target = {"train": total * ratios[0], "val": total * ratios[1], "test": total * ratios[2]}
        cur = {"train": 0.0, "val": 0.0, "test": 0.0}

        for _, gitems in group_list:
            gdur = sum(x.duration_sec for x in gitems)
            best_set = None
            best_score = None
            for s in ("train", "val", "test"):
                new_cur = cur[s] + gdur
                score = abs(new_cur - target[s]) + 0.01 * (cur[s] / (target[s] + 1e-9))
                if best_score is None or score < best_score:
                    best_score = score
                    best_set = s
            split[best_set].extend(gitems)
            cur[best_set] += gdur
    return split


def dump_split(split: Dict[str, List[AudioItem]], out_json: Path, out_csv: Path):
    payload = {}
    rows = []
    for s in ("train", "val", "test"):
        payload[s] = []
        for x in split[s]:
            rec = {
                "split": s,
                "wav_path": str(x.wav_path),
                "class_name": x.class_name,
                "label": x.label,
                "duration_sec": x.duration_sec,
                "source_long_id": x.source_long_id,
            }
            payload[s].append(rec)
            rows.append(rec)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["split", "wav_path", "class_name", "label", "duration_sec", "source_long_id"],
        )
        w.writeheader()
        w.writerows(rows)


def load_split(split_json: Path):
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    split = {"train": [], "val": [], "test": []}
    for s in split:
        for r in payload[s]:
            split[s].append(
                AudioItem(
                    wav_path=Path(r["wav_path"]),
                    class_name=r["class_name"],
                    label=int(r["label"]),
                    duration_sec=float(r["duration_sec"]),
                    source_long_id=r["source_long_id"],
                )
            )
    return split


class LogMelDataset(Dataset):
    def __init__(
        self,
        items: List[AudioItem],
        sample_rate=16000,
        clip_sec=5.0,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=128,
        fmin=20.0,
        fmax=8000.0,
    ):
        self.items = items
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * clip_sec)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
        )
        self.ampto_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav, sr = torchaudio.load(str(item.wav_path))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.shape[1] < self.target_samples:
            pad = self.target_samples - wav.shape[1]
            wav = F.pad(wav, (0, pad))
        elif wav.shape[1] > self.target_samples:
            wav = wav[:, : self.target_samples]

        mel = self.mel(wav)  # [1, n_mels, t]
        mel_db = self.ampto_db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        mel_db = F.interpolate(mel_db.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        x = mel_db.repeat(3, 1, 1)  # [3, 224, 224]
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y, str(item.wav_path), item.class_name


class PrecomputedFeatureDataset(Dataset):
    def __init__(self, items: List[AudioItem], feature_root: Path):
        self.items = items
        self.feature_root = feature_root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        rel = item.wav_path.parent.name + "/" + item.wav_path.stem + ".pt"
        fpath = self.feature_root / rel
        x = torch.load(fpath, map_location="cpu")
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y, str(item.wav_path), item.class_name


class ShardedFeatureDataset(Dataset):
    def __init__(self, items: List[AudioItem], shard_root: Path):
        self.items = items
        self.shard_root = shard_root
        idx_csv = shard_root / "feature_shard_index.csv"
        if not idx_csv.exists():
            raise FileNotFoundError(f"Missing shard index: {idx_csv}")
        mapping = {}
        with idx_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                mapping[r["wav_path"]] = (r["shard_name"], int(r["index_in_shard"]))
        self.mapping = mapping
        self._cache_name = None
        self._cache_arr = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        key = str(item.wav_path)
        if key not in self.mapping:
            raise KeyError(f"wav_path not in shard index: {key}")
        shard_name, inner_idx = self.mapping[key]
        if self._cache_name != shard_name:
            self._cache_arr = np.load(self.shard_root / shard_name, mmap_mode="r")
            self._cache_name = shard_name
        x_np = self._cache_arr[inner_idx]
        # Make a writable copy to avoid PyTorch warning on readonly memmap slices.
        x = torch.from_numpy(np.array(x_np, copy=True)).float()
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y, key, item.class_name


def build_model(num_classes: int, use_imagenet_pretrain: bool = True):
    if use_imagenet_pretrain:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def suggest_batch_size() -> int:
    if not torch.cuda.is_available():
        return 8
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if mem_gb >= 20:
        return 128
    if mem_gb >= 12:
        return 64
    if mem_gb >= 8:
        return 32
    if mem_gb >= 6:
        return 8
    return 8


def make_loaders(
    split,
    batch_size,
    num_workers,
    feat_cfg,
    feature_root: Optional[Path] = None,
    shard_feature_root: Optional[Path] = None,
):
    if feature_root is None:
        if shard_feature_root is None:
            train_ds = LogMelDataset(split["train"], **feat_cfg)
            val_ds = LogMelDataset(split["val"], **feat_cfg)
            test_ds = LogMelDataset(split["test"], **feat_cfg)
        else:
            train_ds = ShardedFeatureDataset(split["train"], shard_root=shard_feature_root)
            val_ds = ShardedFeatureDataset(split["val"], shard_root=shard_feature_root)
            test_ds = ShardedFeatureDataset(split["test"], shard_root=shard_feature_root)
    else:
        train_ds = PrecomputedFeatureDataset(split["train"], feature_root=feature_root)
        val_ds = PrecomputedFeatureDataset(split["val"], feature_root=feature_root)
        test_ds = PrecomputedFeatureDataset(split["test"], feature_root=feature_root)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl


def _metrics_from_preds(y_true: List[int], y_pred: List[int], num_classes: int):
    n = len(y_true)
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    acc = sum(cm[i][i] for i in range(num_classes)) / max(1, n)

    per_class = []
    macro_p = macro_r = macro_f1 = 0.0
    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        per_class.append({"class_idx": i, "precision": p, "recall": r, "f1": f1})
        macro_p += p
        macro_r += r
        macro_f1 += f1
    macro_p /= num_classes
    macro_r /= num_classes
    macro_f1 /= num_classes
    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def evaluate(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred = [], []
    sample_rows = []
    with torch.no_grad():
        for x, y, paths, class_names in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().tolist()
            yt = y.tolist()
            y_true.extend(yt)
            y_pred.extend(pred)
            probs = torch.softmax(logits, dim=1).cpu()
            for i in range(len(paths)):
                sample_rows.append(
                    {
                        "wav_path": paths[i],
                        "true_label": yt[i],
                        "pred_label": pred[i],
                        "true_class_name": class_names[i],
                        "pred_confidence": float(probs[i, pred[i]].item()),
                    }
                )
    m = _metrics_from_preds(y_true, y_pred, num_classes)
    return m, sample_rows


def train_one_experiment(
    split,
    class_names,
    exp_dir: Path,
    feat_cfg: dict,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=None,
    num_workers=4,
    seed=42,
    feature_root: Optional[Path] = None,
    shard_feature_root: Optional[Path] = None,
    print_every_batch: bool = False,
    print_every_n_batch: int = 50,
    profile_batch_timing: bool = False,
    use_imagenet_pretrain: bool = True,
):
    torch.manual_seed(seed)
    random.seed(seed)
    exp_dir.mkdir(parents=True, exist_ok=True)

    log(f"Experiment dir: {exp_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = suggest_batch_size()
    log(f"Config: device={device}, batch_size={batch_size}, num_workers={num_workers}, epochs={epochs}")
    if torch.cuda.is_available():
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")
        log(f"CUDA memory total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    if feature_root is None and shard_feature_root is None:
        log("Feature mode: online extraction")
    elif feature_root is not None:
        log(f"Feature mode: offline precomputed pt ({feature_root})")
    else:
        log(f"Feature mode: offline sharded npy ({shard_feature_root})")
    log("Building dataloaders...")
    train_dl, val_dl, test_dl = make_loaders(
        split,
        batch_size,
        num_workers,
        feat_cfg,
        feature_root=feature_root,
        shard_feature_root=shard_feature_root,
    )
    log(f"Dataloaders ready: train_batches={len(train_dl)}, val_batches={len(val_dl)}, test_batches={len(test_dl)}")

    log("Building model...")
    log(f"Model init: use_imagenet_pretrain={use_imagenet_pretrain}")
    model = build_model(
        num_classes=len(class_names),
        use_imagenet_pretrain=use_imagenet_pretrain,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_f1 = -1.0
    best_path = exp_dir / "best_model.pt"
    train_log_rows = []
    batch_timing_rows = []
    live_log_path = exp_dir / "live_log.csv"
    with live_log_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_accuracy", "val_macro_f1"])
        w.writeheader()

    log("Start training loop...")
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        total_batches = len(train_dl)
        iter_start = perf_counter()
        for bi, (x, y, _, _) in enumerate(train_dl, start=1):
            data_wait_s = perf_counter() - iter_start
            t0 = perf_counter()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            h2d_s = perf_counter() - t0

            t1 = perf_counter()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_s = perf_counter() - t1
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
            step_total_s = data_wait_s + h2d_s + compute_s
            if profile_batch_timing:
                batch_timing_rows.append(
                    {
                        "epoch": ep,
                        "batch": bi,
                        "data_wait_s": data_wait_s,
                        "h2d_s": h2d_s,
                        "compute_s": compute_s,
                        "step_total_s": step_total_s,
                    }
                )
            if print_every_batch or bi == 1 or bi % max(1, print_every_n_batch) == 0 or bi == total_batches:
                cur_loss = loss_sum / max(1, n)
                print(
                    f"[Epoch {ep:03d}/{epochs:03d}] "
                    f"batch {bi}/{total_batches} "
                    f"running_train_loss={cur_loss:.6f} "
                    f"(wait={data_wait_s:.3f}s h2d={h2d_s:.3f}s compute={compute_s:.3f}s)",
                    flush=True,
                )
            iter_start = perf_counter()
        scheduler.step()

        train_loss = loss_sum / max(1, n)
        val_metrics, _ = evaluate(model, val_dl, device, len(class_names))
        train_log_rows.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        # Real-time progress output
        print(
            f"[Epoch {ep:03d}/{epochs:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_acc={val_metrics['accuracy']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.6f}",
            flush=True,
        )
        with live_log_path.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_accuracy", "val_macro_f1"])
            w.writerow(train_log_rows[-1])
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "class_names": class_names}, best_path)
            log(f"New best model saved at epoch {ep}: val_macro_f1={best_val_f1:.6f}")

    log("Training finished. Loading best model for final evaluation...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    val_metrics, val_rows = evaluate(model, val_dl, device, len(class_names))
    test_metrics, test_rows = evaluate(model, test_dl, device, len(class_names))

    with (exp_dir / "train_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_accuracy", "val_macro_f1"])
        w.writeheader()
        w.writerows(train_log_rows)
    if profile_batch_timing and batch_timing_rows:
        with (exp_dir / "batch_timing.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["epoch", "batch", "data_wait_s", "h2d_s", "compute_s", "step_total_s"],
            )
            w.writeheader()
            w.writerows(batch_timing_rows)

        # summarize bottleneck
        mean_wait = sum(r["data_wait_s"] for r in batch_timing_rows) / len(batch_timing_rows)
        mean_h2d = sum(r["h2d_s"] for r in batch_timing_rows) / len(batch_timing_rows)
        mean_compute = sum(r["compute_s"] for r in batch_timing_rows) / len(batch_timing_rows)
        bottleneck = "compute"
        if mean_wait >= mean_h2d and mean_wait >= mean_compute:
            bottleneck = "data_wait"
        elif mean_h2d >= mean_wait and mean_h2d >= mean_compute:
            bottleneck = "h2d_transfer"
        timing_summary = {
            "mean_data_wait_s": mean_wait,
            "mean_h2d_s": mean_h2d,
            "mean_compute_s": mean_compute,
            "bottleneck_guess": bottleneck,
        }
        (exp_dir / "timing_summary.json").write_text(
            json.dumps(timing_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def dump_rows(rows, path):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["wav_path", "true_label", "pred_label", "true_class_name", "pred_confidence"],
            )
            w.writeheader()
            w.writerows(rows)

    dump_rows(val_rows, exp_dir / "val_predictions.csv")
    dump_rows(test_rows, exp_dir / "test_predictions.csv")

    summary = {
        "timestamp": now_str(),
        "device": str(device),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "feature_config": feat_cfg,
        "class_names": class_names,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (exp_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("All done.")
    log("Writing logs and metrics files...")
