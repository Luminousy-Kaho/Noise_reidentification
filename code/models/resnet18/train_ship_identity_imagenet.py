import argparse
import json
from pathlib import Path

import sys

# Reuse the packaged training core.
sys.path.append(str(Path(__file__).resolve().parent))
from core import train_one_experiment, load_split, now_str  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-split-json", required=True)
    ap.add_argument("--workdir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--feature-root", default="")
    ap.add_argument("--shard-feature-root", default="")
    ap.add_argument("--print-every-batch", action="store_true")
    ap.add_argument("--print-every-n-batch", type=int, default=50)
    ap.add_argument("--profile-batch-timing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prepared_split_json = Path(args.prepared_split_json)
    split = load_split(prepared_split_json)

    class_names = sorted({x.class_name for s in split.values() for x in s})
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # ensure labels aligned to class_names order
    for s in split:
        for i, it in enumerate(split[s]):
            split[s][i].label = class_to_idx[it.class_name]

    workdir = Path(args.workdir)
    runs = workdir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    exp_dir = runs / f"ship_identity_{prepared_split_json.parent.name}_{now_str()}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    feat_cfg = {
        "sample_rate": 16000,
        "clip_sec": 5.0,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 512,
        "n_mels": 128,
        "fmin": 20.0,
        "fmax": 8000.0,
    }
    train_one_experiment(
        split=split,
        class_names=class_names,
        exp_dir=exp_dir,
        feat_cfg=feat_cfg,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        feature_root=(Path(args.feature_root) if args.feature_root else None),
        shard_feature_root=(Path(args.shard_feature_root) if args.shard_feature_root else None),
        print_every_batch=args.print_every_batch,
        print_every_n_batch=args.print_every_n_batch,
        profile_batch_timing=args.profile_batch_timing,
    )

    (exp_dir / "run_params.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(exp_dir)


if __name__ == "__main__":
    main()

