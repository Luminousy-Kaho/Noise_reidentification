import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from train_eval_deepship_panns_cnn14 import (  # noqa: E402
    PROJECT_ROOT,
    Cnn14Classifier,
    evaluate_closed_set,
    load_split,
    make_loaders,
    run_eval,
    set_seed,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--init-type", choices=["pretrained_audioset", "random_init"], required=True)
    ap.add_argument(
        "--split-json",
        default=str(
            PROJECT_ROOT
            / "split_protocols"
            / "deepship"
            / "ship_identity"
            / "min5_median_split_20260511_163606"
            / "split_detail.json"
        ),
    )
    ap.add_argument(
        "--split-csv",
        default=str(
            PROJECT_ROOT
            / "split_protocols"
            / "deepship"
            / "ship_identity"
            / "min5_median_split_20260511_163606"
            / "split_detail.csv"
        ),
    )
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--resume-checkpoint", default="latest_model.pt")
    ap.add_argument("--target-epoch", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--print-every-n-batch", type=int, default=500)
    args = ap.parse_args()

    set_seed(args.seed)
    run_dir = Path(args.run_dir)
    rows = load_split(Path(args.split_json), Path(args.data_root))
    class_names = sorted(rows["class_name"].unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    train_dl, val_dl, test_dl = make_loaders(rows, class_to_idx, args.batch_size, args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cnn14Classifier(len(class_names)).to(device)
    resume_path = run_dir / args.resume_checkpoint
    resume_ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(resume_ckpt["model"])
    start_epoch = int(resume_ckpt.get("epoch", 10))

    best_path = run_dir / "best_model.pt"
    best_ckpt = torch.load(best_path, map_location="cpu")
    best_val_f1 = float(best_ckpt.get("best_val_macro_f1", -1.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.target_epoch - start_epoch)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))
    restored_optimizer_state = "optimizer" in resume_ckpt and "scheduler" in resume_ckpt
    if restored_optimizer_state:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        if "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])

    log_path = run_dir / "train_log.csv"
    train_rows = pd.read_csv(log_path).to_dict(orient="records") if log_path.exists() else []

    note = [
        "# Resume Training Note",
        "",
        f"- Resumed from: `{resume_path}`",
        f"- Start epoch in checkpoint: {start_epoch}",
        f"- Target epoch: {args.target_epoch}",
        "- Model weights were restored.",
        f"- Optimizer/scheduler state restored: {restored_optimizer_state}",
        "- If optimizer/scheduler state is false, this first resume starts from epoch-10 weights only; later checkpoints from this resumed run contain full optimizer/scheduler/scaler state.",
        "- PANNs-CNN14 resumed training uses FP32 unless `--amp` is explicitly passed.",
    ]
    (run_dir / "resume_training_note.md").write_text("\n".join(note), encoding="utf-8")

    for epoch in range(start_epoch + 1, args.target_epoch + 1):
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
                print(
                    f"[{args.init_type} resume] epoch {epoch:03d}/{args.target_epoch:03d} "
                    f"batch {bi}/{len(train_dl)} loss={loss_sum / max(1, n):.6f}",
                    flush=True,
                )
        scheduler.step()
        val_m, _ = evaluate_closed_set(model, val_dl, device, class_names)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, n),
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "resume_from_epoch": start_epoch,
        }
        train_rows.append(row)
        pd.DataFrame(train_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
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
            run_dir / "latest_model.pt",
        )
        print(f"[{args.init_type} resume] epoch {epoch:03d} val_macro_f1={val_m['macro_f1']:.6f}", flush=True)
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_names": class_names,
                    "best_val_macro_f1": best_val_f1,
                    "init_type": args.init_type,
                    "epoch": epoch,
                    "resume_from_epoch": start_epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                best_path,
            )
            print(f"saved best: {best_path}", flush=True)

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
        run_dir,
        {
            "model_name": "PANNs-CNN14-AudioSet" if args.init_type == "pretrained_audioset" else "PANNs-CNN14-random-init",
            "init_type": args.init_type,
            "pretrained_checkpoint": "resumed_from_existing_run",
            "best_val_macro_f1": best_val_f1,
            "resumed_from_epoch": start_epoch,
            "target_epoch": args.target_epoch,
        },
    )
    print(f"resume complete and evaluated: {run_dir}")


if __name__ == "__main__":
    main()
