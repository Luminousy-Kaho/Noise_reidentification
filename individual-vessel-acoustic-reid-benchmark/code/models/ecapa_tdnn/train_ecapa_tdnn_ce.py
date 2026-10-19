import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecapa_model import ECAPATDNN, ECAPAFeatureDataset
from utils_ecapa import (
    attach_metadata_to_split,
    build_metadata_index,
    build_ship_label_from_train,
    load_split_json,
    metrics_from_preds,
    safety_check_no_wav_overlap,
    safety_check_recording_overlap,
    safety_check_shipname_from_train_only,
    safety_check_train_shipnames_match_split_class,
    set_seed,
    write_json,
)


def now_str():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


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


def make_loader(items, batch_size, num_workers, feature_root, shard_root, input_layout, shuffle):
    ds = ECAPAFeatureDataset(items, feature_root=feature_root, shard_root=shard_root, input_layout=input_layout)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def ensure_bmt(x, input_layout):
    # dataset returns [B,M,T] for BMT or [B,T,M] for BTM
    if input_layout == 'BTM':
        x = x.transpose(1, 2)
    return x


def evaluate(model, loader, device, num_classes, input_layout):
    model.eval()
    y_true, y_pred = [], []
    rows = []
    with torch.no_grad():
        for x, y, paths, class_names in loader:
            x = ensure_bmt(x, input_layout).to(device, non_blocking=True)
            logits = model(x)
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
    m = metrics_from_preds(y_true, y_pred, num_classes)
    return m, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-json', required=True)
    ap.add_argument('--metadata-csv', required=True)
    ap.add_argument('--output-dir', required=True)

    ap.add_argument('--wav-path-col', default='')
    ap.add_argument('--ship-name-col', default='ship_name')
    ap.add_argument('--ship-type-col', default='ship_type')
    ap.add_argument('--recording-id-col', default='recording_id')
    ap.add_argument('--split-col', default='split')

    ap.add_argument('--feature-root', default='')
    ap.add_argument('--shard-feature-root', default='')
    ap.add_argument('--input-layout', default='BMT', choices=['BMT', 'BTM'])

    ap.add_argument('--channels', type=int, default=512)
    ap.add_argument('--embedding-dim', type=int, default=192)
    ap.add_argument('--n-mels', type=int, default=128)

    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--print-every-n-batch', type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)

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

    # safety checks
    safety_check_recording_overlap(split_meta)
    safety_check_no_wav_overlap(split_meta)
    safety_check_shipname_from_train_only(split_meta)
    safety_check_train_shipnames_match_split_class(split_payload, split_meta)

    ship_names, ship_to_idx = build_ship_label_from_train(split_meta)
    split = build_items(split_meta, ship_to_idx)
    split_counts = {k: len(v) for k, v in split.items()}

    feature_root = Path(args.feature_root) if args.feature_root else None
    shard_root = Path(args.shard_feature_root) if args.shard_feature_root else None
    if feature_root is None and shard_root is None:
        raise ValueError('either --feature-root or --shard-feature-root must be provided')

    train_dl = make_loader(split['train'], args.batch_size, args.num_workers, feature_root, shard_root, args.input_layout, True)
    val_dl = make_loader(split['val'], args.batch_size, args.num_workers, feature_root, shard_root, args.input_layout, False)
    test_dl = make_loader(split['test'], args.batch_size, args.num_workers, feature_root, shard_root, args.input_layout, False)

    # input shape check (first batch)
    xb, _, _, _ = next(iter(train_dl))
    if args.input_layout == 'BMT':
        if xb.ndim != 3:
            raise RuntimeError(f'ECAPA input must be 3D [B,Mel,Time], got shape={tuple(xb.shape)}')
        if xb.shape[1] != args.n_mels:
            print(f'[Warn] n_mels mismatch: arg={args.n_mels}, actual={xb.shape[1]} (using actual)')
            args.n_mels = int(xb.shape[1])
    else:
        if xb.ndim != 3:
            raise RuntimeError(f'ECAPA input must be 3D [B,Time,Mel], got shape={tuple(xb.shape)}')
        if xb.shape[2] != args.n_mels:
            print(f'[Warn] n_mels mismatch: arg={args.n_mels}, actual={xb.shape[2]} (using actual)')
            args.n_mels = int(xb.shape[2])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECAPATDNN(n_mels=args.n_mels, channels=args.channels, emb_dim=args.embedding_dim, num_classes=len(ship_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f'ecapa_tdnn_ce_{now_str()}'
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_path = run_dir / 'best_checkpoint.pt'

    train_log_path = run_dir / 'train_log.csv'
    with train_log_path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_accuracy', 'val_macro_f1'])
        w.writeheader()
    batch_timing_rows = []

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        total_batches = len(train_dl)
        iter_start = perf_counter()

        for bi, (x, y, _, _) in enumerate(train_dl, start=1):
            data_wait_s = perf_counter() - iter_start
            t0 = perf_counter()
            x = ensure_bmt(x, args.input_layout).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            h2d_s = perf_counter() - t0

            t1 = perf_counter()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            compute_s = perf_counter() - t1

            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
            step_total_s = data_wait_s + h2d_s + compute_s
            batch_timing_rows.append(
                {
                    'epoch': ep,
                    'batch': bi,
                    'data_wait_s': data_wait_s,
                    'h2d_s': h2d_s,
                    'compute_s': compute_s,
                    'step_total_s': step_total_s,
                }
            )

            if bi == 1 or bi % max(1, args.print_every_n_batch) == 0 or bi == total_batches:
                print(
                    f'[Epoch {ep:03d}/{args.epochs:03d}] batch {bi}/{total_batches} '
                    f'running_train_loss={loss_sum/max(1,n):.6f} '
                    f'(wait={data_wait_s:.3f}s h2d={h2d_s:.3f}s compute={compute_s:.3f}s total={step_total_s:.3f}s)',
                    flush=True,
                )
            iter_start = perf_counter()

        scheduler.step()
        train_loss = loss_sum / max(1, n)
        val_m, _ = evaluate(model, val_dl, device, len(ship_names), args.input_layout)

        row = {'epoch': ep, 'train_loss': train_loss, 'val_accuracy': val_m['accuracy'], 'val_macro_f1': val_m['macro_f1']}
        with train_log_path.open('a', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_accuracy', 'val_macro_f1'])
            w.writerow(row)

        print(f"[Epoch {ep:03d}/{args.epochs:03d}] train_loss={train_loss:.6f} val_acc={val_m['accuracy']:.6f} val_macro_f1={val_m['macro_f1']:.6f}", flush=True)

        if val_m['macro_f1'] > best_f1:
            best_f1 = val_m['macro_f1']
            torch.save({'model': model.state_dict(), 'ship_names': ship_names}, best_path)

    # final eval by best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    val_m, val_rows = evaluate(model, val_dl, device, len(ship_names), args.input_layout)
    test_m, test_rows = evaluate(model, test_dl, device, len(ship_names), args.input_layout)

    def dump_rows(rows, path):
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['wav_path', 'true_label', 'pred_label', 'true_class_name', 'pred_confidence'])
            w.writeheader(); w.writerows(rows)

    dump_rows(val_rows, run_dir / 'val_predictions.csv')
    dump_rows(test_rows, run_dir / 'test_predictions.csv')

    per_class = []
    for pc in test_m['per_class']:
        per_class.append({'ship_name': ship_names[pc['class_idx']], 'precision': pc['precision'], 'recall': pc['recall'], 'f1': pc['f1']})
    with (run_dir / 'test_per_class_recall.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ship_name', 'precision', 'recall', 'f1'])
        w.writeheader(); w.writerows(per_class)
    with (run_dir / 'batch_timing.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'batch', 'data_wait_s', 'h2d_s', 'compute_s', 'step_total_s'])
        w.writeheader(); w.writerows(batch_timing_rows)

    summary = {
        'task': 'ecapa_tdnn_ce_ship_identity',
        'model_name': 'ECAPA-TDNN-CE',
        'model_type': 'ecapa_tdnn',
        'init_type': 'NoImageNet',
        'pretrain_source': 'none',
        'sampling': 'no_resample',
        'loss_type': 'cross_entropy',
        'seed': args.seed,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'channels': args.channels,
        'embedding_dim': args.embedding_dim,
        'n_mels': args.n_mels,
        'input_layout': args.input_layout,
        'split_counts': split_counts,
        'ship_name_class_count': len(ship_names),
        'ship_names': ship_names,
        'best_selection_metric': 'val_macro_f1',
        'val_metrics': val_m,
        'test_metrics': test_m,
        'safety_checks': {
            'recording_overlap_checked': True,
            'wav_overlap_checked': True,
            'test_not_used_for_selection': True,
            'val_test_not_resampled': True,
            'ship_name_from_train_only_checked': True,
            'input_dim_checked': True,
        },
    }
    write_json(run_dir / 'metrics_summary.json', summary)
    write_json(run_dir / 'ship_label_map.json', {k: v for k, v in sorted(ship_to_idx.items(), key=lambda kv: kv[1])})

    print(run_dir)


if __name__ == '__main__':
    main()
