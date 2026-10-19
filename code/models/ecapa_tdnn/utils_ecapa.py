import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_path(s: str) -> str:
    return str(Path(s)).replace('/', '\\').lower()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def load_split_json(path: Path):
    data = read_json(path)
    for k in ('train', 'val', 'test'):
        if k not in data:
            raise ValueError(f'split json missing key `{k}`')
    return data


def detect_column(cols, candidates, required_name):
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f'metadata missing required field `{required_name}`; candidates={candidates}, actual={cols}')


def build_metadata_index(
    metadata_csv: Path,
    wav_path_col: str = '',
    ship_name_col: str = 'ship_name',
    ship_type_col: str = 'ship_type',
    recording_id_col: str = 'recording_id',
    split_col: str = 'split',
):
    import csv

    with metadata_csv.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('metadata csv is empty')

    cols = list(rows[0].keys())
    if not wav_path_col:
        wav_path_col = detect_column(cols, ['wav_path', 'file_path', 'feature_path', 'segment_path', 'path'], 'wav_path/file_path/feature_path')

    for col, nm in [(ship_name_col, 'ship_name'), (recording_id_col, 'recording_id'), (split_col, 'split')]:
        if col not in cols:
            raise ValueError(f'metadata missing field `{nm}` (configured as `{col}`)')

    ship_type_mode = 'direct'
    if ship_type_col not in cols:
        if 'source_wav_relpath' in cols:
            ship_type_mode = 'from_source_wav_relpath'
        else:
            raise ValueError(f'metadata missing ship_type field `{ship_type_col}`, and no `source_wav_relpath` to derive ship_type')

    idx = {}
    for r in rows:
        k = normalize_path(r[wav_path_col])
        if ship_type_mode == 'direct':
            st = r[ship_type_col]
        else:
            rel = str(r.get('source_wav_relpath', '')).replace('/', '\\')
            st = rel.split('\\')[0] if rel else 'UNKNOWN'
        idx[k] = {
            'ship_name': r[ship_name_col],
            'ship_type': st,
            'recording_id': r[recording_id_col],
            'split': r[split_col],
        }
    return idx


def attach_metadata_to_split(split_payload, meta_idx):
    out = {'train': [], 'val': [], 'test': []}
    missing = []
    for s in ('train', 'val', 'test'):
        for r in split_payload[s]:
            k = normalize_path(r['wav_path'])
            if k not in meta_idx:
                missing.append(r['wav_path'])
                continue
            rec = dict(r)
            rec.update(meta_idx[k])
            out[s].append(rec)
    if missing:
        raise KeyError('split contains wavs not found in metadata index; first 10=' + json.dumps(missing[:10], ensure_ascii=False))
    return out


def safety_check_recording_overlap(split_with_meta):
    rec = {s: {r['recording_id'] for r in split_with_meta[s]} for s in ('train', 'val', 'test')}
    tv = rec['train'] & rec['val']
    tt = rec['train'] & rec['test']
    vt = rec['val'] & rec['test']
    if tv or tt or vt:
        raise RuntimeError('recording_id overlap detected among splits: ' + json.dumps({'train_val': list(tv)[:5], 'train_test': list(tt)[:5], 'val_test': list(vt)[:5]}, ensure_ascii=False))


def safety_check_no_wav_overlap(split_with_meta):
    w = {s: {normalize_path(r['wav_path']) for r in split_with_meta[s]} for s in ('train', 'val', 'test')}
    if (w['train'] & w['val']) or (w['train'] & w['test']) or (w['val'] & w['test']):
        raise RuntimeError('wav overlap detected among train/val/test')


def safety_check_shipname_from_train_only(split_with_meta):
    train_ship = {r['ship_name'] for r in split_with_meta['train']}
    for s in ('val', 'test'):
        ss = {r['ship_name'] for r in split_with_meta[s]}
        if not ss.issubset(train_ship):
            diff = sorted(list(ss - train_ship))[:10]
            raise RuntimeError(f'{s} contains ship_name not in train: {diff}')


def safety_check_train_shipnames_match_split_class(split_payload, split_with_meta):
    split_train = {r['class_name'] for r in split_payload['train']}
    meta_train = {r['ship_name'] for r in split_with_meta['train']}
    if split_train != meta_train:
        raise RuntimeError('train ship_name set mismatch between split.class_name and metadata.ship_name')


def build_ship_label_from_train(split_with_meta):
    ships = sorted({r['ship_name'] for r in split_with_meta['train']})
    return ships, {s: i for i, s in enumerate(ships)}


def metrics_from_preds(y_true, y_pred, num_classes):
    n = len(y_true)
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    acc = sum(cm[i][i] for i in range(num_classes)) / max(1, n)

    per_class = []
    mp = mr = mf = 0.0
    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        per_class.append({'class_idx': i, 'precision': p, 'recall': r, 'f1': f1})
        mp += p; mr += r; mf += f1
    mp /= max(1, num_classes); mr /= max(1, num_classes); mf /= max(1, num_classes)
    return {
        'accuracy': acc,
        'macro_precision': mp,
        'macro_recall': mr,
        'macro_f1': mf,
        'confusion_matrix': cm,
        'per_class': per_class,
    }


def l2n(x: np.ndarray):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)
