import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
from transformers import AutoFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def localize_path(path_text: str, data_root: Path) -> Path:
    p = Path(path_text)
    text = str(p)
    markers = ["数据集", "鏁版嵁闆"]
    if p.exists():
        return p
    for marker in markers:
        if marker in text:
            rel = text.split(marker, 1)[1].lstrip("\\/")
            return data_root / rel
    return p


def read_wav_mono_16k(path: Path, target_sr: int) -> np.ndarray:
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
    return data


def load_split_rows(split_json: Path):
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    rows = []
    for split_name in ("train", "val", "test"):
        for row in payload[split_name]:
            row = dict(row)
            row["split"] = split_name
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = ap.parse_args()

    split_json = Path(args.split_json)
    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    target_sr = int(feature_extractor.sampling_rate)

    rows = load_split_rows(split_json)
    index_path = out_root / "ast_fbank_shard_index.csv"
    meta_path = out_root / "ast_fbank_feature_config.json"
    dtype = np.float16 if args.dtype == "float16" else np.float32

    index_rows = []
    shard = []
    shard_id = 0
    index_in_shard = 0
    failures = []

    def flush():
        nonlocal shard, shard_id, index_in_shard
        if not shard:
            return
        arr = np.stack(shard).astype(dtype)
        shard_name = f"ast_fbank_{shard_id:05d}.npy"
        np.save(out_root / shard_name, arr)
        shard = []
        shard_id += 1
        index_in_shard = 0

    for n, row in enumerate(rows, start=1):
        wav_path = str(row["wav_path"])
        local_path = localize_path(wav_path, data_root)
        try:
            audio = read_wav_mono_16k(local_path, target_sr)
            feat = feature_extractor(audio, sampling_rate=target_sr, return_tensors="np")["input_values"][0]
            shard.append(feat)
            index_rows.append(
                {
                    "wav_path": wav_path,
                    "local_wav_path": str(local_path),
                    "split": row["split"],
                    "class_name": row["class_name"],
                    "label": int(row["label"]),
                    "duration_sec": float(row["duration_sec"]),
                    "source_long_id": row["source_long_id"],
                    "shard_name": f"ast_fbank_{shard_id:05d}.npy",
                    "index_in_shard": index_in_shard,
                    "feature_shape": "1024,128",
                    "dtype": args.dtype,
                }
            )
            index_in_shard += 1
            if len(shard) >= args.shard_size:
                flush()
        except Exception as exc:
            failures.append({"wav_path": wav_path, "local_wav_path": str(local_path), "error": repr(exc)})
        if n == 1 or n % 500 == 0 or n == len(rows):
            print(f"processed {n}/{len(rows)} failures={len(failures)}", flush=True)

    flush()

    with index_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "wav_path",
            "local_wav_path",
            "split",
            "class_name",
            "label",
            "duration_sec",
            "source_long_id",
            "shard_name",
            "index_in_shard",
            "feature_shape",
            "dtype",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    if failures:
        with (out_root / "ast_fbank_failures.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wav_path", "local_wav_path", "error"])
            writer.writeheader()
            writer.writerows(failures)

    meta = {
        "split_json": str(split_json.resolve()),
        "data_root": str(data_root),
        "model_name": args.model_name,
        "feature_extractor": feature_extractor.to_dict(),
        "num_rows": len(rows),
        "num_success": len(index_rows),
        "num_failures": len(failures),
        "num_shards": shard_id,
        "dtype": args.dtype,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

