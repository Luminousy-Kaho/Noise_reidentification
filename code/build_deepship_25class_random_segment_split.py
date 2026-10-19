from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPLIT = ROOT / "splits" / "deepship" / "min5_median_segment_manifest.csv"
OUT_DIR = ROOT / "generated_random_segment_split"
SEED = 42
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_split_json(path: Path, rows: list[dict]) -> None:
    payload = {"train": [], "val": [], "test": []}
    for row in rows:
        rec = {
            "split": row["split"],
            "wav_path": row.get("segment_relpath", row.get("wav_path", "")),
            "class_name": row["class_name"],
            "label": int(row["label"]),
            "duration_sec": float(row["duration_sec"]),
            "source_long_id": row["source_long_id"],
        }
        payload[row["split"]].append(rec)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def assign_random_segment(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        clean = dict(row)
        clean["duration_sec"] = float(clean["duration_sec"])
        by_class[clean["class_name"]].append(clean)

    out = []
    for class_name in sorted(by_class):
        arr = by_class[class_name]
        rng.shuffle(arr)
        total = sum(float(r["duration_sec"]) for r in arr)
        target_train = total * RATIOS["train"]
        target_val = total * RATIOS["val"]
        cur_train = 0.0
        cur_val = 0.0
        for row in arr:
            rec = dict(row)
            dur = float(rec["duration_sec"])
            if cur_train < target_train:
                rec["split"] = "train"
                cur_train += dur
            elif cur_val < target_val:
                rec["split"] = "val"
                cur_val += dur
            else:
                rec["split"] = "test"
            out.append(rec)
    return out


def summarize(rows: list[dict]) -> dict:
    summary = {
        "protocol": "DeepShip 25-class random segment split",
        "source_split": str(SOURCE_SPLIT.relative_to(ROOT)),
        "seed": SEED,
        "ratios": RATIOS,
        "ship_count": len({r["class_name"] for r in rows}),
        "segment_count": len(rows),
        "split_counts": {},
        "split_duration_sec": {},
        "per_class": {},
    }
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        summary["split_counts"][split] = len(part)
        summary["split_duration_sec"][split] = sum(float(r["duration_sec"]) for r in part)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)
    for class_name in sorted(by_class):
        crows = by_class[class_name]
        ctotal = sum(float(r["duration_sec"]) for r in crows)
        summary["per_class"][class_name] = {
            "total_segments": len(crows),
            "total_duration_sec": ctotal,
            "split_counts": {s: sum(1 for r in crows if r["split"] == s) for s in ("train", "val", "test")},
            "split_duration_sec": {
                s: sum(float(r["duration_sec"]) for r in crows if r["split"] == s)
                for s in ("train", "val", "test")
            },
        }
    return summary


def leakage_rows(rows: list[dict]) -> list[dict]:
    by_split = {
        split: {r["source_long_id"] for r in rows if r["split"] == split}
        for split in ("train", "val", "test")
    }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    out = []
    all_ids = set().union(*by_split.values())
    for a, b in pairs:
        overlap = by_split[a] & by_split[b]
        out.append(
            {
                "pair": f"{a}_{b}",
                "source_long_id_overlap_count": len(overlap),
                "source_long_id_overlap_ratio_vs_all_ids": len(overlap) / max(1, len(all_ids)),
                "source_long_id_overlap_ratio_vs_left": len(overlap) / max(1, len(by_split[a])),
                "source_long_id_overlap_ratio_vs_right": len(overlap) / max(1, len(by_split[b])),
            }
        )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(SOURCE_SPLIT)
    split_rows = assign_random_segment(rows)
    fieldnames = [
        "split",
        "segment_relpath",
        "class_name",
        "label",
        "duration_sec",
        "source_long_id",
        "source_wav_name",
        "source_wav_relpath",
        "source_duration_sec",
    ]
    write_csv(OUT_DIR / "split_detail.csv", split_rows, [f for f in fieldnames if f in split_rows[0]])
    write_split_json(OUT_DIR / "split_detail.json", split_rows)

    summary = summarize(split_rows)
    (OUT_DIR / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    leaks = leakage_rows(split_rows)
    write_csv(
        OUT_DIR / "leakage_check.csv",
        leaks,
        [
            "pair",
            "source_long_id_overlap_count",
            "source_long_id_overlap_ratio_vs_all_ids",
            "source_long_id_overlap_ratio_vs_left",
            "source_long_id_overlap_ratio_vs_right",
        ],
    )

    report = [
        "# DeepShip 25-Class Random Segment Split",
        "",
        f"- Source segment inventory: `{summary['source_split']}`",
        f"- Seed: `{SEED}`",
        "- Unit of split: segment, not recording.",
        "- Label space: 25 ship identities.",
        "- Intended use: protocol comparison against the existing min5 median recording-wise split.",
        "- Important: source_long_id overlap is expected and is reported below.",
        "",
        "## Overall",
        "",
        f"- Ships: {summary['ship_count']}",
        f"- Segments: {summary['segment_count']}",
        f"- Split counts: {summary['split_counts']}",
        f"- Split duration sec: {summary['split_duration_sec']}",
        "",
        "## Leakage Check",
        "",
        "| pair | source_long_id_overlap_count | ratio_vs_all_ids | ratio_vs_left | ratio_vs_right |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in leaks:
        report.append(
            f"| {row['pair']} | {row['source_long_id_overlap_count']} | "
            f"{row['source_long_id_overlap_ratio_vs_all_ids']:.6f} | "
            f"{row['source_long_id_overlap_ratio_vs_left']:.6f} | "
            f"{row['source_long_id_overlap_ratio_vs_right']:.6f} |"
        )
    (OUT_DIR / "split_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
