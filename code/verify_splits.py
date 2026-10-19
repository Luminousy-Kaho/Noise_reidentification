#!/usr/bin/env python3
"""Validate the frozen DeepShip and QiandaoEar22 split manifests."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def verify_deepship() -> None:
    base = ROOT / "splits/deepship"
    recordings = read_csv(base / "min5_median_recording_manifest.csv")
    segments = read_csv(base / "min5_median_segment_manifest.csv")
    random_segments = read_csv(base / "random_segment_25class_manifest.csv")
    label_map = json.loads((base / "ship_label_map.json").read_text(encoding="utf-8"))

    check(len(label_map) == 25, "DeepShip label map contains 25 vessel identities")
    check(len(recordings) == 277, "DeepShip strict split contains 277 source recordings")
    check(len(segments) == 14960, "DeepShip strict split contains 14,960 five-second segments")

    recording_counts = Counter(row["split"] for row in recordings)
    check(recording_counts == Counter({"train": 227, "val": 25, "test": 25}), "DeepShip recording counts are train=227, val=25, test=25")

    split_recordings: dict[str, set[str]] = defaultdict(set)
    identity_splits: dict[str, Counter[str]] = defaultdict(Counter)
    for row in recordings:
        split_recordings[row["split"]].add(row["source_long_id"])
        identity_splits[row["class_name"]][row["split"]] += 1

    check(not (split_recordings["train"] & split_recordings["val"]), "DeepShip train/val recording overlap is zero")
    check(not (split_recordings["train"] & split_recordings["test"]), "DeepShip train/test recording overlap is zero")
    check(not (split_recordings["val"] & split_recordings["test"]), "DeepShip val/test recording overlap is zero")
    check(all(counts["val"] == 1 and counts["test"] == 1 for counts in identity_splits.values()), "Every DeepShip identity has one validation and one test recording")

    strict_ids = {row["source_long_id"] for row in recordings}
    check(all(row["source_long_id"] in strict_ids for row in segments), "Every strict segment maps to a published source recording")
    check(all(not Path(row["segment_relpath"]).is_absolute() for row in segments), "DeepShip segment paths are relative")
    check(len({row["class_name"] for row in random_segments}) == 25, "DeepShip random-segment comparison contains the same 25 identities")


def verify_qiandao() -> None:
    base = ROOT / "splits/qiandaoear22"
    expected = {
        "fullDATA_random_5s_manifest.csv",
        "fullDATA_recordingwise_5s_manifest.csv",
        "dataxxxx30_random_5s_manifest.csv",
        "dataxxxx30_recordingwise_5s_manifest.csv",
        "noDATA_random_5s_manifest.csv",
        "noDATA_recordingwise_5s_manifest.csv",
    }
    actual = {path.name for path in base.glob("*_manifest.csv")}
    check(actual == expected, "All six QiandaoEar22 manifests are present")

    target_columns = {
        "target_SpeedBoat",
        "target_QianDao",
        "target_KaiYuan",
        "target_No7",
        "target_GreenCity",
        "target_TheEarl",
    }
    for path in sorted(base.glob("*_manifest.csv")):
        rows = read_csv(path)
        check(bool(rows), f"{path.name} is non-empty")
        check(target_columns.issubset(rows[0]), f"{path.name} contains all six target columns")
        check(all(row.get("rebuilt_uid") and row.get("split") in {"train", "val", "test"} for row in rows), f"{path.name} has valid rebuilt_uid and split values")
        check(all(not Path(row["rel_path"]).is_absolute() for row in rows if row.get("rel_path")), f"{path.name} uses relative audio paths")

        if "recordingwise" in path.name:
            owners: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                owners[row["rebuilt_uid"]].add(row["split"])
            check(all(len(splits) == 1 for splits in owners.values()), f"{path.name} has zero rebuilt_uid leakage")


def main() -> None:
    verify_deepship()
    verify_qiandao()
    print("\nALL SPLIT CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSPLIT CHECK FAILED: {exc}", file=sys.stderr)
        raise
