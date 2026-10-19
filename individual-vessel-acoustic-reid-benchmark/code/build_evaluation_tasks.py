#!/usr/bin/env python3
"""Build the strict-RWP identity, gallery/query, and negative-pair definitions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def vessel_type(row: dict[str, str]) -> str:
    rel = row.get("source_wav_relpath", "").replace("\\", "/")
    return PurePosixPath(rel).parts[0] if rel else "unknown"


def load_recordings(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"split", "class_name", "source_long_id", "source_wav_relpath"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return rows


def build(rows: list[dict[str, str]]) -> dict[str, object]:
    by_identity: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    identity_type: dict[str, str] = {}
    for row in rows:
        identity = row["class_name"]
        by_identity[identity][row["split"]].append(row["source_long_id"])
        identity_type.setdefault(identity, vessel_type(row))

    identities = sorted(by_identity)
    for identity in identities:
        if len(set(by_identity[identity]["val"])) != 1 or len(set(by_identity[identity]["test"])) != 1:
            raise ValueError(f"{identity} must have exactly one val and one test recording")

    hard_pairs = []
    easy_pairs = []
    for i, left in enumerate(identities):
        for right in identities[i + 1 :]:
            pair = [left, right]
            if identity_type[left] == identity_type[right]:
                hard_pairs.append(pair)
            else:
                easy_pairs.append(pair)

    return {
        "protocol": "strict recording-wise evaluation",
        "identities": identities,
        "identity_type": identity_type,
        "train_recordings": {k: sorted(set(v["train"])) for k, v in by_identity.items()},
        "enrollment_gallery_recordings": {k: sorted(set(v["val"])) for k, v in by_identity.items()},
        "query_recordings": {k: sorted(set(v["test"])) for k, v in by_identity.items()},
        "hard_negative_identity_pairs": hard_pairs,
        "easy_negative_identity_pairs": easy_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording-manifest",
        type=Path,
        default=ROOT / "splits/deepship/min5_median_recording_manifest.csv",
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation_tasks.json"))
    args = parser.parse_args()
    payload = build(load_recordings(args.recording_manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

