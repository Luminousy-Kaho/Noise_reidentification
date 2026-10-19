#!/usr/bin/env python3
"""Scan the release tree for common personal-path and credential patterns."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
PATTERNS = {
    "email_address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "windows_user_directory": re.compile(r"[A-Z]:\\Users\\[^\\\s]+", re.IGNORECASE),
    "windows_absolute_path": re.compile(r"[A-Z]:\\(?![nrt])", re.IGNORECASE),
    "unix_home_directory": re.compile(r"/(?:home|Users)/[^/\s]+", re.IGNORECASE),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


def main() -> None:
    findings: list[tuple[str, str, int]] = []
    self_path = Path(__file__).resolve()
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.resolve() == self_path or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append((path.relative_to(ROOT).as_posix(), name, line))

    forbidden_suffixes = {
        ".docx",
        ".pdf",
        ".png",
        ".jpg",
        ".wav",
        ".pt",
        ".pth",
        ".ckpt",
        ".npy",
        ".npz",
        ".pyc",
    }
    forbidden = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]

    if findings or forbidden:
        for path, kind, line in findings:
            print(f"{path}:{line}: {kind}")
        for path in forbidden:
            print(f"{path}: forbidden release artifact")
        raise SystemExit(1)

    print("PRIVACY SCAN PASSED")
    print("No email, user directory, local drive path, or common credential pattern found.")
    print("No manuscript, figure, raw audio, feature tensor, or checkpoint artifact found.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PRIVACY SCAN FAILED: {exc}", file=sys.stderr)
        raise

