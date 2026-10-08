from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "vendor" / "ihop-code" / "datasets_pro" / "enron-full.pkl"
PACKAGES = ("matplotlib", "numpy", "pandas", "scikit-learn", "scipy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    missing = []
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "missing_packages": missing,
        "dataset_exists": DATASET.is_file(),
        "dataset_bytes": DATASET.stat().st_size if DATASET.is_file() else None,
        "dataset_sha256": sha256(DATASET) if DATASET.is_file() else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if missing or not DATASET.is_file() else 0


if __name__ == "__main__":
    raise SystemExit(main())

