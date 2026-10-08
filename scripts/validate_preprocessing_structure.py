from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_preprocessing_structure import build_comparison, build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate the preprocessing structural comparison.")
    parser.add_argument("comparison", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    stored = json.loads(args.comparison.read_text(encoding="utf-8"))
    recomputed = build_comparison(args.config)
    if stored != recomputed:
        raise SystemExit("Stored structural comparison differs from independent recomputation")
    if args.report.read_text(encoding="utf-8") != build_report(recomputed):
        raise SystemExit("Stored structural report differs from independent recomputation")
    print(f"Validated {len(recomputed['profiles'])} preprocessing conditions")


if __name__ == "__main__":
    main()
