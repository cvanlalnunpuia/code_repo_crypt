from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_fixed_split_experiments import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute fixed-split IHOP experiments.")
    parser.add_argument("result", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    stored = json.loads(args.result.read_text(encoding="utf-8"))
    recomputed = run_experiments(args.config, workers=args.workers, progress=args.progress)
    if stored != recomputed:
        raise SystemExit("Stored fixed-split result differs from recomputation")
    print(f"Validated {len(recomputed['runs'])} fixed-split IHOP runs")


if __name__ == "__main__":
    main()
