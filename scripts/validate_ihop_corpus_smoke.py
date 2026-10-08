from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_ihop_corpus_smoke import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the IHOP corpus smoke check.")
    parser.add_argument("result", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    stored = json.loads(args.result.read_text(encoding="utf-8"))
    recomputed = run_smoke(args.config)
    if stored != recomputed:
        raise SystemExit("Stored IHOP corpus smoke result differs from recomputation")
    print(f"Validated IHOP smoke checks for {len(recomputed['datasets'])} datasets")


if __name__ == "__main__":
    main()
