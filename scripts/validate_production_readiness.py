from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_production_readiness import build_report, check_readiness


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the production-readiness report.")
    parser.add_argument("result", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    stored = json.loads(args.result.read_text(encoding="utf-8"))
    recomputed = check_readiness(args.config)
    if stored != recomputed:
        raise SystemExit("Stored production-readiness result differs from recomputation")
    if args.report.read_text(encoding="utf-8") != build_report(recomputed):
        raise SystemExit("Stored production-readiness report differs from recomputation")
    print(f"Validated production status {recomputed['status']}")


if __name__ == "__main__":
    main()
