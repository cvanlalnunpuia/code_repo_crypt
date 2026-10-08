from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyse_keyword_size_sensitivity import build, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the keyword-size sensitivity comparison.")
    parser.add_argument("--result", type=Path, default=Path("results/corpus/keyword-size-sensitivity-comparison.json"))
    parser.add_argument("--document", type=Path, default=Path("docs/keyword-size-sensitivity.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    rebuilt = build(root)
    stored = json.loads((root / args.result).read_text(encoding="utf-8"))
    if rebuilt != stored:
        raise SystemExit("Stored keyword-size comparison differs from recomputation")
    if (root / args.document).read_text(encoding="utf-8") != markdown(rebuilt):
        raise SystemExit("Stored keyword-size document differs from recomputation")
    print(f"Validated {len(rebuilt['sensitivity_comparisons'])} keyword-size comparisons")


if __name__ == "__main__":
    main()
