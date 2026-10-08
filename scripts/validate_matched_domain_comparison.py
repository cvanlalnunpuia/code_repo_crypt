from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyse_matched_domain_comparison import build, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the matched FLORES-200 and WMT23 comparison.")
    parser.add_argument("--result", type=Path, default=Path("results/corpus/flores200-wmt23-matched-comparison.json"))
    parser.add_argument("--document", type=Path, default=Path("docs/matched-domain-comparison.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    rebuilt = build(root)
    stored = json.loads((root / args.result).read_text(encoding="utf-8"))
    if rebuilt != stored:
        raise SystemExit("Stored matched comparison differs from recomputation")
    expected_document = markdown(rebuilt)
    if (root / args.document).read_text(encoding="utf-8") != expected_document:
        raise SystemExit("Stored matched comparison document differs from recomputation")
    print(f"Validated {len(rebuilt['comparisons'])} matched comparisons")


if __name__ == "__main__":
    main()
