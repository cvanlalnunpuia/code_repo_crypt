from __future__ import annotations

import argparse
from pathlib import Path

from prepare_linguistic_review import build


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the assigned linguistic review record.")
    parser.add_argument("--config", type=Path, default=Path("configs/linguistic_review.json"))
    parser.add_argument("--document", type=Path, default=Path("docs/linguistic-review.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    expected = build(root, root / args.config)
    if (root / args.document).read_text(encoding="utf-8") != expected:
        raise SystemExit("Stored linguistic review record differs from recomputation")
    print("Validated linguistic review assignment")


if __name__ == "__main__":
    main()
