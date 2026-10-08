from __future__ import annotations

import argparse
from pathlib import Path

from corpus_stemming import build_stemming_condition, write_stemming_condition


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reviewed stopword and stemming corpus matrix.")
    parser.add_argument("input", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result, metadata = build_stemming_condition(args.input, args.selection, args.config)
    write_stemming_condition(args.output, result, metadata)
    changed = sum(item["changed_stem_count"] for item in result["documents"])
    print(f"Built {result['summary']['document_count']} documents with {changed} changed token occurrences")


if __name__ == "__main__":
    main()
