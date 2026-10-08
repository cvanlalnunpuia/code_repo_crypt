from __future__ import annotations

import argparse
from pathlib import Path

from corpus_stopwords import build_stopword_condition, write_stopword_condition


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reviewed stopword-removal corpus matrix.")
    parser.add_argument("input", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result, metadata = build_stopword_condition(args.input, args.selection, args.config)
    write_stopword_condition(args.output, result, metadata)
    removed = sum(item["removed_stopword_count"] for item in result["documents"])
    print(f"Built {result['summary']['document_count']} documents after removing {removed} stopword occurrences")


if __name__ == "__main__":
    main()
