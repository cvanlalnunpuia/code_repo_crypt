from __future__ import annotations

import argparse
from pathlib import Path

from corpus_tokenisation import build_from_files, write_tokenised_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the shared tokenisation-only corpus matrix.")
    parser.add_argument("input", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result, metadata = build_from_files(args.input, args.selection, args.config)
    write_tokenised_corpus(args.output, result, metadata)
    summary = result["summary"]
    print(f"Built {summary['document_count']} documents and {summary['vocabulary_size']} tokens")


if __name__ == "__main__":
    main()
