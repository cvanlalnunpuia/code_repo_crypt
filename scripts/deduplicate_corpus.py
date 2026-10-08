from __future__ import annotations

import argparse
from pathlib import Path

from corpus_records import deduplicate, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark exact and near-duplicate corpus records.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    records = deduplicate(read_jsonl(args.input), args.threshold)
    write_jsonl(args.output, records)
    counts = {status: sum(item["duplicate"]["status"] == status for item in records) for status in ("unique", "exact", "near")}
    print(f"Processed {len(records)} records, unique={counts['unique']}, exact={counts['exact']}, near={counts['near']}")


if __name__ == "__main__":
    main()
