from __future__ import annotations

import argparse
from pathlib import Path

from corpus_records import load_manifest, normalise_record, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise document-level corpus records.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    sources = load_manifest(args.manifest)
    output = []
    for raw in read_jsonl(args.input):
        source_id = raw.get("source_id")
        if source_id not in sources:
            raise ValueError(f"Unknown source {source_id!r}")
        output.append(normalise_record(raw, sources[source_id]))
    write_jsonl(args.output, output)
    print(f"Normalised {len(output)} records")


if __name__ == "__main__":
    main()
