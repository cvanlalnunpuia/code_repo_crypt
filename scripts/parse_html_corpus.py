from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from article_parsers import PARSERS, parse_article
from corpus_records import load_manifest, write_jsonl
from source_policy import assert_acquisition_allowed


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse an approved local HTML snapshot into raw corpus JSONL.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--language", required=True, choices=("en", "lus"))
    parser.add_argument("--retrieved-at")
    args = parser.parse_args()
    sources = load_manifest(args.manifest)
    if args.source_id not in sources:
        raise ValueError(f"Unknown source {args.source_id}")
    source = sources[args.source_id]
    assert_acquisition_allowed(source, args.url, args.language, set(PARSERS))
    raw_bytes = args.input.read_bytes()
    parsed = parse_article(source["parser_id"], raw_bytes.decode("utf-8"))
    parsed.update({
        "source_id": args.source_id,
        "language": args.language,
        "canonical_url": args.url,
        "fetched_url": args.url,
        "retrieved_at": args.retrieved_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "raw_reference": str(args.input.as_posix()),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "http_status": 200,
        "content_type": "text/html",
        "parser_version": source["parser_id"],
    })
    write_jsonl(args.output, [parsed])
    print("Parsed 1 approved HTML snapshot")


if __name__ == "__main__":
    main()
