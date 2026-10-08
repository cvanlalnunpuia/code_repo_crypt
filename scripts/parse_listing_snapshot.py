from __future__ import annotations

import argparse
import json
from pathlib import Path

from article_parsers import PARSERS
from corpus_records import load_manifest
from source_discovery import parse_dipr_listing
from source_policy import assert_acquisition_allowed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract article URLs from an approved local listing snapshot.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--listing-url", required=True)
    parser.add_argument("--language", required=True, choices=("en", "lus"))
    args = parser.parse_args()
    sources = load_manifest(args.manifest)
    if args.source_id not in sources:
        raise ValueError(f"Unknown source {args.source_id}")
    source = sources[args.source_id]
    assert_acquisition_allowed(source, args.listing_url, args.language, set(PARSERS))
    if args.source_id != "dipr_mizoram":
        raise ValueError(f"No listing parser is registered for {args.source_id}")
    urls = parse_dipr_listing(args.input.read_text(encoding="utf-8"), source["base_url"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_id": args.source_id,
        "language": args.language,
        "listing_url": args.listing_url,
        "article_urls": urls,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Extracted {len(urls)} article URLs")


if __name__ == "__main__":
    main()
