from __future__ import annotations

import argparse
import json
from pathlib import Path

from source_policy import validate_source_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate corpus source approval records.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args()
    json.loads(args.schema.read_text(encoding="utf-8"))
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = []
    identifiers = set()
    for index, source in enumerate(payload.get("sources", []), 1):
        source_errors = validate_source_entry(source)
        if source.get("source_id") in identifiers:
            source_errors.append("duplicate source_id")
        identifiers.add(source.get("source_id"))
        errors.extend(f"source {index}: {error}" for error in source_errors)
    if errors:
        raise SystemExit("\n".join(errors))
    approved = sum(source["status"] == "approved" for source in payload["sources"])
    print(f"Validated {len(payload['sources'])} sources, approved={approved}")


if __name__ == "__main__":
    main()
