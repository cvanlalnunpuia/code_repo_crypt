from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_wmt23_corpus import build, file_sha256
from corpus_records import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the acquired WMT23 files and rebuilt paired corpus.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent
    provenance_path = root / config["provenance_output"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for item in provenance["files"]:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Byte count changed for {item['path']}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"SHA-256 changed for {item['path']}")
    records, selection = build(args.config)
    stored_records = read_jsonl(root / config["corpus_output"])
    if records != stored_records:
        raise ValueError("Stored corpus differs from a deterministic rebuild")
    corpus_path = root / config["corpus_output"]
    selection["corpus_sha256"] = file_sha256(corpus_path)
    selection["config_sha256"] = file_sha256(args.config)
    stored_selection = json.loads((root / config["selection_output"]).read_text(encoding="utf-8"))
    if selection != stored_selection:
        raise ValueError("Stored selection differs from a deterministic rebuild")
    print(f"Validated {selection['selected_pair_count']} WMT23 paired documents")


if __name__ == "__main__":
    main()
