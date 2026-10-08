from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_flores200_corpus import build, file_sha256
from corpus_records import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the acquired FLORES-200 files and rebuilt paired corpus.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent
    provenance = json.loads((root / config["provenance_output"]).read_text(encoding="utf-8"))
    if provenance["archive_sha256"] != config["expected_archive_sha256"]:
        raise ValueError("Stored archive SHA-256 differs from the pinned value")
    for item in provenance["files"]:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Byte count changed for {item['path']}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"SHA-256 changed for {item['path']}")
    records, selection = build(args.config)
    if records != read_jsonl(root / config["corpus_output"]):
        raise ValueError("Stored corpus differs from a deterministic rebuild")
    corpus_path = root / config["corpus_output"]
    selection["corpus_sha256"] = file_sha256(corpus_path)
    selection["config_sha256"] = file_sha256(args.config)
    stored_selection = json.loads((root / config["selection_output"]).read_text(encoding="utf-8"))
    if selection != stored_selection:
        raise ValueError("Stored selection differs from a deterministic rebuild")
    print(f"Validated {selection['selected_pair_count']} FLORES-200 paired documents")


if __name__ == "__main__":
    main()
