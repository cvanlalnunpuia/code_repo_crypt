from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_matched_size_control import build, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a deterministic matched-size control selection.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent
    rebuilt = build(args.config)
    rebuilt["config_sha256"] = file_sha256(args.config)
    stored = json.loads((root / config["output_selection"]).read_text(encoding="utf-8"))
    if rebuilt != stored:
        raise SystemExit("Stored matched-size selection differs from recomputation")
    source = json.loads((root / config["input_selection"]).read_text(encoding="utf-8"))
    source_ids = {
        language: {pair[language] for pair in source["pairs"]}
        for language in source["config"]["languages"]
    }
    for pair in stored["pairs"]:
        for language in stored["config"]["languages"]:
            if pair[language] not in source_ids[language]:
                raise SystemExit(f"Selected {language} record is absent from the source selection")
    print(f"Validated {stored['selected_pair_count']} matched control pairs")


if __name__ == "__main__":
    main()
