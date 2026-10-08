from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from build_length_balanced_control import build, file_sha256, linked_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate linked length-balanced corpus selections and splits.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent
    rebuilt = build(args.config)
    stored_matching = json.loads((root / config["matching_output"]).read_text(encoding="utf-8"))
    comparable_matching = dict(rebuilt["matching"])
    comparable_matching["config_sha256"] = file_sha256(args.config)
    if comparable_matching != stored_matching:
        raise SystemExit("Stored length matching differs from recomputation")
    selection_paths = {
        "flores": root / config["flores_output_selection"],
        "wmt23": root / config["wmt23_output_selection"],
    }
    split_paths = {
        "flores": root / config["flores_output_split"],
        "wmt23": root / config["wmt23_output_split"],
    }
    for side in ("flores", "wmt23"):
        selection_payload = rebuilt["selections"][side]
        stored_selection = json.loads(selection_paths[side].read_text(encoding="utf-8"))
        if selection_payload != stored_selection:
            raise SystemExit(f"Stored {side} length-balanced selection differs")
        expected_split = linked_split(
            selection_payload,
            rebuilt["matching"]["matches"],
            rebuilt["split_indices"]["auxiliary"],
            rebuilt["split_indices"]["client"],
            int(config["split_seed"]),
            float(config["auxiliary_fraction"]),
            selection_paths[side].relative_to(root).as_posix(),
            file_sha256(selection_paths[side]),
        )
        stored_split = json.loads(split_paths[side].read_text(encoding="utf-8"))
        if expected_split != stored_split:
            raise SystemExit(f"Stored {side} linked split differs")
    flores_split = json.loads(split_paths["flores"].read_text(encoding="utf-8"))
    wmt_split = json.loads(split_paths["wmt23"].read_text(encoding="utf-8"))
    if flores_split["auxiliary_pair_indices"] != wmt_split["auxiliary_pair_indices"]:
        raise SystemExit("Linked auxiliary indices differ")
    if flores_split["client_pair_indices"] != wmt_split["client_pair_indices"]:
        raise SystemExit("Linked client indices differ")
    print(f"Validated {stored_matching['matched_pair_count']} linked length-balanced pairs")


if __name__ == "__main__":
    main()
