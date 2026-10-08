from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path

from export_ihop_corpus import build_exports


def validate_payload(path: Path) -> None:
    with path.open("rb") as stream:
        dataset, keywords, auxiliary = pickle.load(stream)
    if len(keywords) != len(set(keywords)):
        raise SystemExit(f"Repeated keywords in {path}")
    valid = set(range(len(keywords)))
    for document in dataset:
        if document != sorted(set(document)) or not set(document).issubset(valid):
            raise SystemExit(f"Invalid document keyword identifiers in {path}")
    split = auxiliary["fixed_split"]
    all_indices = set(range(len(dataset)))
    if not set(split["auxiliary_indices"]).issubset(all_indices) or not set(split["client_indices"]).issubset(all_indices):
        raise SystemExit(f"Invalid fixed split indices in {path}")
    if split["mode"] in {"disjoint_by_matched_pair", "linked_disjoint_by_length_match"} and set(split["auxiliary_indices"]) & set(split["client_indices"]):
        raise SystemExit(f"Overlapping disjoint split in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate IHOP corpus exports.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    stored = json.loads(args.manifest.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        temporary_config = json.loads(args.config.read_text(encoding="utf-8"))
        project_root = args.config.resolve().parent.parent
        temporary_config["output_directory"] = str(Path(directory).resolve())
        temporary_config_path = project_root / "configs" / "ihop-validation-temporary.json"
        temporary_config_path.write_text(json.dumps(temporary_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            rebuilt_manifest_path, rebuilt = build_exports(temporary_config_path)
        finally:
            temporary_config_path.unlink(missing_ok=True)
        comparable = dict(rebuilt)
        comparable["config"] = stored["config"]
        comparable["config_sha256"] = stored["config_sha256"]
        for key, item in comparable["datasets"].items():
            stored_item = stored["datasets"][key]
            item["path"] = stored_item["path"]
        if comparable != stored:
            raise SystemExit("Rebuilt IHOP manifest differs")
        for item in stored["datasets"].values():
            original_path = args.manifest.parent / item["path"]
            rebuilt_path = rebuilt_manifest_path.parent / item["path"]
            validate_payload(original_path)
            if original_path.read_bytes() != rebuilt_path.read_bytes():
                raise SystemExit(f"Rebuilt export differs for {item['path']}")
    print(f"Validated {len(stored['datasets'])} IHOP datasets")


if __name__ == "__main__":
    main()
