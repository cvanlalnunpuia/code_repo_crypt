from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_score(seed: int, pair: dict[str, Any], languages: list[str]) -> str:
    value = "\n".join([str(seed)] + [pair[language] for language in languages])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    root = config_path.resolve().parent.parent
    input_path = root / config["input_selection"]
    target_path = root / config["target_selection"]
    source = json.loads(input_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    languages = source["config"]["languages"]
    if languages != target["config"]["languages"]:
        raise ValueError("Source and target selections use different language orders")
    target_count = int(target["selected_pair_count"])
    source_pairs = source["pairs"]
    if target_count <= 0 or target_count > len(source_pairs):
        raise ValueError("Target pair count is outside the source selection")
    ranked = sorted(
        enumerate(source_pairs),
        key=lambda item: (pair_score(int(config["seed"]), item[1], languages), item[0]),
    )
    chosen = ranked[:target_count]
    pairs = []
    for rank, (source_index, pair) in enumerate(chosen, 1):
        item = dict(pair)
        item["matched_control_rank"] = rank
        item["parent_pair_index"] = source_index
        pairs.append(item)
    return {
        "algorithm": "deterministic_hash_matched_size_control_v1",
        "config": {
            "config_version": config["config_version"],
            "source_id": source["config"]["source_id"],
            "languages": languages,
            "seed": int(config["seed"]),
            "target_source_id": target["config"]["source_id"],
        },
        "input_selection": config["input_selection"],
        "input_selection_sha256": file_sha256(input_path),
        "input_pair_count": len(source_pairs),
        "target_selection": config["target_selection"],
        "target_selection_sha256": file_sha256(target_path),
        "target_pair_count": target_count,
        "selected_pair_count": len(pairs),
        "selected_record_count": len(pairs) * len(languages),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic corpus control with the target selection size.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent
    result = build(args.config)
    result["config_sha256"] = file_sha256(args.config)
    output = root / config["output_selection"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Selected {result['selected_pair_count']} matched control pairs")


if __name__ == "__main__":
    main()
