from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


EXPORT_VERSION = "1.0.0"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_ids(split: dict[str, Any], language: str) -> tuple[list[str], list[int], list[int]]:
    assignment = split["assignments"][language]
    auxiliary = assignment["auxiliary_record_ids"]
    client = assignment["client_record_ids"]
    identifiers = []
    for record_id in auxiliary + client:
        if record_id not in identifiers:
            identifiers.append(record_id)
    index = {record_id: position for position, record_id in enumerate(identifiers)}
    return identifiers, [index[item] for item in auxiliary], [index[item] for item in client]


def build_export(
    condition_dir: Path,
    condition: str,
    language: str,
    split: dict[str, Any],
    keyword_limit: int,
    require_split_support: bool = False,
) -> tuple[list[list[int]], list[str], dict[str, Any]]:
    documents = json.loads((condition_dir / "documents.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((condition_dir / "vocabulary.json").read_text(encoding="utf-8"))
    matrix = sparse.load_npz(condition_dir / "incidence.npz").tocsr()
    row_by_id = {document["record_id"]: index for index, document in enumerate(documents)}
    identifiers, auxiliary_indices, client_indices = ordered_ids(split, language)
    missing = sorted(set(identifiers) - set(row_by_id))
    if missing:
        raise ValueError(f"Condition {condition} lacks split records {missing}")
    rows = [row_by_id[record_id] for record_id in identifiers]
    language_matrix = matrix[rows, :]
    frequency = np.asarray(language_matrix.sum(axis=0)).ravel().astype(int)
    if require_split_support:
        auxiliary_frequency = np.asarray(language_matrix[auxiliary_indices, :].sum(axis=0)).ravel().astype(int)
        client_frequency = np.asarray(language_matrix[client_indices, :].sum(axis=0)).ravel().astype(int)
        selected = [
            index
            for index in range(len(frequency))
            if 0 < auxiliary_frequency[index] < len(auxiliary_indices)
            and 0 < client_frequency[index] < len(client_indices)
        ]
    else:
        selected = [index for index, count in enumerate(frequency) if count > 0]
    selected.sort(key=lambda index: (-int(frequency[index]), vocabulary[index]["token"]))
    selected = selected[:keyword_limit]
    restricted = language_matrix[:, selected].tocsr()
    dataset = [
        sorted(int(value) for value in restricted.getrow(row).indices.tolist())
        for row in range(restricted.shape[0])
    ]
    keywords = [vocabulary[index]["token"] for index in selected]
    auxiliary = {
        "stems_to_words": {keyword: [keyword] for keyword in keywords},
        "document_ids": identifiers,
        "language": language,
        "preprocessing_condition": condition,
        "keyword_selection": {
            "limit": keyword_limit,
            "require_split_support": require_split_support,
        },
        "fixed_split": {
            "mode": split["mode"],
            "auxiliary_indices": auxiliary_indices,
            "client_indices": client_indices,
        },
        "source_artifacts": {
            filename: file_sha256(condition_dir / filename)
            for filename in ("documents.json", "vocabulary.json", "incidence.npz", "metadata.json")
        },
        "export_version": EXPORT_VERSION,
    }
    return dataset, keywords, auxiliary


def write_pickle(path: Path, payload: tuple[list[list[int]], list[str], dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=4)


def build_exports(config_path: Path) -> tuple[Path, dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    project_root = config_path.resolve().parent.parent
    split_path = project_root / config["split"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    output_dir = project_root / config["output_directory"]
    manifest = {
        "export_version": EXPORT_VERSION,
        "config": config,
        "config_sha256": file_sha256(config_path),
        "split_sha256": file_sha256(split_path),
        "datasets": {},
    }
    if "keyword_universe_sizes" in config:
        keyword_sizes = [int(value) for value in config["keyword_universe_sizes"]]
        if not keyword_sizes or len(keyword_sizes) != len(set(keyword_sizes)) or any(value <= 0 for value in keyword_sizes):
            raise ValueError("keyword_universe_sizes must contain unique positive integers")
    else:
        keyword_sizes = [int(config["keyword_universe_size"])]
    multiple_sizes = len(keyword_sizes) > 1
    for condition, relative_dir in config["conditions"].items():
        condition_dir = project_root / relative_dir
        for language in config["languages"]:
            for keyword_size in keyword_sizes:
                payload = build_export(
                    condition_dir,
                    condition,
                    language,
                    split,
                    keyword_size,
                    bool(config.get("require_split_support", False)),
                )
                suffix = f"-k{keyword_size}" if multiple_sizes else ""
                filename = f"{language}-{condition}{suffix}.pkl"
                dataset_id = f"{language}:{condition}:k{keyword_size}" if multiple_sizes else f"{language}:{condition}"
                output_path = output_dir / filename
                write_pickle(output_path, payload)
                dataset, keywords, auxiliary = payload
                manifest["datasets"][dataset_id] = {
                    "path": filename,
                    "sha256": file_sha256(output_path),
                    "document_count": len(dataset),
                    "keyword_count": len(keywords),
                    "incidence_count": sum(len(document) for document in dataset),
                    "auxiliary_document_count": len(auxiliary["fixed_split"]["auxiliary_indices"]),
                    "client_document_count": len(auxiliary["fixed_split"]["client_indices"]),
                }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export bilingual corpus matrices in the IHOP processed-dataset format.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    manifest_path, manifest = build_exports(args.config)
    print(f"Exported {len(manifest['datasets'])} datasets to {manifest_path}")


if __name__ == "__main__":
    main()
