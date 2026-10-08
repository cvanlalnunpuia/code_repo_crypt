from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_archive(url: str) -> tuple[bytes, dict[str, str | None]]:
    request = urllib.request.Request(url, headers={"User-Agent": "static-sse-study/1.0"})
    with urllib.request.urlopen(request) as response:
        payload = response.read()
        headers = {
            "content_length": response.headers.get("Content-Length"),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "version_id": response.headers.get("x-amz-version-id"),
        }
    return payload, headers


def archive_entry(split: str, language_code: str) -> str:
    return f"./flores200_dataset/{split}/{language_code}.{split}"


def acquire(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    root = config_path.resolve().parent.parent
    payload, headers = read_archive(config["archive_url"])
    archive_sha256 = sha256_bytes(payload)
    if archive_sha256 != config["expected_archive_sha256"]:
        raise ValueError("FLORES-200 archive SHA-256 differs from the pinned value")
    output_directory = root / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    requested: list[tuple[str, str, str]] = [("documentation", "README", "./flores200_dataset/README")]
    for split in config["splits"]:
        requested.append((split, "metadata", f"./flores200_dataset/metadata_{split}.tsv"))
        for language, language_code in config["languages"].items():
            requested.append((split, language, archive_entry(split, language_code)))
    files = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        available = set(archive.getnames())
        for split, kind, entry_name in requested:
            if entry_name not in available:
                raise ValueError(f"Archive entry is absent: {entry_name}")
            stream = archive.extractfile(entry_name)
            if stream is None:
                raise ValueError(f"Archive entry cannot be read: {entry_name}")
            value = stream.read()
            value.decode("utf-8")
            if kind == "README":
                filename = "README.txt"
            elif kind == "metadata":
                filename = f"{split}-metadata.tsv"
            else:
                filename = f"{split}-{kind}.txt"
            output_path = output_directory / filename
            output_path.write_bytes(value)
            files.append({
                "split": split,
                "kind": kind,
                "archive_entry": entry_name,
                "path": output_path.relative_to(root).as_posix(),
                "bytes": len(value),
                "sha256": sha256_bytes(value),
            })
    return {
        "status": "acquired",
        "source_id": config["source_id"],
        "archive_url": config["archive_url"],
        "archive_bytes": len(payload),
        "archive_sha256": archive_sha256,
        "archive_etag": headers["etag"],
        "archive_last_modified": headers["last_modified"],
        "archive_version_id": headers["version_id"],
        "retrieved_at": config["retrieved_at"],
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire the English-Mizo files from the official FLORES-200 archive.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = acquire(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent
    output = root / config["provenance_output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Acquired {len(result['files'])} FLORES-200 files")


if __name__ == "__main__":
    main()
