from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import struct
import urllib.request
import zlib
from pathlib import Path
from typing import Any


EOCD = struct.Struct("<4s4H2LH")
CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
LOCAL_HEADER = struct.Struct("<4s5H3L2H")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def request_bytes(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    expected = end - start + 1
    if len(payload) != expected:
        raise ValueError(f"Range response contained {len(payload)} bytes, expected {expected}")
    return payload


def archive_size(url: str) -> tuple[int, dict[str, str | None]]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        size = int(response.headers["Content-Length"])
        headers = {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }
    return size, headers


def read_zip_directory(url: str) -> tuple[int, dict[str, str | None], dict[str, dict[str, int]]]:
    size, headers = archive_size(url)
    tail_start = max(0, size - 65557)
    tail = request_bytes(url, tail_start, size - 1)
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise ValueError("ZIP end-of-central-directory record is absent")
    fields = EOCD.unpack(tail[eocd_offset:eocd_offset + EOCD.size])
    total_entries, central_size, central_offset = fields[4], fields[5], fields[6]
    central = request_bytes(url, central_offset, central_offset + central_size - 1)
    entries: dict[str, dict[str, int]] = {}
    offset = 0
    while offset < len(central):
        fields = CENTRAL_HEADER.unpack(central[offset:offset + CENTRAL_HEADER.size])
        if fields[0] != b"PK\x01\x02":
            raise ValueError(f"Invalid central-directory header at byte {offset}")
        method, crc32, compressed_size, uncompressed_size = fields[4], fields[7], fields[8], fields[9]
        name_length, extra_length, comment_length = fields[10], fields[11], fields[12]
        local_offset = fields[16]
        name_start = offset + CENTRAL_HEADER.size
        name = central[name_start:name_start + name_length].decode("utf-8")
        entries[name] = {
            "method": method,
            "crc32": crc32,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_offset": local_offset,
        }
        offset += CENTRAL_HEADER.size + name_length + extra_length + comment_length
    if len(entries) != total_entries:
        raise ValueError(f"ZIP directory contains {len(entries)} entries, expected {total_entries}")
    return size, headers, entries


def read_zip_entry(url: str, name: str, entry: dict[str, int]) -> bytes:
    local_offset = entry["local_offset"]
    header = request_bytes(url, local_offset, local_offset + LOCAL_HEADER.size - 1)
    fields = LOCAL_HEADER.unpack(header)
    if fields[0] != b"PK\x03\x04":
        raise ValueError(f"Invalid local header for {name}")
    name_length, extra_length = fields[9], fields[10]
    data_start = local_offset + LOCAL_HEADER.size + name_length + extra_length
    compressed = request_bytes(url, data_start, data_start + entry["compressed_size"] - 1)
    if entry["method"] == 0:
        payload = compressed
    elif entry["method"] == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"Unsupported ZIP compression method {entry['method']} for {name}")
    if len(payload) != entry["uncompressed_size"]:
        raise ValueError(f"Uncompressed size mismatch for {name}")
    if binascii.crc32(payload) & 0xFFFFFFFF != entry["crc32"]:
        raise ValueError(f"CRC-32 mismatch for {name}")
    return payload


def acquire(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ValueError("Unsupported config_version")
    root = config_path.resolve().parent.parent
    url = config["archive_url"]
    archive_bytes, response_headers, directory = read_zip_directory(url)
    output_directory = root / config["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    files = []
    for split, languages in config["splits"].items():
        for language, entry_name in languages.items():
            if entry_name not in directory:
                raise ValueError(f"Archive entry is absent: {entry_name}")
            payload = read_zip_entry(url, entry_name, directory[entry_name])
            payload.decode("utf-8-sig")
            output_path = output_directory / f"{split}-{language}.txt"
            output_path.write_bytes(payload)
            files.append({
                "split": split,
                "language": language,
                "archive_entry": entry_name,
                "path": output_path.relative_to(root).as_posix(),
                "bytes": len(payload),
                "crc32": f"{directory[entry_name]['crc32']:08x}",
                "sha256": sha256_bytes(payload),
            })
    return {
        "status": "acquired",
        "source_id": config["source_id"],
        "archive_url": url,
        "archive_bytes": archive_bytes,
        "archive_etag": response_headers["etag"],
        "archive_last_modified": response_headers["last_modified"],
        "retrieved_at": config["retrieved_at"],
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire the English-Mizo files from the WMT23 IndicNECorp archive.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = acquire(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent
    output = root / config["provenance_output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Acquired {len(result['files'])} English-Mizo files")


if __name__ == "__main__":
    main()
