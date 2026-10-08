from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from scipy import sparse

from corpus_stemming import build_stemming_condition, write_stemming_condition
from corpus_tokenisation import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently rebuild and validate a stemmed corpus.")
    parser.add_argument("input", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.output / "metadata.json").read_text(encoding="utf-8"))
    for name, filename in (("documents_sha256", "documents.json"), ("vocabulary_sha256", "vocabulary.json"), ("incidence_sha256", "incidence.npz")):
        if metadata["artifacts"][name] != file_sha256(args.output / filename):
            raise SystemExit(f"Stored digest failed for {filename}")
    result, rebuilt_metadata = build_stemming_condition(args.input, args.selection, args.config)
    with tempfile.TemporaryDirectory() as directory:
        rebuilt = Path(directory)
        write_stemming_condition(rebuilt, result, rebuilt_metadata)
        for filename in ("documents.json", "vocabulary.json", "metadata.json"):
            if (args.output / filename).read_bytes() != (rebuilt / filename).read_bytes():
                raise SystemExit(f"Rebuilt {filename} differs")
        stored_matrix = sparse.load_npz(args.output / "incidence.npz")
        rebuilt_matrix = sparse.load_npz(rebuilt / "incidence.npz")
        if stored_matrix.shape != rebuilt_matrix.shape or (stored_matrix != rebuilt_matrix).nnz:
            raise SystemExit("Rebuilt incidence matrix differs")
    print(f"Validated {metadata['summary']['document_count']} stemmed documents")


if __name__ == "__main__":
    main()
