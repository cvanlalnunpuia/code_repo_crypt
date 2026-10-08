from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from scipy import sparse

from corpus_tokenisation import build_from_files, file_sha256, write_tokenised_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently rebuild and validate a tokenised corpus matrix.")
    parser.add_argument("input", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    stored_metadata = json.loads((args.output / "metadata.json").read_text(encoding="utf-8"))
    for name, filename in (("documents_sha256", "documents.json"), ("vocabulary_sha256", "vocabulary.json"), ("incidence_sha256", "incidence.npz")):
        if stored_metadata["artifacts"][name] != file_sha256(args.output / filename):
            raise SystemExit(f"Stored digest failed for {filename}")
    result, metadata = build_from_files(args.input, args.selection, args.config)
    with tempfile.TemporaryDirectory() as directory:
        rebuilt = Path(directory)
        write_tokenised_corpus(rebuilt, result, metadata)
        for filename in ("documents.json", "vocabulary.json", "metadata.json"):
            if (args.output / filename).read_bytes() != (rebuilt / filename).read_bytes():
                raise SystemExit(f"Rebuilt {filename} differs")
        stored_matrix = sparse.load_npz(args.output / "incidence.npz")
        rebuilt_matrix = sparse.load_npz(rebuilt / "incidence.npz")
        if stored_matrix.shape != rebuilt_matrix.shape or (stored_matrix != rebuilt_matrix).nnz:
            raise SystemExit("Rebuilt incidence matrix differs")
    print(f"Validated {stored_metadata['summary']['document_count']} tokenised documents")


if __name__ == "__main__":
    main()
