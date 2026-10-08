from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build document-frequency candidates for native-speaker stopword review.")
    parser.add_argument("tokenised_corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", required=True, choices=("en", "lus"))
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    documents = json.loads((args.tokenised_corpus / "documents.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((args.tokenised_corpus / "vocabulary.json").read_text(encoding="utf-8"))
    matrix = sparse.load_npz(args.tokenised_corpus / "incidence.npz")
    rows = [index for index, document in enumerate(documents) if document["language"] == args.language]
    if not rows:
        raise ValueError(f"No documents found for {args.language}")
    frequencies = np.asarray(matrix[rows, :].sum(axis=0)).ravel().astype(int)
    candidates = []
    for index, item in enumerate(vocabulary):
        count = int(frequencies[index])
        if count:
            candidates.append((count / len(rows), count, item["token"]))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    entries = [
        {
            "token": token,
            "decision": "pending",
            "basis": f"document frequency {count} of {len(rows)} in the {args.language} review corpus",
        }
        for ratio, count, token in candidates[:args.limit]
    ]
    payload = {
        "list_version": "0.1.0",
        "language": args.language,
        "scope": "production",
        "status": "pending",
        "reviewed_by": None,
        "reviewed_at": None,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} pending {args.language} candidates")


if __name__ == "__main__":
    main()
