from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "vendor" / "ihop-code" / "datasets_pro" / "enron-full.pkl"
RESULT = ROOT / "results" / "structure" / "enron-full.json"


def main() -> None:
    with DATASET.open("rb") as stream:
        dataset, keywords, _ = pickle.load(stream)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    schema = result["schema"]

    document_lengths = [len(document) for document in dataset]
    document_frequency = [0] * len(keywords)
    access_patterns = [[] for _ in keywords]
    for document_id, document in enumerate(dataset):
        if len(document) != len(set(document)):
            raise AssertionError(f"Duplicate keyword in document {document_id}")
        for keyword in document:
            document_frequency[keyword] += 1
            access_patterns[keyword].append(document_id)

    checks = {
        "document_count": len(dataset) == schema["document_count"],
        "keyword_count": len(keywords) == schema["keyword_count"],
        "document_incidence_sum": sum(document_lengths) == schema["incidence_count"],
        "frequency_incidence_sum": sum(document_frequency)
        == schema["incidence_count"],
        "all_keywords_nonempty": min(document_frequency) > 0,
    }

    groups = Counter(document_frequency)
    exact_pair_count = sum(math.comb(size, 2) for size in groups.values())
    checks["exact_frequency_pair_count"] = (
        exact_pair_count
        == result["frequency_ambiguity"]["0"]["ambiguous_pair_count"]
    )
    checks["largest_frequency_group"] = (
        max(groups.values())
        == result["exact_frequency_groups"]["largest_group_size"]
    )
    checks["identical_access_pattern_count"] = (
        len(access_patterns) - len({tuple(pattern) for pattern in access_patterns})
        == result["access_pattern_similarity"][
            "keywords_with_identical_pattern_count"
        ]
    )

    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

