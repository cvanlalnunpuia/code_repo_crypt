from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "enron_seed_structure.json"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output_directory = ROOT / config["output_directory"]
    aggregate = json.loads(
        (output_directory / "aggregate.json").read_text(encoding="utf-8")
    )
    baseline_directory = ROOT / config["baseline_results"]
    records = aggregate["seed_records"]

    checks = {
        "seed_count": len(records) == aggregate["seed_count"] == 10,
        "unique_seeds": len({record["seed"] for record in records}) == len(records),
        "unique_keyword_hashes": len(
            {record["selected_keyword_ids_sha256"] for record in records}
        )
        == len(records),
        "unique_client_hashes": len(
            {record["client_documents_sha256"] for record in records}
        )
        == len(records),
        "matrix_shapes": all(
            record["structure"]["schema"]["document_count"] == 20000
            and record["structure"]["schema"]["keyword_count"] == 500
            for record in records
        ),
        "keyword_ids_unique_within_seed": all(
            len(record["selected_keyword_ids"])
            == len(set(record["selected_keyword_ids"]))
            == 500
            for record in records
        ),
        "positive_incidence": all(
            record["structure"]["schema"]["incidence_count"] > 0
            for record in records
        ),
        "primary_features_finite": all(
            math.isfinite(value)
            for record in records
            for value in record["structure"]["primary_features"].values()
        ),
        "correlations_finite": all(
            math.isfinite(value)
            for feature in aggregate["correlations"].values()
            for checkpoint in feature.values()
            for key, value in checkpoint.items()
            if key != "n"
        ),
    }

    recovery_matches = True
    for record in records:
        baseline = json.loads(
            (baseline_directory / f"seed-{record['seed']}.json").read_text(
                encoding="utf-8"
            )
        )
        recovery_matches &= (
            record["query_weighted_recovery"]
            == baseline["query_weighted_recovery"]
        )
        recovery_matches &= (
            record["distinct_query_macro_recovery"]
            == baseline["distinct_query_macro_recovery"]
        )
    checks["baseline_recovery_join"] = recovery_matches

    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

