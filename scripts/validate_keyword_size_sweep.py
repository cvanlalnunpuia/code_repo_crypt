from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "keyword_size_sweep.json"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output_directory = ROOT / config["output_directory"]
    result = json.loads(
        (output_directory / "aggregate.json").read_text(encoding="utf-8")
    )
    records = result["records"]
    sizes = config["keyword_counts"]
    seeds = config["seeds"]
    record_map = {
        (record["seed"], record["keyword_count"]): record for record in records
    }

    checks = {
        "record_count": len(records) == len(sizes) * len(seeds),
        "complete_design": set(record_map)
        == {(seed, size) for seed in seeds for size in sizes},
        "matrix_shapes": all(
            record["structure"]["schema"]["document_count"]
            == config["client_documents_after_split"]
            and record["structure"]["schema"]["keyword_count"]
            == record["keyword_count"]
            for record in records
        ),
        "unique_keywords_within_record": all(
            len(record["selected_keyword_ids"])
            == len(set(record["selected_keyword_ids"]))
            == record["keyword_count"]
            for record in records
        ),
        "finite_recovery": all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for record in records
            for value in record["query_weighted_recovery"]
        ),
        "weighted_equals_macro": all(
            record["query_weighted_recovery"]
            == record["distinct_query_macro_recovery"]
            for record in records
        ),
        "finite_structure": all(
            math.isfinite(value)
            for record in records
            for value in record["structure"]["primary_features"].values()
        ),
    }

    checks["identical_document_splits_within_seed"] = all(
        len(
            {
                record_map[(seed, size)]["client_documents_sha256"]
                for size in sizes
            }
        )
        == 1
        and len(
            {
                record_map[(seed, size)]["auxiliary_documents_sha256"]
                for size in sizes
            }
        )
        == 1
        for seed in seeds
    )

    checks["nested_keyword_prefixes"] = all(
        record_map[(seed, larger)]["selected_keyword_ids"][:smaller]
        == record_map[(seed, smaller)]["selected_keyword_ids"]
        for seed in seeds
        for smaller, larger in zip(sizes, sizes[1:])
    )

    baseline_directory = ROOT / config["baseline_results"]
    checks["reused_500_matches_baseline"] = all(
        record_map[(seed, 500)]["query_weighted_recovery"]
        == json.loads(
            (baseline_directory / f"seed-{seed}.json").read_text(
                encoding="utf-8"
            )
        )["query_weighted_recovery"]
        and record_map[(seed, 500)]["source"] == "reused_baseline"
        for seed in seeds
    )

    aggregate = result["aggregate"]
    checkpoints = config["attack"]["iteration_checkpoints"]
    checks["aggregate_means"] = all(
        math.isclose(
            aggregate["by_keyword_count"][str(size)]["recovery"][
                str(checkpoint)
            ]["mean"],
            float(
                np.mean(
                    [
                        record_map[(seed, size)]["query_weighted_recovery"][
                            index
                        ]
                        for seed in seeds
                    ]
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for size in sizes
        for index, checkpoint in enumerate(checkpoints)
    )

    checks["paired_means"] = all(
        math.isclose(
            aggregate["paired_recovery_differences"][
                f"{larger}_minus_{smaller}"
            ][str(checkpoint)]["mean"],
            float(
                np.mean(
                    [
                        record_map[(seed, larger)]["query_weighted_recovery"][
                            index
                        ]
                        - record_map[(seed, smaller)][
                            "query_weighted_recovery"
                        ][index]
                        for seed in seeds
                    ]
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for smaller, larger in zip(sizes, sizes[1:])
        for index, checkpoint in enumerate(checkpoints)
    )

    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
