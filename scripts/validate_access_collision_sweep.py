from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "access_collision_sweep.json"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(
        (ROOT / config["output_directory"] / "aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    records = result["records"]
    seeds = config["seeds"]
    pair_counts = config["forced_pair_counts"]
    record_map = {
        (record["seed"], record["forced_pair_count"]): record
        for record in records
    }
    expected_keys = {
        (seed, pair_count) for seed in seeds for pair_count in pair_counts
    }

    checks = {
        "record_count": len(records) == len(expected_keys),
        "complete_design": set(record_map) == expected_keys,
        "matrix_shapes": all(
            record["structure"]["schema"]["document_count"]
            == config["client_documents_after_split"]
            and record["structure"]["schema"]["keyword_count"]
            == config["keyword_count"]
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
        "forced_pairs_identical": all(
            record["forced_pairs_identical_client"]
            and record["forced_pairs_identical_auxiliary"]
            for record in records
        ),
        "forced_pair_metadata": all(
            len(record["collision_pairs"]) == record["forced_pair_count"]
            and record["forced_keyword_count"]
            == 2 * record["forced_pair_count"]
            and math.isclose(
                record["forced_keyword_proportion"],
                2 * record["forced_pair_count"] / config["keyword_count"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for record in records
        ),
        "exact_observed_collision_counts": all(
            record["structure"]["access_pattern_similarity"][
                "identical_pair_count"
            ]
            == record["forced_pair_count"]
            and record["structure"]["access_pattern_similarity"][
                "keywords_with_identical_pattern_count"
            ]
            == record["forced_keyword_count"]
            for record in records
        ),
        "finite_structure": all(
            math.isfinite(value)
            for record in records
            for value in record["structure"]["primary_features"].values()
        ),
    }

    checks["same_base_within_seed"] = all(
        len(
            {
                record_map[(seed, pair_count)][
                    "selected_keyword_ids_sha256"
                ]
                for pair_count in pair_counts
            }
        )
        == 1
        and len(
            {
                record_map[(seed, pair_count)][
                    "base_client_documents_sha256"
                ]
                for pair_count in pair_counts
            }
        )
        == 1
        and len(
            {
                record_map[(seed, pair_count)][
                    "base_auxiliary_documents_sha256"
                ]
                for pair_count in pair_counts
            }
        )
        == 1
        for seed in seeds
    )

    checks["nested_collision_pairs"] = all(
        record_map[(seed, larger)]["collision_pairs"][:smaller]
        == record_map[(seed, smaller)]["collision_pairs"]
        for seed in seeds
        for smaller, larger in zip(pair_counts[1:], pair_counts[2:])
    )

    control_directory = ROOT / config["control_results"]
    checks["control_matches_source"] = all(
        record_map[(seed, 0)]["query_weighted_recovery"]
        == json.loads(
            (control_directory / f"seed-{seed}.json").read_text(
                encoding="utf-8"
            )
        )["query_weighted_recovery"]
        and record_map[(seed, 0)]["source"] == "reused_control"
        for seed in seeds
    )

    aggregate = result["aggregate"]
    checkpoints = config["attack"]["iteration_checkpoints"]
    checks["aggregate_means"] = all(
        math.isclose(
            aggregate["by_condition"][str(pair_count)]["recovery"][
                str(checkpoint)
            ]["mean"],
            float(
                np.mean(
                    [
                        record_map[(seed, pair_count)][
                            "query_weighted_recovery"
                        ][index]
                        for seed in seeds
                    ]
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for pair_count in pair_counts
        for index, checkpoint in enumerate(checkpoints)
    )

    checks["paired_means"] = all(
        math.isclose(
            aggregate["paired_differences_from_control"][str(pair_count)][
                str(checkpoint)
            ]["mean"],
            float(
                np.mean(
                    [
                        record_map[(seed, pair_count)][
                            "query_weighted_recovery"
                        ][index]
                        - record_map[(seed, 0)]["query_weighted_recovery"][
                            index
                        ]
                        for seed in seeds
                    ]
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for pair_count in pair_counts[1:]
        for index, checkpoint in enumerate(checkpoints)
    )

    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
