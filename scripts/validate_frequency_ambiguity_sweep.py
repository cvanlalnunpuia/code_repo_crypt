from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "frequency_ambiguity_sweep.json"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(
        (ROOT / config["output_directory"] / "aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    records = result["records"]
    seeds = config["seeds"]
    pair_counts = config["equalized_pair_counts"]
    record_map = {
        (record["seed"], record["equalized_pair_count"]): record
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
        "incidence_counts_preserved": all(
            record["base_client_incidence_count"]
            == record["intervened_client_incidence_count"]
            and record["base_auxiliary_incidence_count"]
            == record["intervened_auxiliary_incidence_count"]
            for record in records
        ),
        "active_pair_frequencies_equal": all(
            record["client_active_pairs_frequency_equal"]
            and record["auxiliary_active_pairs_frequency_equal"]
            for record in records
        ),
        "active_pair_overlap_zero": all(
            value == 0.0
            for record in records
            for value in (
                record["client_active_pair_jaccards"]
                + record["auxiliary_active_pair_jaccards"]
            )
        ),
        "pair_metadata": all(
            len(record["pair_pool"]) == max(pair_counts)
            and len(record["active_pairs"]) == record["equalized_pair_count"]
            and record["equalized_keyword_count"]
            == 2 * record["equalized_pair_count"]
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
                record_map[(seed, pair_count)]["selected_keyword_ids_sha256"]
                for pair_count in pair_counts
            }
        )
        == 1
        and len(
            {
                record_map[(seed, pair_count)]["base_client_matrix_sha256"]
                for pair_count in pair_counts
            }
        )
        == 1
        and len(
            {
                record_map[(seed, pair_count)]["base_auxiliary_matrix_sha256"]
                for pair_count in pair_counts
            }
        )
        == 1
        for seed in seeds
    )

    checks["same_pair_pool_within_seed"] = all(
        record_map[(seed, pair_count)]["pair_pool"]
        == record_map[(seed, 0)]["pair_pool"]
        for seed in seeds
        for pair_count in pair_counts[1:]
    )

    checks["nested_active_pairs"] = all(
        record_map[(seed, pair_count)]["active_pairs"]
        == record_map[(seed, 0)]["pair_pool"][:pair_count]
        for seed in seeds
        for pair_count in pair_counts
    )

    checks["fixed_density_within_seed"] = all(
        math.isclose(
            record_map[(seed, pair_count)]["structure"]["schema"][
                "matrix_density"
            ],
            record_map[(seed, 0)]["structure"]["schema"]["matrix_density"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for seed in seeds
        for pair_count in pair_counts[1:]
    )

    checks["frequency_ambiguity_nondecreasing"] = all(
        values == sorted(values)
        for seed in seeds
        for values in [
            [
                record_map[(seed, pair_count)]["structure"]["primary_features"][
                    "exact_frequency_alternative_proportion"
                ]
                for pair_count in pair_counts
            ]
        ]
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
