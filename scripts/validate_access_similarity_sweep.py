from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "access_similarity_sweep.json"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(
        (ROOT / config["output_directory"] / "aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    records = result["records"]
    seeds = config["seeds"]
    conditions = ["control"] + [
        f"target_{target:.2f}" for target in config["target_jaccard_values"]
    ]
    record_map = {
        (record["seed"], record["condition"]): record for record in records
    }
    expected_keys = {
        (seed, condition) for seed in seeds for condition in conditions
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
        "frequency_flags": all(
            record["client_frequency_preserved"]
            and record["auxiliary_frequency_preserved"]
            for record in records
        ),
        "frequency_hashes": all(
            record["base_client_frequency_sha256"]
            == record["intervened_client_frequency_sha256"]
            and record["base_auxiliary_frequency_sha256"]
            == record["intervened_auxiliary_frequency_sha256"]
            for record in records
        ),
        "pair_count": all(
            len(record["pairs"]) == config["pair_count"] for record in records
        ),
        "disjoint_pairs": all(
            len(
                {
                    position
                    for pair in record["pairs"]
                    for position in (
                        pair["donor_position"],
                        pair["recipient_position"],
                    )
                }
            )
            == 2 * config["pair_count"]
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
                record_map[(seed, condition)]["selected_keyword_ids_sha256"]
                for condition in conditions
            }
        )
        == 1
        and len(
            {
                record_map[(seed, condition)]["base_client_matrix_sha256"]
                for condition in conditions
            }
        )
        == 1
        and len(
            {
                record_map[(seed, condition)]["base_auxiliary_matrix_sha256"]
                for condition in conditions
            }
        )
        == 1
        for seed in seeds
    )

    checks["same_pairs_within_seed"] = all(
        record_map[(seed, condition)]["pairs"]
        == record_map[(seed, "control")]["pairs"]
        for seed in seeds
        for condition in conditions[1:]
    )

    checks["fixed_density_and_frequency_ambiguity"] = all(
        math.isclose(
            record_map[(seed, condition)]["structure"]["schema"][
                "matrix_density"
            ],
            record_map[(seed, "control")]["structure"]["schema"][
                "matrix_density"
            ],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            record_map[(seed, condition)]["structure"]["primary_features"][
                "exact_frequency_alternative_proportion"
            ],
            record_map[(seed, "control")]["structure"]["primary_features"][
                "exact_frequency_alternative_proportion"
            ],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for seed in seeds
        for condition in conditions[1:]
    )

    checks["target_jaccard_tolerance"] = all(
        abs(value - record["target_jaccard"]) <= 0.02
        for record in records
        if record["condition"] != "control"
        for value in (
            record["client_pair_jaccards"] + record["auxiliary_pair_jaccards"]
        )
    )

    control_directory = ROOT / config["control_results"]
    checks["control_matches_source"] = all(
        record_map[(seed, "control")]["query_weighted_recovery"]
        == json.loads(
            (control_directory / f"seed-{seed}.json").read_text(
                encoding="utf-8"
            )
        )["query_weighted_recovery"]
        and record_map[(seed, "control")]["source"] == "reused_control"
        for seed in seeds
    )

    aggregate = result["aggregate"]
    checkpoints = config["attack"]["iteration_checkpoints"]
    checks["aggregate_means"] = all(
        math.isclose(
            aggregate["by_condition"][condition]["recovery"][str(checkpoint)][
                "mean"
            ],
            float(
                np.mean(
                    [
                        record_map[(seed, condition)]["query_weighted_recovery"][
                            index
                        ]
                        for seed in seeds
                    ]
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for condition in conditions
        for index, checkpoint in enumerate(checkpoints)
    )

    checks["paired_means"] = all(
        math.isclose(
            aggregate["paired_differences_from_control"][condition][
                str(checkpoint)
            ]["mean"],
            float(
                np.mean(
                    [
                        record_map[(seed, condition)]["query_weighted_recovery"][
                            index
                        ]
                        - record_map[(seed, "control")][
                            "query_weighted_recovery"
                        ][index]
                        for seed in seeds
                    ]
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for condition in conditions[1:]
        for index, checkpoint in enumerate(checkpoints)
    )

    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
