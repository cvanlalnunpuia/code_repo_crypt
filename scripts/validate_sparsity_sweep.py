from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sparsity_sweep.json"
sys.path.insert(0, str(ROOT / "scripts"))

from measure_seed_structure import canonical_hash
from run_sparsity_sweep import prepare_base, retained_access_sets


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(
        (ROOT / config["output_directory"] / "aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    records = result["records"]
    seeds = config["seeds"]
    rates = config["retention_rates"]
    record_map = {
        (record["seed"], record["retention_rate"]): record
        for record in records
    }
    expected_keys = {(seed, rate) for seed in seeds for rate in rates}

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
        "frequency_order_preserved": all(
            record["client_frequency_order_preserved"]
            and record["auxiliary_frequency_order_preserved"]
            for record in records
        ),
        "nonempty_columns": all(
            record["client_minimum_frequency"]
            >= config["minimum_column_frequency"]
            and record["auxiliary_minimum_frequency"]
            >= config["minimum_column_frequency"]
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
                record_map[(seed, rate)]["selected_keyword_ids_sha256"]
                for rate in rates
            }
        )
        == 1
        and len(
            {
                record_map[(seed, rate)]["base_client_matrix_sha256"]
                for rate in rates
            }
        )
        == 1
        and len(
            {
                record_map[(seed, rate)]["base_auxiliary_matrix_sha256"]
                for rate in rates
            }
        )
        == 1
        for seed in seeds
    )

    regenerated_hashes_match = True
    nested_sets = True
    for seed in seeds:
        (
            _,
            _,
            _,
            _,
            selected_keywords,
            client_matrix,
            auxiliary_matrix,
        ) = prepare_base(config, seed)
        regenerated = {}
        for rate in rates:
            client_retained = retained_access_sets(
                client_matrix,
                selected_keywords,
                rate,
                config["minimum_column_frequency"],
                seed,
                "client",
            )
            auxiliary_retained = retained_access_sets(
                auxiliary_matrix,
                selected_keywords,
                rate,
                config["minimum_column_frequency"],
                seed,
                "auxiliary",
            )
            regenerated[rate] = (client_retained, auxiliary_retained)
            record = record_map[(seed, rate)]
            regenerated_hashes_match &= (
                canonical_hash(client_retained)
                == record["client_retained_access_sha256"]
            )
            regenerated_hashes_match &= (
                canonical_hash(auxiliary_retained)
                == record["auxiliary_retained_access_sha256"]
            )
            regenerated_hashes_match &= canonical_hash(
                [len(values) for values in client_retained]
            ) == record["intervened_client_frequency_sha256"]
            regenerated_hashes_match &= canonical_hash(
                [len(values) for values in auxiliary_retained]
            ) == record["intervened_auxiliary_frequency_sha256"]

        ascending_rates = sorted(rates)
        for smaller, larger in zip(ascending_rates, ascending_rates[1:]):
            for matrix_index in (0, 1):
                nested_sets &= all(
                    set(smaller_values).issubset(larger_values)
                    for smaller_values, larger_values in zip(
                        regenerated[smaller][matrix_index],
                        regenerated[larger][matrix_index],
                    )
                )
    checks["regenerated_access_hashes"] = regenerated_hashes_match
    checks["nested_access_sets"] = nested_sets

    checks["achieved_retention"] = all(
        abs(record["achieved_client_incidence_retention"] - record["retention_rate"])
        <= 0.002
        and abs(
            record["achieved_auxiliary_incidence_retention"]
            - record["retention_rate"]
        )
        <= 0.002
        for record in records
    )

    checks["density_decreases_with_retention"] = all(
        values == sorted(values)
        for seed in seeds
        for values in [
            [
                record_map[(seed, rate)]["structure"]["schema"][
                    "matrix_density"
                ]
                for rate in sorted(rates)
            ]
        ]
    )

    checks["frequency_ambiguity_increases_with_thinning"] = all(
        values == sorted(values, reverse=True)
        for seed in seeds
        for values in [
            [
                record_map[(seed, rate)]["structure"]["primary_features"][
                    "exact_frequency_alternative_proportion"
                ]
                for rate in sorted(rates)
            ]
        ]
    )

    control_directory = ROOT / config["control_results"]
    checks["control_matches_source"] = all(
        record_map[(seed, 1.0)]["query_weighted_recovery"]
        == json.loads(
            (control_directory / f"seed-{seed}.json").read_text(
                encoding="utf-8"
            )
        )["query_weighted_recovery"]
        and record_map[(seed, 1.0)]["source"] == "reused_control"
        for seed in seeds
    )

    aggregate = result["aggregate"]
    checkpoints = config["attack"]["iteration_checkpoints"]
    checks["aggregate_means"] = all(
        math.isclose(
            aggregate["by_condition"][f"{rate:.2f}"]["recovery"][
                str(checkpoint)
            ]["mean"],
            float(
                np.mean(
                    [
                        record_map[(seed, rate)]["query_weighted_recovery"][
                            index
                        ]
                        for seed in seeds
                    ]
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for rate in rates
        for index, checkpoint in enumerate(checkpoints)
    )

    checks["paired_means"] = all(
        math.isclose(
            aggregate["paired_differences_from_control"][f"{rate:.2f}"][
                str(checkpoint)
            ]["mean"],
            float(
                np.mean(
                    [
                        record_map[(seed, rate)]["query_weighted_recovery"][
                            index
                        ]
                        - record_map[(seed, 1.0)]["query_weighted_recovery"][
                            index
                        ]
                        for seed in seeds
                    ]
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for rate in rates[1:]
        for index, checkpoint in enumerate(checkpoints)
    )

    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
