from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    records = []
    for path in sorted(RESULTS.glob("ihop-enron-volume-seed0-*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        expected = result["expected_recovery"]
        observed = result["observed_recovery"]
        records.append(
            {
                "file": path.name,
                "runtime": result.get(
                    "runtime_id", path.stem.removeprefix("ihop-enron-volume-seed0-")
                ),
                "observed": observed,
                "absolute_differences": [
                    abs(actual - target)
                    for actual, target in zip(observed, expected)
                ],
                "maximum_absolute_difference": max(
                    abs(actual - target)
                    for actual, target in zip(observed, expected)
                ),
                "strict_status": result["status"],
            }
        )
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
