from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "enron_ablation_summary.json"
sys.path.insert(0, str(ROOT / "scripts"))

from build_enron_ablation_summary import build_contrast


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result_path = ROOT / config["output_json"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    contrast_map = {contrast["id"]: contrast for contrast in result["contrasts"]}
    checks = {
        "contrast_count": len(result["contrasts"]) == len(config["contrasts"]) == 6,
        "unique_contrasts": len(contrast_map) == len(result["contrasts"]),
        "configuration_hash": result["configuration_sha256"] == sha256(CONFIG),
        "source_hashes": all(
            contrast_map[definition["id"]]["source_sha256"]
            == sha256(ROOT / definition["source"])
            for definition in config["contrasts"]
        ),
        "paired_seed_count": all(
            len(contrast["seeds"]) == 10
            and len(set(contrast["seeds"])) == 10
            for contrast in result["contrasts"]
        ),
        "finite_statistics": all(
            math.isfinite(value)
            for contrast in result["contrasts"]
            for checkpoint in contrast["checkpoints"].values()
            for statistic in checkpoint.values()
            for key, value in statistic.items()
            if key != "n"
        ),
        "artifact_files": all(
            (ROOT / config[key]).exists()
            and (ROOT / config[key]).stat().st_size > 0
            for key in ("output_report", "output_figure_png", "output_figure_pdf")
        ),
    }

    recomputed = {
        definition["id"]: build_contrast(definition, config)
        for definition in config["contrasts"]
    }
    checks["independent_recomputation"] = all(
        contrast_map[contrast_id] == contrast
        for contrast_id, contrast in recomputed.items()
    )

    checks["seed_difference_means"] = all(
        math.isclose(
            sum(contrast["final_seed_differences"].values()) / 10,
            contrast["checkpoints"]["1000"]["paired_difference"]["mean"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for contrast in result["contrasts"]
    )

    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
