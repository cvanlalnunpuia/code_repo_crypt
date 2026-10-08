from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus_stemming import evaluate_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reviewed stemmers against annotated tokens.")
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_annotations(args.annotations, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Evaluated {result['overall']['item_count']} annotations with accuracy {result['overall']['accuracy']:.6f}")


if __name__ == "__main__":
    main()
