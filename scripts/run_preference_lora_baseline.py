"""Train a DPO/IPO-like preference LoRA baseline from chosen/rejected pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.lora_training import TrainingExample, train_lora_from_examples  # noqa: E402
from pappo.preference_pairs import load_preference_pairs  # noqa: E402


def _preference_examples(pairs_path: Path) -> list[TrainingExample]:
    pairs = load_preference_pairs(pairs_path)
    examples: list[TrainingExample] = []
    for pair in pairs:
        margin = max(float(pair.reward_delta), 1e-6)
        examples.append(
            TrainingExample(
                prompt=pair.prompt,
                response=pair.chosen,
                weight=margin,
            )
        )
        examples.append(
            TrainingExample(
                prompt=pair.prompt,
                response=pair.rejected,
                weight=-margin,
            )
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        try:
            import torch

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass

    examples = _preference_examples(args.pairs)
    result = train_lora_from_examples(
        model_name=args.model,
        examples=examples,
        method="dpo_ipo_like",
        output_dir=args.output_dir / "adapter",
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        local_files_only=args.local_files_only,
    )
    payload = {
        **asdict(result),
        "status": "preference_lora_completed",
        "method": "dpo_ipo_like",
        "pairs": len(examples) // 2,
        "examples": len(examples),
        "pairs_path": str(args.pairs),
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
