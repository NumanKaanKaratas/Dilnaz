import argparse
import random
from pathlib import Path


def format_addition(left: int, right: int) -> str:
    return f"{left} + {right} = {left + right}"


def sample_unique_pairs(rng: random.Random, count: int, max_operand: int, forbidden: set[tuple[int, int]]):
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < count:
        pair = (rng.randint(0, max_operand), rng.randint(0, max_operand))
        if pair not in forbidden:
            pairs.add(pair)
    return pairs


def write_lines(path: Path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for left, right in pairs:
            handle.write(format_addition(left, right))
            handle.write("\n")


def write_prompt_answer_tsv(path: Path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for left, right in pairs:
            handle.write(f"{left} + {right} =\t{left + right}")
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--eval-out", type=Path, required=True)
    parser.add_argument("--prompt-out", type=Path, required=True)
    parser.add_argument("--train-sft-out", type=Path, default=None)
    parser.add_argument("--eval-sft-out", type=Path, default=None)
    parser.add_argument("--train-count", type=int, default=1_000_000)
    parser.add_argument("--eval-count", type=int, default=100)
    parser.add_argument("--max-operand", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    if args.train_count <= 0 or args.eval_count <= 0:
        raise ValueError("train-count and eval-count must be positive")
    if args.max_operand <= 0:
        raise ValueError("max-operand must be positive")
    universe_size = (args.max_operand + 1) * (args.max_operand + 1)
    if args.train_count + args.eval_count > universe_size:
        raise ValueError("requested unique pairs exceed operand universe")

    rng = random.Random(args.seed)
    eval_pairs = sample_unique_pairs(rng, args.eval_count, args.max_operand, set())
    train_pairs = sample_unique_pairs(rng, args.train_count, args.max_operand, eval_pairs)

    eval_sorted = sorted(eval_pairs)
    train_sorted = sorted(train_pairs)
    write_lines(args.train_out, train_sorted)
    write_lines(args.eval_out, eval_sorted)
    if args.train_sft_out is not None:
        write_prompt_answer_tsv(args.train_sft_out, train_sorted)
    if args.eval_sft_out is not None:
        write_prompt_answer_tsv(args.eval_sft_out, eval_sorted)
    args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
    with args.prompt_out.open("w", encoding="utf-8", newline="\n") as handle:
        for left, right in eval_sorted:
            handle.write(f"{left} + {right} =")
            handle.write("\n")

    print(
        f"train={args.train_out} train_count={len(train_sorted)} "
        f"eval={args.eval_out} eval_count={len(eval_sorted)} "
        f"prompts={args.prompt_out} train_sft={args.train_sft_out} eval_sft={args.eval_sft_out} "
        f"max_operand={args.max_operand} seed={args.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
