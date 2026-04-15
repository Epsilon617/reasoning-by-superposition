import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Coconut-compatible modular addition state-tracking dataset."
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--name", default="modadd_len5_mod10")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--valid-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--modulus", type=int, default=10)
    parser.add_argument(
        "--choice-format",
        choices=["binary", "all"],
        default="binary",
        help="binary exposes target/neg_target; all exposes every residue as the answer set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_sample(seq, modulus, sample_id, choice_format):
    states = []
    total = 0
    for value in seq:
        total = (total + value) % modulus
        states.append(total)

    target = states[-1]
    neg_candidates = [value for value in range(modulus) if value != target]
    neg_target = neg_candidates[(sample_id * 7 + sum(seq)) % len(neg_candidates)]

    # Existing ProsQA dataloader shuffles "edges"; include position so order remains recoverable.
    edges = [[idx, value] for idx, value in enumerate(seq)]
    steps = [
        f"After reading position {idx}, the running sum modulo {modulus} is {state}."
        for idx, state in enumerate(states)
    ]

    sample = {
        "question": " ".join(str(x) for x in seq),
        "answer": str(target),
        "steps": steps,
        "idx_to_symbol": [str(x) for x in range(max(modulus, len(seq)))],
        "edges": edges,
        "root": 0,
        "target": target,
        "neg_target": neg_target,
        "neighbor_k": {str(idx + 1): [state] for idx, state in enumerate(states)},
    }
    if choice_format == "all":
        sample["choices"] = list(range(modulus))
    return sample


def generate_split(size, seq_len, modulus, rng, seen, choice_format):
    samples = []

    while len(samples) < size:
        seq = tuple(rng.randrange(modulus) for _ in range(seq_len))
        if seq in seen:
            continue
        seen.add(seq)
        samples.append(make_sample(seq, modulus, len(samples), choice_format))
    return samples


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    splits = {
        "train": args.train_size,
        "valid": args.valid_size,
        "test": args.test_size,
    }
    total_size = sum(splits.values())
    max_unique = args.modulus ** args.seq_len
    if total_size > max_unique:
        raise ValueError(
            f"Requested {total_size} examples but only {max_unique} unique sequences exist."
        )

    seen = set()
    for split, size in splits.items():
        data = generate_split(size, args.seq_len, args.modulus, rng, seen, args.choice_format)
        path = output_dir / f"{args.name}_{split}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[modadd] wrote {len(data)} examples to {path}")


if __name__ == "__main__":
    main()
