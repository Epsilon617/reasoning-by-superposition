import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import MyCollator
from pilot_capacity import build_model, extract_answer, load_raw_examples
from stokenizer import STokenizer
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Suffix-isomorphic causal patching for modular addition thoughts."
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pass-idx", type=int, default=2)
    parser.add_argument("--limit-pairs", type=int, default=50)
    parser.add_argument("--pairs-only", action="store_true")
    parser.add_argument(
        "--patch-mode",
        choices=["single", "last", "all"],
        default="single",
        help="single patches --pass-idx, last patches the final latent, all overwrites the full trajectory.",
    )
    parser.add_argument(
        "--basis-path",
        default=None,
        help="Optional svd_basis.pt path. If provided, patch in the leading subspace instead of full 768d.",
    )
    parser.add_argument(
        "--subspace-rank",
        type=int,
        default=None,
        help="Number of leading SVD directions to patch when --basis-path is set.",
    )
    parser.add_argument(
        "--kv-cache-policy",
        choices=["full", "latent_only", "prompt_last_1", "prompt_last_2", "prompt_last_4", "prompt_last_8"],
        default="full",
        help="full is the original Coconut behavior; latent_only prevents later passes from reusing prompt KV.",
    )
    return parser.parse_args()


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def seq_from_sample(sample):
    return [digit for _, digit in sorted(sample["edges"], key=lambda pair: pair[0])]


def running_states(seq, modulus):
    total = 0
    states = []
    for digit in seq:
        total = (total + digit) % modulus
        states.append(total)
    return states


def make_prompt_from_seq(seq, choices, latent_steps):
    edges = [[idx, value] for idx, value in enumerate(seq)]
    prefix = (
        "<eos> "
        + "|".join([f" {src} {dst} " for src, dst in edges]).strip()
        + " [Q] "
        + " ".join(str(choice) for choice in choices)
        + " [R] 0"
    )
    return prefix + " <|latent|>" * latent_steps + " [A] "


def make_example(tokenizer, sample_id, prompt):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "position_ids": list(range(len(input_ids))),
        "idx": sample_id,
    }


def run_generate(model, tokenizer, collator, example, device, capture=False):
    model.capture_thoughts = capture
    model.reset_thought_trace()
    batch = collator([example])
    batch = {key: value.to(device) for key, value in batch.items() if key != "idx"}
    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=1,
            synced_gpus=False,
        )
    prediction = extract_answer(tokenizer, outputs)
    captured = [thought.detach().cpu().float() for thought in model.captured_thoughts]
    model.capture_thoughts = False
    model.reset_thought_trace()
    return prediction, captured, tokenizer.decode(outputs[0])


def make_multi_replacement_perturber(patch_map):
    cache = {}

    def perturb(thought, pass_idx, max_n_latents, batch_idx, token_idx):
        del max_n_latents, batch_idx, token_idx
        if pass_idx not in patch_map:
            return thought
        key = (pass_idx, str(thought.device), str(thought.dtype))
        if key not in cache:
            cache[key] = patch_map[pass_idx].to(device=thought.device, dtype=thought.dtype)
        return cache[key]

    return perturb


def make_subspace_replacement_perturber(patch_map, basis):
    cache = {}

    def perturb(thought, pass_idx, max_n_latents, batch_idx, token_idx):
        del max_n_latents, batch_idx, token_idx
        if pass_idx not in patch_map:
            return thought
        key = (pass_idx, str(thought.device), str(thought.dtype))
        if key not in cache:
            donor = patch_map[pass_idx].to(device=thought.device, dtype=thought.dtype)
            dirs = basis.to(device=thought.device, dtype=thought.dtype)
            cache[key] = (donor, dirs)
        donor, dirs = cache[key]
        return thought + dirs @ (dirs.T @ (donor - thought))

    return perturb


def candidate_pairs(raw_examples, pass_idx, limit_pairs):
    if not raw_examples:
        return []
    modulus = len(raw_examples[0]["idx_to_symbol"])
    latent_steps = len(raw_examples[0]["steps"])
    suffix_start = pass_idx + 1
    if suffix_start >= latent_steps:
        raise ValueError(
            f"pass_idx={pass_idx} leaves no suffix for latent_steps={latent_steps}."
        )

    records = []
    for sample_id, sample in enumerate(raw_examples):
        seq = seq_from_sample(sample)
        states = running_states(seq, modulus)
        target = states[-1]
        if int(sample["target"]) != target:
            raise ValueError(f"Sample {sample_id} target does not match modadd state.")
        records.append(
            {
                "sample_id": sample_id,
                "seq": seq,
                "state_at_pass": states[pass_idx],
                "target": target,
                "suffix": tuple(seq[suffix_start:]),
            }
        )

    by_suffix = {}
    for record in records:
        by_suffix.setdefault(record["suffix"], []).append(record)

    pairs = []
    for suffix, group in sorted(by_suffix.items(), key=lambda item: item[0]):
        for base in group:
            for donor in group:
                if base["sample_id"] == donor["sample_id"]:
                    continue
                if base["state_at_pass"] == donor["state_at_pass"]:
                    continue
                if base["target"] == donor["target"]:
                    continue
                pairs.append(
                    {
                        "base_sample_id": base["sample_id"],
                        "donor_sample_id": donor["sample_id"],
                        "suffix": list(suffix),
                        "pass_idx": pass_idx,
                        "base_seq": base["seq"],
                        "donor_seq": donor["seq"],
                        "base_state_at_pass": base["state_at_pass"],
                        "donor_state_at_pass": donor["state_at_pass"],
                        "base_target": base["target"],
                        "donor_target": donor["target"],
                    }
                )
                break
            if len(pairs) >= limit_pairs:
                return pairs
    return pairs


def build_patch_map(donor_thoughts, patch_mode, pass_idx):
    if patch_mode == "single":
        if len(donor_thoughts) <= pass_idx:
            raise ValueError(
                f"Donor has {len(donor_thoughts)} thoughts; cannot patch pass_idx={pass_idx}."
            )
        return {pass_idx: donor_thoughts[pass_idx]}
    if patch_mode == "last":
        return {len(donor_thoughts) - 1: donor_thoughts[-1]}
    return {idx: thought for idx, thought in enumerate(donor_thoughts)}


def load_subspace_basis(basis_path, subspace_rank):
    if basis_path is None:
        return None
    if subspace_rank is None:
        raise ValueError("--subspace-rank is required when --basis-path is set.")
    payload = torch.load(basis_path, map_location="cpu")
    basis = payload["basis"].float()
    if subspace_rank > basis.shape[1]:
        raise ValueError(
            f"Requested subspace rank {subspace_rank}, but basis only has {basis.shape[1]} directions."
        )
    return basis[:, :subspace_rank].contiguous()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_examples = load_raw_examples(args.test_path)
    global_choices = raw_examples[0].get("choices") if raw_examples else None
    pairs = candidate_pairs(raw_examples, args.pass_idx, args.limit_pairs)
    if not pairs:
        raise RuntimeError("No suffix-isomorphic modadd pairs found.")

    pairs_path = output_dir / "candidate_pairs.json"
    write_json(pairs_path, {"pairs": pairs, "n_pairs": len(pairs)})
    print(f"[modadd-patch] Wrote candidate pairs to {pairs_path}", flush=True)

    if args.pairs_only:
        return

    device = torch.device(args.device)
    model, tokenizer, load_result = build_model(args.config_path, args.checkpoint_path, device)
    model.kv_cache_policy = args.kv_cache_policy
    collator = MyCollator(
        tokenizer,
        latent_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        label_pad_token_id=-100,
    )
    latent_steps = len(raw_examples[0]["steps"])
    subspace_basis = load_subspace_basis(args.basis_path, args.subspace_rank)

    results = []
    skipped = []
    for pair in tqdm(pairs, desc="modadd patch"):
        choices = global_choices or [pair["base_target"], pair["donor_target"]]
        base_prompt = make_prompt_from_seq(pair["base_seq"], choices, latent_steps)
        donor_prompt = make_prompt_from_seq(pair["donor_seq"], choices, latent_steps)
        base_example = make_example(tokenizer, pair["base_sample_id"], base_prompt)
        donor_example = make_example(tokenizer, pair["donor_sample_id"], donor_prompt)

        model.thought_perturber = None
        base_prediction, _, base_decoded = run_generate(
            model, tokenizer, collator, base_example, device, capture=False
        )
        donor_prediction, donor_thoughts, donor_decoded = run_generate(
            model, tokenizer, collator, donor_example, device, capture=True
        )
        if base_prediction != str(pair["base_target"]):
            skipped.append({**pair, "reason": "base_baseline_wrong", "prediction": base_prediction})
            continue
        if donor_prediction != str(pair["donor_target"]):
            skipped.append(
                {**pair, "reason": "donor_baseline_wrong", "prediction": donor_prediction}
            )
            continue

        patch_map = build_patch_map(donor_thoughts, args.patch_mode, args.pass_idx)
        if subspace_basis is None:
            model.thought_perturber = make_multi_replacement_perturber(patch_map)
        else:
            model.thought_perturber = make_subspace_replacement_perturber(
                patch_map, subspace_basis
            )
        patched_prediction, _, patched_decoded = run_generate(
            model, tokenizer, collator, base_example, device, capture=False
        )
        model.thought_perturber = None

        results.append(
            {
                **pair,
                "base_prediction": base_prediction,
                "donor_prediction": donor_prediction,
                "patched_prediction": patched_prediction,
                "donor_success": patched_prediction == str(pair["donor_target"]),
                "base_preserved": patched_prediction == str(pair["base_target"]),
                "changed": patched_prediction != base_prediction,
                "base_decoded": base_decoded,
                "donor_decoded": donor_decoded,
                "patched_decoded": patched_decoded,
            }
        )

    if not results:
        raise RuntimeError(f"No pairs survived baseline correctness filters: {skipped[:5]}")

    summary = {
        "n_candidate_pairs": len(pairs),
        "n_evaluated_pairs": len(results),
        "n_skipped": len(skipped),
        "patch_mode": args.patch_mode,
        "pass_idx": args.pass_idx,
        "basis_path": args.basis_path,
        "subspace_rank": args.subspace_rank,
        "kv_cache_policy": args.kv_cache_policy,
        "donor_success_rate": sum(item["donor_success"] for item in results) / len(results),
        "base_preservation_rate": sum(item["base_preserved"] for item in results) / len(results),
        "change_rate": sum(item["changed"] for item in results) / len(results),
        "load_state_dict": str(load_result),
    }
    payload = {
        "summary": summary,
        "pairs": results,
        "skipped": skipped,
        "config": vars(args),
    }
    output_path = output_dir / "modadd_causal_patch_results.json"
    write_json(output_path, payload)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[modadd-patch] Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
