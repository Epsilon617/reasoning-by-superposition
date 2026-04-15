import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from coconut import Coconut
from dataset import MyCollator
from stokenizer import STokenizer
from utils import set_seed


DEFAULT_HF_REPO_ID = "Shibo-UCSD/coconut-theory"
DEFAULT_HF_FILENAME = "checkpoint_300"
DEFAULT_CONFIG_PATH = "configs/symbol-2layer-8head-768dim.json"
DEFAULT_TEST_PATH = "data/prosqa_test_graph_4_coconut.json"
DEFAULT_SEED = 0
THEORY_N = 23
THEORY_BITS = math.log2(4 * THEORY_N)
THEORY_SIGMA = 1.0 / (4 * THEORY_N)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pilot sweep for functional bit capacity of continuous thoughts."
    )
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--theory-n", type=int, default=THEORY_N)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--checkpoint-source", choices=["hf", "local"], default="hf")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-filename", default=DEFAULT_HF_FILENAME)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument(
        "--kv-cache-policy",
        choices=["full", "latent_only", "prompt_last_1", "prompt_last_2", "prompt_last_4", "prompt_last_8"],
        default="full",
        help="full is the original Coconut behavior; latent_only prevents later passes from reusing prompt KV.",
    )
    parser.add_argument("--command", default=None)
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_raw_examples(test_path):
    with open(test_path) as f:
        return json.load(f)


def build_eval_examples(raw_examples, tokenizer, seed):
    rng = random.Random(seed)
    examples = []
    answers = []
    thought_counts = []
    node_counts = []
    for idx, sample in enumerate(raw_examples):
        edges = [list(edge) for edge in sample["edges"]]
        rng.shuffle(edges)
        prefix = (
            "<eos> "
            + "|".join([f" {src} {dst} " for src, dst in edges]).strip()
            + " [Q] "
        )
        if "choices" in sample:
            prefix += " ".join(str(choice) for choice in sample["choices"])
        elif rng.random() < 0.5:
            prefix += f"{sample['target']} {sample['neg_target']}"
        else:
            prefix += f"{sample['neg_target']} {sample['target']}"
        prefix += f" [R] {sample['root']}"
        latent_steps = len(sample["steps"])
        question = prefix + " <|latent|>" * latent_steps + " [A] "
        question_tokenized = tokenizer.encode(question, add_special_tokens=False)
        examples.append(
            {
                "input_ids": question_tokenized,
                "attention_mask": [1] * len(question_tokenized),
                "position_ids": list(range(len(question_tokenized))),
                "idx": idx,
            }
        )
        answers.append(str(sample["target"]))
        thought_counts.append(latent_steps)
        node_counts.append(len(sample["idx_to_symbol"]))
    return examples, answers, thought_counts, node_counts


def load_checkpoint(args, output_dir):
    if args.checkpoint_source == "local":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint-path is required when --checkpoint-source=local")
        checkpoint_path = Path(args.checkpoint_path)
    else:
        cache_dir = args.hf_cache_dir or str(output_dir / "hf_cache")
        print(
            f"[pilot] Downloading released checkpoint from {args.hf_repo_id}/{args.hf_filename} into {cache_dir}",
            flush=True,
        )
        checkpoint_path = Path(
            hf_hub_download(
                repo_id=args.hf_repo_id,
                filename=args.hf_filename,
                cache_dir=cache_dir,
            )
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[pilot] Checkpoint ready at {checkpoint_path}", flush=True)
    return checkpoint_path


def build_model(config_path, checkpoint_path, device, kv_cache_policy="full"):
    tokenizer = STokenizer()
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    print(f"[pilot] Building GPT-2 config from {config_path}", flush=True)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(config_path))
    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    model.kv_cache_policy = kv_cache_policy
    print("[pilot] Loading checkpoint weights", flush=True)
    saved_weights = torch.load(checkpoint_path, map_location="cpu")
    load_result = model.load_state_dict(saved_weights, strict=False)
    model.to(device)
    model.eval()
    print(f"[pilot] Model moved to {device}", flush=True)
    return model, tokenizer, load_result


def quantize_uniform(h, bits):
    if bits >= 16:
        return h
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    h_min = h.min()
    h_max = h.max()
    if torch.isclose(h_max, h_min):
        return h
    levels = 2 ** bits
    scale = (h_max - h_min) / (levels - 1)
    if torch.isclose(scale, torch.zeros_like(scale)):
        return h
    h_q = torch.round((h - h_min) / scale) * scale + h_min
    return h_q.to(h.dtype)


def add_gaussian_noise(h, sigma):
    d = h.numel()
    signal_scale = h.norm() / math.sqrt(d)
    noise = torch.randn_like(h) * sigma * signal_scale
    return h + noise


def fit_lowrank_basis(thoughts):
    _, singular_values, v_t = torch.linalg.svd(thoughts, full_matrices=False)
    return v_t.T, singular_values


def project_lowrank(h, basis):
    return basis @ (basis.T @ h)


def extract_answer(tokenizer, outputs):
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text_output = text_output.replace("<eos>", "").strip()
    answer_output = text_output.split("[A]")[-1].replace(",", "").strip()
    return answer_output.split()[0] if answer_output else ""


def evaluate_model(
    model,
    tokenizer,
    examples,
    answers,
    device,
    collator,
    perturbation=None,
    capture_thoughts=False,
    noise_seed=None,
    desc="eval",
):
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(noise_seed)

    model.thought_perturber = perturbation
    model.capture_thoughts = capture_thoughts
    model.reset_thought_trace()

    total = 0
    correct = 0
    all_thoughts = []
    first_thought_shape = None

    for example in tqdm(examples, desc=desc):
        batch = collator([example])
        batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
        model.reset_thought_trace()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=1,
                synced_gpus=False,
            )
        prediction = extract_answer(tokenizer, outputs)
        answer = answers[example["idx"]]
        correct += int(prediction == answer)
        total += 1

        if capture_thoughts:
            captured = model.captured_thoughts
            if captured:
                if first_thought_shape is None:
                    first_thought_shape = list(captured[0].shape)
                all_thoughts.extend(captured)

    accuracy = correct / total if total else 0.0
    model.thought_perturber = None
    model.capture_thoughts = False
    model.reset_thought_trace()
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "thoughts": all_thoughts,
        "first_thought_shape": first_thought_shape,
    }


def make_perturber(kind, value, basis=None, mode="all"):
    basis_cache = {}

    def should_apply(pass_idx, max_n_latents):
        if mode == "all":
            return True
        if mode == "first":
            return pass_idx == 0
        if mode == "last":
            return pass_idx == max_n_latents - 1
        raise ValueError(f"Unknown perturbation mode: {mode}")

    def perturb(thought, pass_idx, max_n_latents, batch_idx, token_idx):
        del batch_idx, token_idx
        if not should_apply(pass_idx, max_n_latents):
            return thought
        if kind == "quantization":
            return quantize_uniform(thought, int(value))
        if kind == "gaussian_noise":
            return add_gaussian_noise(thought, float(value))
        if kind == "lowrank":
            if basis is None:
                raise ValueError("basis is required for low-rank projection")
            rank = int(value)
            cache_key = (str(thought.device), str(thought.dtype))
            if cache_key not in basis_cache:
                basis_cache[cache_key] = basis.to(device=thought.device, dtype=thought.dtype)
            device_basis = basis_cache[cache_key]
            if rank >= device_basis.shape[1]:
                return thought
            return project_lowrank(thought, device_basis[:, :rank])
        raise ValueError(f"Unknown perturbation kind: {kind}")

    return perturb


def compute_thought_stats(thought_matrix, singular_values):
    norms = thought_matrix.norm(dim=1)
    top20 = singular_values[:20].tolist()
    effective_rank = int((singular_values > 0.01 * singular_values[0]).sum().item())
    squared = singular_values.pow(2)
    energy = squared.cumsum(dim=0) / squared.sum()
    rank_95pct_energy = int(torch.searchsorted(energy, torch.tensor(0.95)).item()) + 1
    rank_99pct_energy = int(torch.searchsorted(energy, torch.tensor(0.99)).item()) + 1
    return {
        "n_thoughts": int(thought_matrix.shape[0]),
        "hidden_size": int(thought_matrix.shape[1]),
        "mean_l2_norm": float(norms.mean().item()),
        "per_coordinate_mean": thought_matrix.mean(dim=0).tolist(),
        "per_coordinate_std": thought_matrix.std(dim=0, unbiased=False).tolist(),
        "effective_rank": effective_rank,
        "effective_rank_1pct_sv": effective_rank,
        "rank_95pct_energy": rank_95pct_energy,
        "rank_99pct_energy": rank_99pct_energy,
        "effective_rank_95pct_energy": rank_95pct_energy,
        "effective_rank_99pct_energy": rank_99pct_energy,
        "top_20_singular_values": top20,
    }


def save_plot_quantization(output_dir, quantization, theory_bits):
    plt = get_plt()
    bits = [item["bits"] for item in quantization]
    accuracies = [item["accuracy"] for item in quantization]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bits, accuracies, marker="o")
    ax.set_xscale("log", base=2)
    ax.axvline(theory_bits, linestyle="--", color="tab:red", label=f"theory={theory_bits:.2f}")
    ax.set_xlabel("Bits")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Quantization Bits")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quantization_curve.png", dpi=300)
    fig.savefig(output_dir / "quantization_curve.pdf")
    plt.close(fig)


def save_plot_noise(output_dir, gaussian_noise, theory_sigma):
    plt = get_plt()
    sigmas = [item["sigma"] for item in gaussian_noise]
    means = [item["accuracy_mean"] for item in gaussian_noise]
    stds = [item["accuracy_std"] for item in gaussian_noise]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sigmas, means, yerr=stds, marker="o", capsize=3)
    ax.set_xscale("log")
    ax.axvline(theory_sigma, linestyle="--", color="tab:red", label=f"theory={theory_sigma:.4f}")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Gaussian Noise")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "noise_curve.png", dpi=300)
    fig.savefig(output_dir / "noise_curve.pdf")
    plt.close(fig)


def save_plot_lowrank(output_dir, lowrank, theory_rank):
    plt = get_plt()
    ranks = [item["rank"] for item in lowrank]
    accuracies = [item["accuracy"] for item in lowrank]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, accuracies, marker="o")
    ax.set_xscale("log", base=2)
    ax.axvline(theory_rank, linestyle="--", color="tab:red", label=f"theory={theory_rank}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Low-Rank Projection")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lowrank_curve.png", dpi=300)
    fig.savefig(output_dir / "lowrank_curve.pdf")
    plt.close(fig)


def describe_drop(points, baseline, key):
    if not points:
        return "No points were recorded."
    threshold = baseline - 0.05
    first_drop = None
    for item in points:
        value = item["accuracy"] if "accuracy" in item else item["accuracy_mean"]
        if value <= threshold:
            first_drop = item[key]
            break
    if first_drop is None:
        return (
            f"The curve stayed within 5 accuracy points of the baseline across the full sweep; "
            f"no clear drop point appeared by the largest perturbation tested."
        )
    return (
        f"The curve first moved more than 5 accuracy points below the baseline at {key}={first_drop}."
    )


def write_pilot_log(
    output_dir,
    args,
    checkpoint_path,
    checkpoint_note,
    timings,
    results,
    deviations,
):
    quant_summary = describe_drop(results.get("quantization", []), results["baseline_accuracy"], "bits")
    noise_summary = describe_drop(results.get("gaussian_noise", []), results["baseline_accuracy"], "sigma")
    lowrank_summary = describe_drop(results.get("lowrank", []), results["baseline_accuracy"], "rank")

    lines = [
        "# Pilot Log",
        "",
        "## Commands",
        "",
        "```bash",
        args.command or " ".join(sys.argv),
        "```",
        "",
        "## Checkpoint",
        "",
        f"- Source: {results['checkpoint_source']}",
        f"- Path: {checkpoint_path}",
        f"- Note: {checkpoint_note}",
        "",
        "## Deviations",
        "",
    ]

    if deviations:
        lines.extend([f"- {item}" for item in deviations])
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Wall-Clock Time",
            "",
            f"- Checkpoint/model setup: {timings.get('setup', 0.0):.2f} minutes",
            f"- Baseline + thought collection: {timings.get('baseline', 0.0):.2f} minutes",
            f"- Perturbation sweep: {timings.get('sweep', 0.0):.2f} minutes",
            f"- Plotting + serialization: {timings.get('finalize', 0.0):.2f} minutes",
            "",
            "## Curve Summary",
            "",
            (
                f"Baseline accuracy was {results['baseline_accuracy']:.4f} on "
                f"{results['n_test_examples']} test examples. "
                f"For quantization, {quant_summary} "
                f"The theoretical marker was b*={results['theoretical_thresholds']['bits']:.2f}."
            ),
            "",
            (
                f"For Gaussian noise, {noise_summary} "
                f"The theoretical marker was sigma*={results['theoretical_thresholds']['sigma']:.4f}. "
                f"Each sigma used {results.get('gaussian_noise', [{}])[0].get('n_seeds', 0) if results.get('gaussian_noise') else 0} seeds."
            ),
            "",
            (
                f"For low-rank projection, {lowrank_summary} "
                f"The theoretical marker was r={results['theoretical_thresholds']['rank']}. "
                f"Control runs are stored in the JSON outputs for first-thought-only and last-thought-only perturbations."
            ),
            "",
        ]
    )

    with open(output_dir / "pilot_log.md", "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_seed(args.seed)
    print(f"[pilot] Output directory: {output_dir}", flush=True)
    print(f"[pilot] Using seed {args.seed}", flush=True)

    raw_examples = load_raw_examples(args.test_path)
    if args.limit is not None:
        raw_examples = raw_examples[: args.limit]
    print(f"[pilot] Loaded {len(raw_examples)} raw test examples from {args.test_path}", flush=True)

    deviations = []
    if len(raw_examples) != 500:
        deviations.append(
            f"The repo test split contains {len(raw_examples)} examples rather than the 500 examples described in the pilot instructions."
        )
    deviations.append(
        "The official notebook provides a released Hugging Face checkpoint, but the notebook notes that it is an example checkpoint and may not exactly match the paper's reported ProsQA accuracy."
    )
    deviations.append(
        "The low-rank basis was fit from all unperturbed test-set thoughts so ranks up to 768 are well-defined; the written instructions suggested 256 thoughts, which would cap the SVD rank at 256."
    )
    if str(output_dir).startswith("/home/claude/") is False:
        deviations.append(
            f"Results were written to {output_dir} instead of /home/claude/pilot/results because this workspace is rooted at {Path.cwd()}."
        )

    timing_start = time.perf_counter()
    checkpoint_path = load_checkpoint(args, output_dir)
    checkpoint_note = (
        "Released checkpoint from the official notebook."
        if args.checkpoint_source == "hf"
        else "Local checkpoint path provided by user."
    )
    device = torch.device(args.device)
    model, tokenizer, load_result = build_model(
        args.config_path, checkpoint_path, device, kv_cache_policy=args.kv_cache_policy
    )
    collator = MyCollator(
        tokenizer,
        latent_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        label_pad_token_id=-100,
    )
    examples, answers, thought_counts, node_counts = build_eval_examples(
        raw_examples, tokenizer, seed=args.seed
    )
    print(
        f"[pilot] Prepared {len(examples)} eval prompts with latent-step mean {sum(thought_counts) / len(thought_counts):.2f}",
        flush=True,
    )
    setup_minutes = (time.perf_counter() - timing_start) / 60.0

    baseline_start = time.perf_counter()
    print("[pilot] Running baseline evaluation and collecting thoughts", flush=True)
    baseline_eval = evaluate_model(
        model,
        tokenizer,
        examples,
        answers,
        device,
        collator,
        perturbation=None,
        capture_thoughts=True,
        desc="baseline",
    )
    baseline_minutes = (time.perf_counter() - baseline_start) / 60.0
    print(
        f"[pilot] Baseline accuracy: {baseline_eval['accuracy']:.4f} over {baseline_eval['total']} examples",
        flush=True,
    )

    first_shape = baseline_eval["first_thought_shape"]
    if first_shape != [768]:
        raise RuntimeError(
            f"Unexpected continuous thought shape: {first_shape}. Expected [768]."
        )

    thought_matrix = torch.stack(baseline_eval["thoughts"]).float()
    basis, singular_values = fit_lowrank_basis(thought_matrix)
    thought_stats = compute_thought_stats(thought_matrix, singular_values)
    print(
        f"[pilot] Collected {thought_matrix.shape[0]} thoughts with hidden size {thought_matrix.shape[1]}",
        flush=True,
    )

    results = {
        "baseline_accuracy": baseline_eval["accuracy"],
        "n_test_examples": baseline_eval["total"],
        "checkpoint_source": "downloaded"
        if args.checkpoint_source == "hf"
        else "trained_from_scratch",
        "training_time_hours": None,
        "checkpoint_path": str(checkpoint_path),
        "load_state_dict": str(load_result),
        "continuous_thought_shape": first_shape,
        "latent_steps_summary": {
            "min": min(thought_counts),
            "max": max(thought_counts),
            "mean": sum(thought_counts) / len(thought_counts),
        },
        "node_count_summary": {
            "min": min(node_counts),
            "max": max(node_counts),
            "mean": sum(node_counts) / len(node_counts),
        },
        "theoretical_thresholds": {
            "n": args.theory_n,
            "bits": math.log2(4 * args.theory_n),
            "sigma": 1.0 / (4 * args.theory_n),
            "rank": args.theory_n,
        },
        "thought_tensor_location": {
            "file": "coconut.py",
            "note": "The hidden state immediately before latent feedback is captured and optionally perturbed inside Coconut.forward().",
        },
        "quantization": [],
        "gaussian_noise": [],
        "lowrank": [],
        "controls": {
            "first_thought_only": {},
            "last_thought_only": {},
        },
    }

    with open(output_dir / "thought_stats.json", "w") as f:
        json.dump(thought_stats, f, indent=2)
    write_json(output_dir / "results.json", results)

    if baseline_eval["accuracy"] < 0.90:
        results["failure_reason"] = (
            f"Baseline accuracy {baseline_eval['accuracy']:.4f} is below the 0.90 stop threshold."
        )
        write_json(output_dir / "results.json", results)
        write_pilot_log(
            output_dir,
            args,
            checkpoint_path,
            checkpoint_note,
            {
                "setup": setup_minutes,
                "baseline": baseline_minutes,
                "sweep": 0.0,
                "finalize": 0.0,
            },
            results,
            deviations,
        )
        raise RuntimeError(results["failure_reason"])

    sweep_minutes = 0.0
    if not args.skip_sweep:
        sweep_start = time.perf_counter()
        print("[pilot] Running quantization sweep", flush=True)

        quant_bits = [1, 2, 3, 4, 6, 8, 12, 16]
        for bits in quant_bits:
            perturber = make_perturber("quantization", bits, mode="all")
            eval_result = evaluate_model(
                model,
                tokenizer,
                examples,
                answers,
                device,
                collator,
                perturbation=perturber,
                capture_thoughts=False,
                desc=f"quant-{bits}bit",
            )
            results["quantization"].append(
                {"bits": bits, "accuracy": eval_result["accuracy"]}
            )
        write_json(output_dir / "results.json", results)

        sigmas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        print("[pilot] Running Gaussian noise sweep", flush=True)
        for sigma_idx, sigma in enumerate(sigmas):
            accuracies = []
            for seed_offset in range(3):
                eval_result = evaluate_model(
                    model,
                    tokenizer,
                    examples,
                    answers,
                    device,
                    collator,
                    perturbation=make_perturber("gaussian_noise", sigma, mode="all"),
                    capture_thoughts=False,
                    noise_seed=args.seed + sigma_idx * 100 + seed_offset,
                    desc=f"noise-{sigma}-seed{seed_offset}",
                )
                accuracies.append(eval_result["accuracy"])
            mean_accuracy = sum(accuracies) / len(accuracies)
            variance = sum((x - mean_accuracy) ** 2 for x in accuracies) / len(accuracies)
            results["gaussian_noise"].append(
                {
                    "sigma": sigma,
                    "accuracy_mean": mean_accuracy,
                    "accuracy_std": math.sqrt(variance),
                    "n_seeds": len(accuracies),
                }
            )
        write_json(output_dir / "results.json", results)

        ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768]
        print("[pilot] Running low-rank projection sweep", flush=True)
        available_rank = basis.shape[1]
        if available_rank < max(ranks):
            raise RuntimeError(
                f"Low-rank basis only has rank {available_rank}, which is insufficient for the requested sweep."
            )
        for rank in ranks:
            eval_result = evaluate_model(
                model,
                tokenizer,
                examples,
                answers,
                device,
                collator,
                perturbation=make_perturber("lowrank", rank, basis=basis, mode="all"),
                capture_thoughts=False,
                desc=f"rank-{rank}",
            )
            results["lowrank"].append({"rank": rank, "accuracy": eval_result["accuracy"]})
        write_json(output_dir / "results.json", results)

        control_specs = [
            ("quantize_4bit", make_perturber("quantization", 4)),
            ("noise_0.1", make_perturber("gaussian_noise", 0.1)),
            ("rank_64", make_perturber("lowrank", 64, basis=basis)),
        ]
        print("[pilot] Running first-thought and last-thought controls", flush=True)
        for control_name, base_perturber in control_specs:
            for mode_key, mode_name in [("first_thought_only", "first"), ("last_thought_only", "last")]:
                kind = control_name.split("_")[0]
                if kind == "quantize":
                    perturber = make_perturber("quantization", 4, mode=mode_name)
                elif kind == "noise":
                    perturber = make_perturber("gaussian_noise", 0.1, mode=mode_name)
                else:
                    perturber = make_perturber("lowrank", 64, basis=basis, mode=mode_name)
                eval_result = evaluate_model(
                    model,
                    tokenizer,
                    examples,
                    answers,
                    device,
                    collator,
                    perturbation=perturber,
                    capture_thoughts=False,
                    noise_seed=args.seed + 999 if control_name == "noise_0.1" else None,
                    desc=f"{mode_key}-{control_name}",
                )
                results["controls"][mode_key][control_name] = eval_result["accuracy"]
        write_json(output_dir / "results.json", results)

        sweep_minutes = (time.perf_counter() - sweep_start) / 60.0

    finalize_start = time.perf_counter()
    write_json(output_dir / "results.json", results)

    if not args.skip_sweep:
        save_plot_quantization(
            output_dir, results["quantization"], results["theoretical_thresholds"]["bits"]
        )
        save_plot_noise(
            output_dir, results["gaussian_noise"], results["theoretical_thresholds"]["sigma"]
        )
        save_plot_lowrank(
            output_dir, results["lowrank"], results["theoretical_thresholds"]["rank"]
        )

    finalize_minutes = (time.perf_counter() - finalize_start) / 60.0
    write_pilot_log(
        output_dir,
        args,
        checkpoint_path,
        checkpoint_note,
        {
            "setup": setup_minutes,
            "baseline": baseline_minutes,
            "sweep": sweep_minutes,
            "finalize": finalize_minutes,
        },
        results,
        deviations,
    )


if __name__ == "__main__":
    main()
