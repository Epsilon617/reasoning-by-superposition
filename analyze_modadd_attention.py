import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dataset import MyCollator
from pilot_capacity import build_eval_examples, build_model, load_raw_examples
from stokenizer import STokenizer
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate latent-step attention mass for ModAdd models."
    )
    parser.add_argument("--standard-checkpoint", required=True)
    parser.add_argument("--isolated-checkpoint", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pass-idx", type=int, default=2)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def key_groups(latent_start, query_abs_pos, key_abs_positions):
    groups = {"prompt": [], "previous_latents": [], "self": []}
    for idx, pos in enumerate(key_abs_positions):
        if pos < latent_start:
            groups["prompt"].append(idx)
        elif pos < query_abs_pos:
            groups["previous_latents"].append(idx)
        elif pos == query_abs_pos:
            groups["self"].append(idx)
    return groups


def run_attention_trace(model, batch, pass_idx_to_capture):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]

    latent_indices = (input_ids == model.latent_token_id).nonzero()
    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents = max(len(items) for items in latent_lists)
    next_compute_range = (0, input_ids.shape[1])
    inputs_embeds = model.embedding(input_ids)

    if max_n_latents > 0:
        next_compute_range = (0, latent_indices[:, 1].min().item())
        latent_start = next_compute_range[1]
    else:
        latent_start = 0

    kv_cache = None
    captured = None

    for pass_idx in range(max_n_latents):
        if kv_cache is None:
            call_start, call_end = next_compute_range
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, call_start:call_end, :],
                attention_mask=attention_mask[:, call_start:call_end],
                position_ids=position_ids[:, call_start:call_end],
                output_hidden_states=True,
                output_attentions=True,
            )
            hidden_states_offset = 0
            key_abs_positions = list(range(call_start, call_end))
            query_abs_positions = list(range(call_start, call_end))
        elif model.kv_cache_policy == "full":
            call_start, call_end = next_compute_range
            past_key_values = model._past_key_values_for_range(
                kv_cache, call_start, keep_start=0
            )
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, call_start:call_end, :],
                attention_mask=attention_mask[:, :call_end],
                position_ids=position_ids[:, call_start:call_end],
                past_key_values=past_key_values,
                output_hidden_states=True,
                output_attentions=True,
            )
            hidden_states_offset = call_start
            key_abs_positions = list(range(0, call_end))
            query_abs_positions = list(range(call_start, call_end))
        elif model.kv_cache_policy == "latent_only":
            call_start, call_end = latent_start, next_compute_range[1]
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, call_start:call_end, :],
                attention_mask=attention_mask[:, call_start:call_end],
                position_ids=position_ids[:, call_start:call_end],
                output_hidden_states=True,
                output_attentions=True,
            )
            hidden_states_offset = latent_start
            key_abs_positions = list(range(call_start, call_end))
            query_abs_positions = list(range(call_start, call_end))
        else:
            raise ValueError(f"Unknown kv_cache_policy: {model.kv_cache_policy}")

        if pass_idx == pass_idx_to_capture:
            query_abs_pos = latent_lists[0][pass_idx] - 1
            if query_abs_pos in query_abs_positions:
                query_rel = query_abs_positions.index(query_abs_pos)
            else:
                query_rel = len(query_abs_positions) - 1
                query_abs_pos = query_abs_positions[query_rel]

            groups = key_groups(latent_start, query_abs_pos, key_abs_positions)
            layer_values = []
            for layer_attn in outputs.attentions:
                # shape: batch, heads, query_len, key_len
                attn = layer_attn[0, :, query_rel, :].detach().float().cpu()
                head_values = {}
                for group_name, indices in groups.items():
                    if indices:
                        head_values[group_name] = attn[:, indices].sum(dim=1)
                    else:
                        head_values[group_name] = torch.zeros(attn.shape[0])
                layer_values.append(head_values)
            captured = {
                "pass_idx": pass_idx,
                "latent_start": latent_start,
                "query_abs_pos": query_abs_pos,
                "key_abs_positions": key_abs_positions,
                "groups": groups,
                "layers": layer_values,
            }

        next_compute_range = (
            next_compute_range[1],
            (
                input_ids.shape[1]
                if pass_idx + 1 >= max_n_latents
                else next_compute_range[1] + 1
            ),
        )

        hidden_states = outputs.hidden_states[-1]
        kv_cache = outputs.past_key_values

        tensor_list = [
            [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
            for batch_idx in range(inputs_embeds.shape[0])
        ]
        filling_indices = [
            (instance_idx, mask_list[pass_idx])
            for instance_idx, mask_list in enumerate(latent_lists)
            if len(mask_list) > pass_idx
        ]
        for batch_idx, token_idx in filling_indices:
            thought = hidden_states[batch_idx, token_idx - 1 - hidden_states_offset, :]
            tensor_list[batch_idx][token_idx] = thought
        inputs_embeds = torch.stack(
            [torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])]
        )

    if captured is None:
        raise RuntimeError(f"No attention captured for pass_idx={pass_idx_to_capture}")
    return captured


def summarize_model(model, tokenizer, examples, pass_idx, device, limit):
    collator = MyCollator(
        tokenizer,
        latent_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        label_pad_token_id=-100,
    )
    totals = {}
    count = 0
    for example in tqdm(examples[:limit], desc=f"attention {model.kv_cache_policy}"):
        batch = collator([example])
        batch = {key: value.to(device) for key, value in batch.items() if key != "idx"}
        with torch.no_grad():
            captured = run_attention_trace(model, batch, pass_idx)
        for layer_idx, layer_values in enumerate(captured["layers"]):
            for group_name, values in layer_values.items():
                key = f"layer_{layer_idx}/{group_name}"
                totals.setdefault(key, []).append(values.mean().item())
        count += 1
    return {
        "n_examples": count,
        "pass_idx": pass_idx,
        "means": {key: float(torch.tensor(values).mean().item()) for key, values in totals.items()},
    }


def plot_summary(summary, output_path):
    policies = ["standard_full_kv", "training_time_latent_only"]
    groups = ["prompt", "previous_latents", "self"]
    group_labels = ["Prompt", "Prev. latents", "Self"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    n_layers = 2

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.45), sharey=True, constrained_layout=True)
    for ax, policy in zip(axes, policies):
        x = range(n_layers)
        width = 0.24
        for offset, group, label, color in zip([-width, 0, width], groups, group_labels, colors):
            values = [summary[policy]["means"].get(f"layer_{layer}/{group}", 0.0) for layer in range(n_layers)]
            ax.bar([idx + offset for idx in x], values, width=width, label=label, color=color, edgecolor="black", linewidth=0.35)
        ax.set_title("Standard full-KV" if policy == "standard_full_kv" else "Training-time latent-only")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"Layer {idx}" for idx in x])
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    axes[0].set_ylabel("Attention mass at pass 2")
    axes[1].legend(frameon=False, loc="upper right")
    fig.savefig(output_path, bbox_inches="tight")


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    tokenizer = STokenizer()
    raw_examples = load_raw_examples(args.test_path)
    examples, _, _, _ = build_eval_examples(raw_examples, tokenizer, seed=args.seed)

    standard_model, _, _ = build_model(args.config_path, args.standard_checkpoint, device)
    standard_model.kv_cache_policy = "full"
    standard_model.eval()

    isolated_model, _, _ = build_model(args.config_path, args.isolated_checkpoint, device)
    isolated_model.kv_cache_policy = "latent_only"
    isolated_model.eval()

    summary = {
        "standard_full_kv": summarize_model(
            standard_model, tokenizer, examples, args.pass_idx, device, args.limit
        ),
        "training_time_latent_only": summarize_model(
            isolated_model, tokenizer, examples, args.pass_idx, device, args.limit
        ),
    }
    summary["config"] = vars(args)

    with open(output_dir / "attention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_summary(summary, output_dir / "attention_mass_pass2.pdf")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[attention] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
