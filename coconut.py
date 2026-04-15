# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.thought_perturber = None
        self.capture_thoughts = False
        self.captured_thoughts = []
        self.kv_cache_policy = "full"

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def reset_thought_trace(self):
        self.captured_thoughts = []

    def _capture_thought(self, thought):
        if self.capture_thoughts:
            self.captured_thoughts.append(thought.detach().cpu())

    def _perturb_thought(self, thought, pass_idx, max_n_latents, batch_idx, token_idx):
        if self.thought_perturber is None:
            return thought
        return self.thought_perturber(
            thought,
            pass_idx=pass_idx,
            max_n_latents=max_n_latents,
            batch_idx=batch_idx,
            token_idx=token_idx,
        )

    def _past_key_values_for_range(self, kv_cache, past_end, keep_start=0):
        return [
            (
                k[:, :, keep_start:past_end, :],
                v[:, :, keep_start:past_end, :],
            )
            for k, v in kv_cache
        ]

    def _attention_mask_for_range(self, attention_mask, range_start, range_end, keep_start=0):
        if keep_start == 0:
            return attention_mask[:, :range_end]
        return torch.cat(
            [
                attention_mask[:, keep_start:range_start],
                attention_mask[:, range_start:range_end],
            ],
            dim=1,
        )

    def _kv_keep_start(self, latent_start):
        if self.kv_cache_policy == "full":
            return 0
        if self.kv_cache_policy == "latent_only":
            return latent_start
        if self.kv_cache_policy.startswith("prompt_last_"):
            n_prompt_tokens = int(self.kv_cache_policy.removeprefix("prompt_last_"))
            return max(0, latent_start - n_prompt_tokens)
        raise ValueError(f"Unknown kv_cache_policy: {self.kv_cache_policy}")

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            latent_start = next_compute_range[1]
            # before the earliest latent token position
        else:
            latent_start = 0

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                keep_start = self._kv_keep_start(latent_start)
                past_key_values = self._past_key_values_for_range(
                    kv_cache, next_compute_range[0], keep_start=keep_start
                )
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=self._attention_mask_for_range(
                        attention_mask,
                        next_compute_range[0],
                        next_compute_range[1],
                        keep_start=keep_start,
                    ),
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]
                logits.append(outputs.logits)
                # when we use kv_cache for previous tokens, those tokens are skipped
                # in `outputs.hidden_states`, so we keep this offset to index thoughts

            if kv_cache == None:
                logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                thought = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
                self._capture_thought(thought)
                tensor_list[batch_idx][token_idx] = self._perturb_thought(
                    thought,
                    pass_idx=pass_idx,
                    max_n_latents=max_n_latents,
                    batch_idx=batch_idx,
                    token_idx=token_idx,
                )

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        if not kv_cache:
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[
                    :, next_compute_range[0] : next_compute_range[1], :
                ],
                attention_mask=attention_mask[:, : next_compute_range[1]],
                position_ids=position_ids[
                    :, next_compute_range[0] : next_compute_range[1]
                ],
                past_key_values=(
                    self._past_key_values_for_range(
                        kv_cache, next_compute_range[0], keep_start=0
                    )
                    if kv_cache
                    else None
                ),
                output_hidden_states=True,
            )
            logits.append(outputs.logits)
        else:
            keep_start = self._kv_keep_start(latent_start)
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[
                    :, next_compute_range[0] : next_compute_range[1], :
                ],
                attention_mask=self._attention_mask_for_range(
                    attention_mask,
                    next_compute_range[0],
                    next_compute_range[1],
                    keep_start=keep_start,
                ),
                position_ids=position_ids[
                    :, next_compute_range[0] : next_compute_range[1]
                ],
                past_key_values=self._past_key_values_for_range(
                    kv_cache, next_compute_range[0], keep_start=keep_start
                ),
                output_hidden_states=True,
            )
            logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)
