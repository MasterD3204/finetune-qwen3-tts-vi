# coding=utf-8
# Copyright 2026 The Alibaba Qwen team. (modified for Vietnamese language adaptation)
# SPDX-License-Identifier: Apache-2.0
#
# Vietnamese Fine-tuning Script for Qwen3-TTS
#
# Strategy:
#   Stage 1 (Language Adaptation):
#     - LoRA on Talker attention layers (Q/K/V/O projections)
#     - Freeze: speaker_encoder, speech_tokenizer, talker MLP layers
#     - Train: LoRA adapters + text_embedding
#     - Data: 80% Vietnamese general + 20% original language mix
#     - Goal: teach text→codec mapping for Vietnamese
#
#   Stage 2 (Voice Integration):  [optional, run after Stage 1]
#     - Merge LoRA weights, fine-tune with very low LR
#     - Data: target Vietnamese speaker data
#     - Goal: integrate voice identity with Vietnamese phonetics
#
# Usage (Stage 1):
#   python sft_vietnamese.py \
#     --stage 1 \
#     --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
#     --train_jsonl train_vi_general_with_codes.jsonl \
#     --output_model_path output_vi_stage1 \
#     --speaker_name vi_speaker \
#     --lora_rank 32 \
#     --num_epochs 5 \
#     --lr 1e-4
#
# Usage (Stage 2):
#   python sft_vietnamese.py \
#     --stage 2 \
#     --init_model_path output_vi_stage1/checkpoint-epoch-4 \
#     --train_jsonl train_vi_speaker_with_codes.jsonl \
#     --output_model_path output_vi_stage2 \
#     --speaker_name vi_speaker \
#     --lora_rank 16 \
#     --num_epochs 3 \
#     --lr 5e-6

import argparse
import json
import os
import shutil
from typing import Optional

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

# LoRA implementation (no extra deps required, pure PyTorch)
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# LoRA Layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adapter.
    Keeps original weight frozen, trains only lora_A and lora_B.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weight
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # LoRA parameters (trainable)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with kaiming uniform, B stays zero (so initial delta = 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA into base weight and return plain nn.Linear."""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        merged.weight.data = self.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        if self.bias is not None:
            merged.bias.data = self.bias.data.clone()
        return merged


# ---------------------------------------------------------------------------
# LoRA injection utilities
# ---------------------------------------------------------------------------

def inject_lora_into_attention(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> list:
    """
    Replace Q/K/V/O projection Linear layers in all attention modules of the
    talker Transformer with LoRALinear. Returns list of replaced module paths.

    Target layers (by name pattern):
      - *.self_attn.q_proj
      - *.self_attn.k_proj
      - *.self_attn.v_proj
      - *.self_attn.o_proj
    """
    replaced = []
    for name, module in model.named_modules():
        # Only inject into talker transformer attention projections
        if not name.startswith("talker."):
            continue
        parts = name.split(".")
        if len(parts) < 2:
            continue
        if parts[-1] not in ("q_proj", "k_proj", "v_proj", "o_proj"):
            continue
        if "self_attn" not in parts:
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Navigate to parent and replace
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, parts[-1], lora_layer)
        replaced.append(name)

    return replaced


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters from model state_dict."""
    return {k: v for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def merge_lora_weights(model: nn.Module) -> None:
    """
    In-place: replace all LoRALinear modules with merged nn.Linear.
    After calling this, the model has no LoRA overhead and can be saved normally.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        merged = module.merge_weights()
        # Cast to same dtype as original
        merged = merged.to(module.weight.dtype)
        setattr(parent, parts[-1], merged)


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

def freeze_module(module: nn.Module, name: str = "") -> int:
    """Freeze all parameters in a module. Returns count of frozen params."""
    count = 0
    for p in module.parameters():
        p.requires_grad_(False)
        count += p.numel()
    return count


def freeze_non_lora_talker(model: nn.Module) -> None:
    """
    Freeze everything in the talker except:
      - LoRA adapters (lora_A, lora_B)
      - text_embedding  (to allow learning Vietnamese text patterns)

    Specifically freezes:
      - talker MLP (gate_proj, up_proj, down_proj)
      - talker layer norms
      - base attention weights (already frozen by LoRALinear, but explicit here)
      - speaker_encoder
      - speech_tokenizer
    """
    # Freeze speaker encoder
    if hasattr(model, "speaker_encoder"):
        frozen = freeze_module(model.speaker_encoder, "speaker_encoder")
        print(f"[freeze] speaker_encoder: {frozen:,} params frozen")

    # Freeze speech tokenizer if loaded
    if hasattr(model, "speech_tokenizer") and model.speech_tokenizer is not None:
        frozen = freeze_module(model.speech_tokenizer, "speech_tokenizer")
        print(f"[freeze] speech_tokenizer: {frozen:,} params frozen")

    # Freeze talker MLP layers and layer norms selectively
    if hasattr(model, "talker"):
        talker = model.talker
        for name, sub in talker.named_modules():
            # Keep LoRA layers trainable
            if isinstance(sub, LoRALinear):
                continue
            # Keep text_embedding trainable
            if "text_embedding" in name:
                continue
            # Freeze everything else
            for p in sub.parameters(recurse=False):
                p.requires_grad_(False)


def print_trainable_params(model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[params] Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")


# ---------------------------------------------------------------------------
# Config patching for Vietnamese
# ---------------------------------------------------------------------------

VIETNAMESE_LANGUAGE_ID = 5  # use a free slot; verify against your model config

def patch_config_for_vietnamese(config_dict: dict, speaker_name: str, speaker_slot: int = 3000) -> dict:
    """
    Add Vietnamese language ID and speaker mapping to config.
    Modifies in-place and returns updated dict.
    """
    config_dict["tts_model_type"] = "custom_voice"

    talker_config = config_dict.get("talker_config", {})

    # Register Vietnamese language if not present
    lang_ids = talker_config.get("codec_language_id", {})
    if "vi" not in lang_ids and "Vietnamese" not in lang_ids:
        # Find max existing ID and add one (or use VIETNAMESE_LANGUAGE_ID)
        existing_ids = list(lang_ids.values()) if lang_ids else []
        new_id = max(existing_ids) + 1 if existing_ids else VIETNAMESE_LANGUAGE_ID
        lang_ids["vi"] = new_id
        lang_ids["Vietnamese"] = new_id
        talker_config["codec_language_id"] = lang_ids
        print(f"[config] Added Vietnamese language ID: {new_id}")
    else:
        print(f"[config] Vietnamese language ID already exists: {lang_ids.get('vi', lang_ids.get('Vietnamese'))}")

    # Register speaker
    spk_id = talker_config.get("spk_id", {})
    spk_id[speaker_name] = speaker_slot
    talker_config["spk_id"] = spk_id

    spk_is_dialect = talker_config.get("spk_is_dialect", {})
    spk_is_dialect[speaker_name] = False
    talker_config["spk_is_dialect"] = spk_is_dialect

    config_dict["talker_config"] = talker_config
    return config_dict


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

target_speaker_embedding: Optional[torch.Tensor] = None


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser(description="Vietnamese fine-tuning for Qwen3-TTS")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output_vi")
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="Path to train_with_codes.jsonl (output of prepare_data.py)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Stage 1: 1e-4 (LoRA), Stage 2: 5e-6 (full fine-tune)")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--speaker_name", type=str, default="vi_speaker",
                        help="Name for the Vietnamese speaker slot")
    parser.add_argument("--speaker_slot", type=int, default=3000,
                        help="Codec embedding slot index for new speaker")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Training stage: 1=LoRA language adaptation, 2=full fine-tune voice integration")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank (Stage 1: 32, Stage 2: 16)")
    parser.add_argument("--lora_alpha", type=float, default=64.0,
                        help="LoRA alpha = 2 * rank is a good default")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every_n_steps", type=int, default=0,
                        help="Save checkpoint every N steps (0 = epoch only)")
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
    )

    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"Vietnamese TTS Fine-tuning — Stage {args.stage}")
    accelerator.print(f"  Model: {args.init_model_path}")
    accelerator.print(f"  Data:  {args.train_jsonl}")
    accelerator.print(f"  Output: {args.output_model_path}")
    accelerator.print(f"  LR: {args.lr} | LoRA rank: {args.lora_rank} | Epochs: {args.num_epochs}")
    accelerator.print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(args.init_model_path)
    model = qwen3tts.model

    # -----------------------------------------------------------------------
    # Stage-specific setup
    # -----------------------------------------------------------------------
    if args.stage == 1:
        accelerator.print("[Stage 1] Injecting LoRA into talker attention layers...")
        replaced = inject_lora_into_attention(model, rank=args.lora_rank, alpha=args.lora_alpha)
        accelerator.print(f"  Injected LoRA into {len(replaced)} projections")

        accelerator.print("[Stage 1] Freezing non-LoRA parameters...")
        freeze_non_lora_talker(model)

    elif args.stage == 2:
        accelerator.print("[Stage 2] Full fine-tune with very low LR — no LoRA injection")
        accelerator.print("[Stage 2] Freezing speaker_encoder and speech_tokenizer only...")
        if hasattr(model, "speaker_encoder"):
            freeze_module(model.speaker_encoder)
        if hasattr(model, "speech_tokenizer") and model.speech_tokenizer is not None:
            freeze_module(model.speech_tokenizer)

    print_trainable_params(model)

    # -----------------------------------------------------------------------
    # Dataset & DataLoader
    # -----------------------------------------------------------------------
    train_data_lines = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data_lines]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # -----------------------------------------------------------------------
    # Optimizer — only over trainable parameters
    # -----------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    model.train()

    total_steps = len(train_dataloader) * args.num_epochs
    accelerator.print(f"[train] Total steps: {total_steps}")

    global_step = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                # Extract speaker embedding (detached — speaker encoder is frozen)
                with torch.no_grad():
                    speaker_embedding = model.speaker_encoder(
                        ref_mels.to(model.device).to(model.dtype)
                    )

                # Keep first speaker embedding as target (for saving after training)
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding.detach().cpu()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # Build input embeddings
                input_text_embedding = (
                    model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                )
                input_codec_embedding = (
                    model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                )
                # Insert speaker embedding at dedicated position (index 6)
                input_codec_embedding[:, 6, :] = speaker_embedding.detach()

                input_embeddings = input_text_embedding + input_codec_embedding

                # Add multi-codebook embeddings from code_predictor
                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # Forward pass through talker
                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                # Sub-talker (code predictor) loss for multi-codebook quality
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]
                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )

                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch+1}/{args.num_epochs} | Step {step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Primary: {outputs.loss.item():.4f} | "
                    f"SubTalker: {sub_talker_loss.item():.4f}"
                )

            # Intermediate checkpoint
            if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                if accelerator.is_main_process:
                    _save_checkpoint(
                        accelerator, model, config, args,
                        tag=f"step-{global_step}",
                    )

        avg_loss = epoch_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch+1} complete | Avg loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            _save_checkpoint(
                accelerator, model, config, args,
                tag=f"epoch-{epoch}",
            )

    accelerator.print("\n[done] Training complete.")


def _save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config,
    args: argparse.Namespace,
    tag: str,
) -> None:
    """Save a checkpoint to disk with updated config and merged weights."""
    global target_speaker_embedding

    output_dir = os.path.join(args.output_model_path, f"checkpoint-{tag}")
    shutil.copytree(args.init_model_path, output_dir, dirs_exist_ok=True)

    # Patch config.json for Vietnamese + new speaker
    input_config_file = os.path.join(args.init_model_path, "config.json")
    output_config_file = os.path.join(output_dir, "config.json")
    with open(input_config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config_dict = patch_config_for_vietnamese(config_dict, args.speaker_name, args.speaker_slot)

    with open(output_config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # Unwrap model and merge LoRA (if Stage 1)
    unwrapped_model = accelerator.unwrap_model(model)

    if args.stage == 1:
        # Deep copy model to avoid mutating the training instance
        import copy
        model_copy = copy.deepcopy(unwrapped_model)
        merge_lora_weights(model_copy)
        state_dict = {k: v.detach().to("cpu") for k, v in model_copy.state_dict().items()}
    else:
        state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

    # Drop speaker_encoder weights (not needed in inference checkpoint)
    keys_to_drop = [k for k in state_dict if k.startswith("speaker_encoder")]
    for k in keys_to_drop:
        del state_dict[k]

    # Write learned speaker embedding into codec_embedding slot
    if target_speaker_embedding is not None:
        weight = state_dict["talker.model.codec_embedding.weight"]
        state_dict["talker.model.codec_embedding.weight"][args.speaker_slot] = (
            target_speaker_embedding[0].to(weight.device).to(weight.dtype)
        )

    save_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, save_path)
    print(f"[save] Checkpoint saved to {output_dir}")


if __name__ == "__main__":
    train()
