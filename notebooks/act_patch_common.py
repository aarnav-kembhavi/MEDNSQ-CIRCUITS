"""Shared activation-patching utilities for anchor sufficiency testing.

Usage: each per-model script imports `run_patching_experiment` and calls it
with the model, tokenizer, anchors, and configuration.
"""

import gc
import json
import os
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PatchConfig:
    n_medqa: int = 300
    n_medmcqa: int = 300
    n_pubmedqa: int = 300
    n_patch_samples: int = 300
    n_random_trials: int = 10
    seeds: List[int] = field(default_factory=lambda: [42, 123, 7, 13, 97])
    max_contexts: int = 3
    max_context_chars: int = 2200
    output_dir: str = "patch_results"
    cache_dir: str = "."


# ---------------------------------------------------------------------------
# Dataset helpers (reused from multi_model_anchor_eval)
# ---------------------------------------------------------------------------

def _get_letter_token_ids(tokenizer) -> Dict[str, int]:
    ids = {}
    for letter in ["A", "B", "C", "D"]:
        encoded = tokenizer(letter, add_special_tokens=False)["input_ids"]
        ids[letter] = encoded[0] if not isinstance(encoded[0], list) else encoded[0][0]
    return ids


def _format_medqa_prompt(question: str, options: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    opts = "\n".join(f"{l}. {t}" for l, t in zip(letters, options))
    return f"Question: {question}\nOptions:\n{opts}\nAnswer:"


def _format_pubmed_prompt(question: str, contexts: List[str], cfg: PatchConfig) -> str:
    ctx = "\n".join(f"- {c}" for c in contexts[:cfg.max_contexts] if c)
    if len(ctx) > cfg.max_context_chars:
        ctx = ctx[:cfg.max_context_chars]
    return (
        "You are answering a biomedical multiple-choice question.\n"
        f"Question: {question}\n"
        f"Context:\n{ctx}\n\n"
        "Options:\n"
        "(A) yes\n"
        "(B) no\n"
        "(C) insufficient information\n"
        "(D) contradictory evidence\n"
        "Answer: ("
    )


def build_medqa_pairs(model, tokenizer, n_total, device, letter_ids):
    ds = load_dataset("openlifescienceai/medqa")["train"]
    n_total = min(n_total, len(ds))
    pairs = []
    for i in range(n_total):
        row = ds[i]
        q = row["data"]["Question"]
        opts_raw = row["data"]["Options"]
        options = [opts_raw["A"], opts_raw["B"], opts_raw["C"], opts_raw["D"]]
        correct_idx = "ABCD".index(row["data"]["Correct Option"])
        prompt = _format_medqa_prompt(q, options)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits[0, -1, :].float()
        pos_letter = "ABCD"[correct_idx]
        pos_id = letter_ids[pos_letter]
        wrong = [(l, letter_ids[l]) for l in "ABCD" if l != pos_letter]
        wrong_logits = logits[torch.tensor([w[1] for w in wrong], device=device)]
        neg_id = wrong[int(torch.argmax(wrong_logits).item())][1]

        # Build safe prompt (swap correct and distractor positions)
        neg_local_idx = [i for i, (l, _) in enumerate(wrong) if _ == neg_id][0]
        neg_letter_idx = "ABCD".index(wrong[neg_local_idx][0])
        safe_options = list(options)
        safe_options[correct_idx], safe_options[neg_letter_idx] = safe_options[neg_letter_idx], safe_options[correct_idx]
        safe_prompt = _format_medqa_prompt(q, safe_options)
        safe_enc = tokenizer(safe_prompt, return_tensors="pt", add_special_tokens=True)

        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "safe_input_ids": safe_enc["input_ids"],
            "safe_attention_mask": safe_enc["attention_mask"],
            "pos_id": pos_id,
            "neg_id": neg_id,
        })
    return pairs


def build_medmcqa_pairs(model, tokenizer, n_total, device, letter_ids):
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    pairs = []
    for row in ds:
        try:
            cop = int(row.get("cop", -1))
        except (TypeError, ValueError):
            continue
        if cop not in idx_to_letter:
            continue
        prompt = (
            f"Question: {str(row.get('question', '')).strip()}\n"
            f"Options: (A) {str(row.get('opa', '')).strip()} "
            f"(B) {str(row.get('opb', '')).strip()} "
            f"(C) {str(row.get('opc', '')).strip()} "
            f"(D) {str(row.get('opd', '')).strip()}\n"
            "Answer: ("
        )
        enc = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits[0, -1, :].float()
        correct = idx_to_letter[cop]
        pos_id = letter_ids[correct]
        wrong = [(l, letter_ids[l]) for l in "ABCD" if l != correct]
        wrong_logits = logits[torch.tensor([w[1] for w in wrong], device=device)]
        neg_id = wrong[int(torch.argmax(wrong_logits).item())][1]
        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "safe_input_ids": enc["input_ids"].clone(),
            "safe_attention_mask": enc["attention_mask"].clone(),
            "pos_id": pos_id,
            "neg_id": neg_id,
        })
        if len(pairs) >= n_total:
            break
    return pairs


def build_pubmedqa_pairs(model, tokenizer, n_total, device, letter_ids, cfg: PatchConfig):
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    pairs = []
    for row in ds:
        gold = str(row.get("final_decision", "")).strip().lower()
        if gold not in {"yes", "no"}:
            continue
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        ctx_obj = row.get("context", {})
        contexts = ctx_obj.get("contexts", []) if isinstance(ctx_obj, dict) else []
        prompt = _format_pubmed_prompt(question, contexts, cfg)
        enc = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits[0, -1, :].float()
        pos_letter = "A" if gold == "yes" else "B"
        pos_id = letter_ids[pos_letter]
        wrong = [(l, letter_ids[l]) for l in "ABCD" if l != pos_letter]
        wrong_logits = logits[torch.tensor([w[1] for w in wrong], device=device)]
        neg_id = wrong[int(torch.argmax(wrong_logits).item())][1]
        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "safe_input_ids": enc["input_ids"].clone(),
            "safe_attention_mask": enc["attention_mask"].clone(),
            "pos_id": pos_id,
            "neg_id": neg_id,
        })
        if len(pairs) >= n_total:
            break
    return pairs


# ---------------------------------------------------------------------------
# Core patching logic
# ---------------------------------------------------------------------------

def _get_backbone(model):
    backbone = getattr(model, "model", model)
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    return backbone


def _left_pad(ids, mask, target_len, pad_id, device):
    seq_len = ids.shape[1]
    if seq_len >= target_len:
        return ids.to(device), mask.to(device)
    b = ids.shape[0]
    new_ids = torch.full((b, target_len), pad_id, dtype=ids.dtype, device=device)
    new_mask = torch.zeros(b, target_len, dtype=mask.dtype, device=device)
    new_ids[:, -seq_len:] = ids.to(device)
    new_mask[:, -seq_len:] = mask.to(device)
    return new_ids, new_mask


def patch_run(
    model,
    layer_stack,
    anchors_by_layer: Dict[int, List[int]],
    pairs: List[Dict],
    n_samples: int,
    letter_token_ids: torch.Tensor,
    pad_id: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Zero-then-restore sufficiency test.

    For each sample (same prompt throughout):
      1. Clean run   → save anchor activations, record clean margin
      2. Corrupt run → zero out anchor activations, record degraded margin
      3. Restore run → inject saved activations back, record restored margin

    Reports:
      - degrade_shift = corrupt_margin - clean_margin  (should be negative)
      - restore_shift = restored_margin - corrupt_margin  (should be positive)
    """

    degrade_shifts = []
    restore_shifts = []
    correct_clean = 0
    correct_corrupt = 0
    correct_restored = 0

    def _make_save_hooks(stored, layers_cols):
        handles = []
        for layer_idx, cols in layers_cols.items():
            down_proj = layer_stack[layer_idx].mlp.down_proj

            def _hook(module, inp, _layer=layer_idx, _cols=cols):
                h = inp[0]
                for c in _cols:
                    stored[(_layer, c)] = h[0, -1, c].detach().clone()

            handles.append(down_proj.register_forward_pre_hook(_hook))
        return handles

    def _make_zero_hooks(layers_cols):
        """Zero out anchor activations (corruption)."""
        handles = []
        for layer_idx, cols in layers_cols.items():
            down_proj = layer_stack[layer_idx].mlp.down_proj

            def _hook(module, inp, _layer=layer_idx, _cols=cols):
                h = inp[0]
                out = h.clone()
                for c in _cols:
                    out[0, -1, c] = 0.0
                return (out,)

            handles.append(down_proj.register_forward_pre_hook(_hook))
        return handles

    def _make_restore_hooks(stored, layers_cols):
        """Restore saved anchor activations."""
        handles = []
        for layer_idx, cols in layers_cols.items():
            down_proj = layer_stack[layer_idx].mlp.down_proj

            def _hook(module, inp, _layer=layer_idx, _cols=cols):
                h = inp[0]
                out = h.clone()
                for c in _cols:
                    key = (_layer, c)
                    if key in stored:
                        out[0, -1, c] = stored[key].to(out.dtype)
                return (out,)

            handles.append(down_proj.register_forward_pre_hook(_hook))
        return handles

    def _eval_logits(logits, pair):
        margin = (logits[0, pair["pos_id"]] - logits[0, pair["neg_id"]]).item()
        pred = letter_token_ids[torch.argmax(logits[0, letter_token_ids]).item()].item()
        correct = int(pred == pair["pos_id"])
        return margin, correct

    with torch.no_grad():
        for i in range(min(n_samples, len(pairs))):
            pair = pairs[i]
            input_ids = pair["input_ids"].to(device)
            attn_mask = pair["attention_mask"].to(device)

            # --- Step 1: Clean run (save anchor activations) ---
            stored: Dict[Tuple[int, int], torch.Tensor] = {}
            handles = _make_save_hooks(stored, anchors_by_layer)
            logits_clean = model(input_ids=input_ids, attention_mask=attn_mask).logits[:, -1, :]
            for h in handles:
                h.remove()
            clean_margin, clean_correct = _eval_logits(logits_clean, pair)
            correct_clean += clean_correct

            # --- Step 2: Corrupt run (zero anchor activations) ---
            handles = _make_zero_hooks(anchors_by_layer)
            logits_corrupt = model(input_ids=input_ids, attention_mask=attn_mask).logits[:, -1, :]
            for h in handles:
                h.remove()
            corrupt_margin, corrupt_correct = _eval_logits(logits_corrupt, pair)
            correct_corrupt += corrupt_correct

            # --- Step 3: Restore run (put saved activations back) ---
            handles = _make_restore_hooks(stored, anchors_by_layer)
            logits_restored = model(input_ids=input_ids, attention_mask=attn_mask).logits[:, -1, :]
            for h in handles:
                h.remove()
            restored_margin, restored_correct = _eval_logits(logits_restored, pair)
            correct_restored += restored_correct

            degrade_shifts.append(corrupt_margin - clean_margin)
            restore_shifts.append(restored_margin - corrupt_margin)

    n = min(n_samples, len(pairs))
    return {
        "mean_degrade_shift": float(np.mean(degrade_shifts)),
        "std_degrade_shift": float(np.std(degrade_shifts)),
        "mean_restore_shift": float(np.mean(restore_shifts)),
        "std_restore_shift": float(np.std(restore_shifts)),
        "acc_clean": correct_clean / n if n else 0,
        "acc_corrupt": correct_corrupt / n if n else 0,
        "acc_restored": correct_restored / n if n else 0,
        "acc_degrade": (correct_corrupt - correct_clean) / n if n else 0,
        "acc_restore": (correct_restored - correct_corrupt) / n if n else 0,
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_patching_experiment(
    model_id: str,
    model_key: str,
    anchors: List[Tuple[int, int]],
    cfg: PatchConfig,
    trust_remote_code: bool = False,
):
    """Full activation patching experiment for one model across 3 datasets × N seeds."""

    print(f"\n{'='*60}")
    print(f"Activation Patching: {model_key} ({model_id})")
    print(f"Anchors: {len(anchors)}, Seeds: {cfg.seeds}")
    print(f"{'='*60}\n")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    device = next(model.parameters()).device
    backbone = _get_backbone(model)
    layer_stack = backbone.layers
    intermediate_size = layer_stack[0].mlp.down_proj.weight.shape[1]
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    letter_ids_map = _get_letter_token_ids(tokenizer)
    letter_token_ids = torch.tensor([letter_ids_map[l] for l in "ABCD"], dtype=torch.long, device=device)

    # Group anchors by layer
    anchors_by_layer: Dict[int, List[int]] = {}
    for (layer_idx, col_idx) in anchors:
        anchors_by_layer.setdefault(layer_idx, []).append(col_idx)

    # Build datasets (once, outside seed loop)
    print("Building MedQA pairs...")
    medqa_pairs = build_medqa_pairs(model, tokenizer, cfg.n_medqa, device, letter_ids_map)
    print(f"  Built {len(medqa_pairs)} MedQA pairs")

    print("Building MedMCQA pairs...")
    medmcqa_pairs = build_medmcqa_pairs(model, tokenizer, cfg.n_medmcqa, device, letter_ids_map)
    print(f"  Built {len(medmcqa_pairs)} MedMCQA pairs")

    print("Building PubMedQA pairs...")
    pubmedqa_pairs = build_pubmedqa_pairs(model, tokenizer, cfg.n_pubmedqa, device, letter_ids_map, cfg)
    print(f"  Built {len(pubmedqa_pairs)} PubMedQA pairs")

    datasets = {
        "medqa": medqa_pairs,
        "medmcqa": medmcqa_pairs,
        "pubmedqa": pubmedqa_pairs,
    }

    all_results = {
        "metadata": {
            "model_id": model_id,
            "model_key": model_key,
            "n_anchors": len(anchors),
            "anchors": anchors,
            "config": asdict(cfg),
            "timestamp": datetime.now().isoformat(),
        },
        "seeds": {},
    }

    for seed in cfg.seeds:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        seed_results: Dict[str, Any] = {}

        for ds_name, pairs in datasets.items():
            print(f"  [{ds_name}] Patching with {len(anchors)} anchors on {min(cfg.n_patch_samples, len(pairs))} samples...")

            # Shuffle pairs for this seed
            shuffled = list(pairs)
            random.shuffle(shuffled)

            # Anchor patching
            anchor_result = patch_run(
                model, layer_stack, anchors_by_layer, shuffled,
                cfg.n_patch_samples, letter_token_ids, pad_id, device,
            )

            # Random neuron control (same number of neurons, from same layers)
            anchor_layers = list(anchors_by_layer.keys())
            random_results = []
            for trial in range(cfg.n_random_trials):
                rand_neurons = set()
                while len(rand_neurons) < len(anchors):
                    rand_neurons.add((random.choice(anchor_layers), random.randint(0, intermediate_size - 1)))
                rand_by_layer: Dict[int, List[int]] = {}
                for (l, c) in rand_neurons:
                    rand_by_layer.setdefault(l, []).append(c)

                rand_result = patch_run(
                    model, layer_stack, rand_by_layer, shuffled,
                    cfg.n_patch_samples, letter_token_ids, pad_id, device,
                )
                random_results.append(rand_result)

            # Compute Cohen's d: anchor degrade vs random degrade
            anchor_degrade = anchor_result["mean_degrade_shift"]
            random_degrades = [r["mean_degrade_shift"] for r in random_results]
            mean_random_degrade = float(np.mean(random_degrades))
            std_random_degrade = float(np.std(random_degrades))

            anchor_restore = anchor_result["mean_restore_shift"]
            random_restores = [r["mean_restore_shift"] for r in random_results]
            mean_random_restore = float(np.mean(random_restores))
            std_random_restore = float(np.std(random_restores))

            # d for degradation (anchor should degrade MORE, i.e. more negative)
            if std_random_degrade > 1e-9:
                d_degrade = (anchor_degrade - mean_random_degrade) / std_random_degrade
            else:
                d_degrade = float("-inf") if anchor_degrade < mean_random_degrade else 0.0

            # d for restoration (anchor should restore MORE, i.e. more positive)
            if std_random_restore > 1e-9:
                d_restore = (anchor_restore - mean_random_restore) / std_random_restore
            else:
                d_restore = float("inf") if anchor_restore > mean_random_restore else 0.0

            seed_results[ds_name] = {
                "anchor": anchor_result,
                "random_mean_degrade": mean_random_degrade,
                "random_std_degrade": std_random_degrade,
                "random_mean_restore": mean_random_restore,
                "random_std_restore": std_random_restore,
                "cohens_d_degrade": d_degrade,
                "cohens_d_restore": d_restore,
                "random_trials": random_results,
            }

            print(f"    Anchor degrade: {anchor_degrade:+.4f} | restore: {anchor_restore:+.4f}")
            print(f"    Acc: clean={anchor_result['acc_clean']:.3f} → corrupt={anchor_result['acc_corrupt']:.3f} → restored={anchor_result['acc_restored']:.3f}")
            print(f"    Random degrade: {mean_random_degrade:+.4f} ± {std_random_degrade:.4f}")
            print(f"    Cohen's d: degrade={d_degrade:+.3f}, restore={d_restore:+.3f}")

        all_results["seeds"][str(seed)] = seed_results

    # Aggregate across seeds
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS (mean across seeds)")
    print(f"{'='*60}")

    summary = {}
    for ds_name in ["medqa", "medmcqa", "pubmedqa"]:
        degrades = [all_results["seeds"][str(s)][ds_name]["anchor"]["mean_degrade_shift"] for s in cfg.seeds]
        restores = [all_results["seeds"][str(s)][ds_name]["anchor"]["mean_restore_shift"] for s in cfg.seeds]
        acc_cleans = [all_results["seeds"][str(s)][ds_name]["anchor"]["acc_clean"] for s in cfg.seeds]
        acc_corrupts = [all_results["seeds"][str(s)][ds_name]["anchor"]["acc_corrupt"] for s in cfg.seeds]
        acc_restoreds = [all_results["seeds"][str(s)][ds_name]["anchor"]["acc_restored"] for s in cfg.seeds]
        d_degs = [all_results["seeds"][str(s)][ds_name]["cohens_d_degrade"] for s in cfg.seeds]
        d_rests = [all_results["seeds"][str(s)][ds_name]["cohens_d_restore"] for s in cfg.seeds]

        summary[ds_name] = {
            "degrade_mean": float(np.mean(degrades)),
            "degrade_std": float(np.std(degrades)),
            "restore_mean": float(np.mean(restores)),
            "restore_std": float(np.std(restores)),
            "acc_clean_mean": float(np.mean(acc_cleans)),
            "acc_corrupt_mean": float(np.mean(acc_corrupts)),
            "acc_restored_mean": float(np.mean(acc_restoreds)),
            "d_degrade_mean": float(np.mean(d_degs)),
            "d_restore_mean": float(np.mean(d_rests)),
        }

        print(f"  {ds_name:>10}: degrade {np.mean(degrades):+.4f} | restore {np.mean(restores):+.4f} | "
              f"acc {np.mean(acc_cleans):.3f}→{np.mean(acc_corrupts):.3f}→{np.mean(acc_restoreds):.3f}")

    all_results["summary"] = summary

    # Save
    out_path = os.path.join(cfg.output_dir, f"{model_key}_activation_patching.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results