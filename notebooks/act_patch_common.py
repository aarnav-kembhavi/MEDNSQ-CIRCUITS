"""
Empirical Margin Sensitivity (EMS) and Activation Patching Utilities.

This module provides core utilities for identifying and validating causal anchor 
neurons in Transformer-based Language Models. It implements the three-stage 
discovery pipeline for Empirical Margin Sensitivity (EMS) and the 
zero-then-restore activation patching protocol used in the paper.

Citation:
"From Behavioral Memorization to Mechanistic Sabotage: 
Neuron-Level Causal Analysis of SFT and RL in Medical LLMs" (NeurIPS 2025)
"""

import gc
import json
import os
import random
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for causal intervention experiments."""
    # Sample sizes for different datasets
    n_medqa: int = 300
    n_medmcqa: int = 300
    n_pubmedqa: int = 300
    
    # Patching hyperparams
    n_patch_samples: int = 120
    n_random_trials: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123])
    
    # Prompting limits
    max_contexts: int = 3
    max_context_chars: int = 2200
    
    # Paths
    output_dir: str = "results"
    cache_dir: str = "."

# ---------------------------------------------------------------------------
# Data Processing Utilities
# ---------------------------------------------------------------------------

def _get_choice_token_ids(tokenizer, choices: str = "ABCD") -> Dict[str, int]:
    """Resolve token IDs for multiple-choice labels."""
    ids = {}
    for char in choices:
        encoded = tokenizer(char, add_special_tokens=False)["input_ids"]
        ids[char] = encoded[0] if not isinstance(encoded[0], list) else encoded[0][0]
    return ids


def _format_standard_mcq(question: str, options: List[str]) -> str:
    """Format a 4-option MCQ prompt."""
    letters = ["A", "B", "C", "D"]
    opts = "\n".join(f"{l}. {t}" for l, t in zip(letters, options))
    return f"Question: {question}\nOptions:\n{opts}\nAnswer:"


def _format_context_mcq(question: str, contexts: List[str], cfg: ExperimentConfig) -> str:
    """Format an MCQ prompt with external context (e.g., PubMedQA)."""
    ctx = "\n".join(f"- {c}" for c in contexts[:cfg.max_contexts] if c)
    if len(ctx) > cfg.max_context_chars:
        ctx = ctx[:cfg.max_context_chars]
    return (
        "You are answering a biomedical multiple-choice question.\n"
        f"Question: {question}\n"
        f"Context:\n{ctx}\n\n"
        "Options:\n(A) yes\n(B) no\n(C) insufficient information\n(D) contradictory evidence\n"
        "Answer: ("
    )


# ---------------------------------------------------------------------------
# Dataset Builders
# ---------------------------------------------------------------------------

def build_dataset_pairs(
    dataset_name: str,
    model: torch.nn.Module,
    tokenizer: Any,
    n_total: int,
    device: torch.device,
    config: ExperimentConfig
) -> List[Dict[str, Any]]:
    """Generic entry point for building adversarial pairs from various medical datasets."""
    choice_ids = _get_choice_token_ids(tokenizer)
    
    if dataset_name == "medqa":
        return _build_medqa(model, tokenizer, n_total, device, choice_ids)
    elif dataset_name == "medmcqa":
        return _build_medmcqa(model, tokenizer, n_total, device, choice_ids)
    elif dataset_name == "pubmedqa":
        return _build_pubmedqa(model, tokenizer, n_total, device, choice_ids, config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _build_medqa(model, tokenizer, n_total, device, choice_ids):
    ds = load_dataset("openlifescienceai/medqa")["train"]
    n_total = min(n_total, len(ds))
    pairs = []
    for i in range(n_total):
        row = ds[i]
        q = row["data"]["Question"]
        opts_raw = row["data"]["Options"]
        options = [opts_raw["A"], opts_raw["B"], opts_raw["C"], opts_raw["D"]]
        correct_idx = "ABCD".index(row["data"]["Correct Option"])
        
        prompt = _format_standard_mcq(q, options)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits[0, -1, :].float()
            
        pos_letter = "ABCD"[correct_idx]
        pos_id = choice_ids[pos_letter]
        
        wrong = [(l, choice_ids[l]) for l in "ABCD" if l != pos_letter]
        wrong_logits = logits[torch.tensor([w[1] for w in wrong], device=device)]
        neg_id = wrong[int(torch.argmax(wrong_logits).item())][1]

        neg_local_idx = [idx for idx, (l, _) in enumerate(wrong) if _ == neg_id][0]
        neg_letter_idx = "ABCD".index(wrong[neg_local_idx][0])
        safe_options = list(options)
        safe_options[correct_idx], safe_options[neg_letter_idx] = safe_options[neg_letter_idx], safe_options[correct_idx]
        safe_prompt = _format_standard_mcq(q, safe_options)
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


def _build_medmcqa(model, tokenizer, n_total, device, choice_ids):
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    pairs = []
    for row in ds:
        try:
            cop = int(row.get("cop", -1))
        except (TypeError, ValueError): continue
        if cop not in idx_to_letter: continue
        
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
            
        pos_id = choice_ids[idx_to_letter[cop]]
        wrong_ids = [choice_ids[l] for l in "ABCD" if choice_ids[l] != pos_id]
        wrong_logits = logits[torch.tensor(wrong_ids, device=device)]
        neg_id = wrong_ids[int(torch.argmax(wrong_logits).item())]
        
        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "safe_input_ids": enc["input_ids"].clone(),
            "safe_attention_mask": enc["attention_mask"].clone(),
            "pos_id": pos_id,
            "neg_id": neg_id,
        })
        if len(pairs) >= n_total: break
    return pairs


def _build_pubmedqa(model, tokenizer, n_total, device, choice_ids, config):
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    pairs = []
    for row in ds:
        gold = str(row.get("final_decision", "")).strip().lower()
        if gold not in {"yes", "no"}: continue
        question = str(row.get("question", "")).strip()
        if not question: continue
        
        ctx_obj = row.get("context", {})
        contexts = ctx_obj.get("contexts", []) if isinstance(ctx_obj, dict) else []
        prompt = _format_context_mcq(question, contexts, config)
        enc = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits[0, -1, :].float()
            
        pos_letter = "A" if gold == "yes" else "B"
        pos_id = choice_ids[pos_letter]
        wrong_ids = [choice_ids[l] for l in "ABCD" if choice_ids[l] != pos_id]
        wrong_logits = logits[torch.tensor(wrong_ids, device=device)]
        neg_id = wrong_ids[int(torch.argmax(wrong_logits).item())]
        
        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "safe_input_ids": enc["input_ids"].clone(),
            "safe_attention_mask": enc["attention_mask"].clone(),
            "pos_id": pos_id,
            "neg_id": neg_id,
        })
        if len(pairs) >= n_total: break
    return pairs


# ---------------------------------------------------------------------------
# Intervention Engine
# ---------------------------------------------------------------------------

def _get_mlp_down_proj(layer):
    """Abstraction for layer-wise MLP output projection access."""
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    elif hasattr(layer, "mlp") and hasattr(layer.mlp, "c_proj"): # GPT-style
        return layer.mlp.c_proj
    elif hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "w2"): # Llama-style variants
        return layer.feed_forward.w2
    raise AttributeError("Could not identify MLP output projection for this architecture.")


def patch_run(
    model: torch.nn.Module,
    layer_stack: torch.nn.ModuleList,
    anchors_by_layer: Dict[int, List[int]],
    pairs: List[Dict],
    n_samples: int,
    choice_token_ids: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Implements the Zero-then-Restore sufficiency test."""
    degrade_shifts = []
    restore_shifts = []
    stats = {"correct_clean": 0, "correct_corrupt": 0, "correct_restored": 0}

    def _make_save_hooks(stored, targets):
        handles = []
        for l_idx, cols in targets.items():
            proj = _get_mlp_down_proj(layer_stack[l_idx])
            def _hook(m, inp, _l=l_idx, _c=cols):
                h = inp[0]
                for c in _c:
                    stored[(_l, c)] = h[0, -1, c].detach().clone()
            handles.append(proj.register_forward_pre_hook(_hook))
        return handles

    def _make_zero_hooks(targets):
        handles = []
        for l_idx, cols in targets.items():
            proj = _get_mlp_down_proj(layer_stack[l_idx])
            def _hook(m, inp, _l=l_idx, _c=cols):
                h = inp[0]
                out = h.clone()
                out[0, -1, _c] = 0.0
                return (out,)
            handles.append(proj.register_forward_pre_hook(_hook))
        return handles

    def _make_restore_hooks(stored, targets):
        handles = []
        for l_idx, cols in targets.items():
            proj = _get_mlp_down_proj(layer_stack[l_idx])
            def _hook(m, inp, _l=l_idx, _c=cols):
                h = inp[0]
                out = h.clone()
                for c in _c:
                    if (_l, c) in stored:
                        out[0, -1, c] = stored[(_l, c)].to(out.dtype)
                return (out,)
            handles.append(proj.register_forward_pre_hook(_hook))
        return handles

    def _eval(logits, pair):
        margin = (logits[0, pair["pos_id"]] - logits[0, pair["neg_id"]]).item()
        pred = choice_token_ids[torch.argmax(logits[0, choice_token_ids]).item()].item()
        return margin, int(pred == pair["pos_id"])

    with torch.no_grad():
        for i in range(min(n_samples, len(pairs))):
            p = pairs[i]
            ids, mask = p["input_ids"].to(device), p["attention_mask"].to(device)
            stored = {}
            h1 = _make_save_hooks(stored, anchors_by_layer)
            l_clean = model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
            for h in h1: h.remove()
            m_clean, c_clean = _eval(l_clean, p)
            stats["correct_clean"] += c_clean

            h2 = _make_zero_hooks(anchors_by_layer)
            l_corrupt = model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
            for h in h2: h.remove()
            m_corrupt, c_corrupt = _eval(l_corrupt, p)
            stats["correct_corrupt"] += c_corrupt

            h3 = _make_restore_hooks(stored, anchors_by_layer)
            l_restored = model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
            for h in h3: h.remove()
            m_restored, c_restored = _eval(l_restored, p)
            stats["correct_restored"] += c_restored

            degrade_shifts.append(m_corrupt - m_clean)
            restore_shifts.append(m_restored - m_corrupt)

    n = len(degrade_shifts)
    return {
        "mean_degrade_shift": float(np.mean(degrade_shifts)),
        "std_degrade_shift": float(np.std(degrade_shifts)),
        "mean_restore_shift": float(np.mean(restore_shifts)),
        "std_restore_shift": float(np.std(restore_shifts)),
        "acc_clean": stats["correct_clean"] / n,
        "acc_corrupt": stats["correct_corrupt"] / n,
        "acc_restored": stats["correct_restored"] / n,
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# High-level Experiment Orchestration
# ---------------------------------------------------------------------------

def run_causal_experiment(
    model_path: str,
    model_tag: str,
    anchors: List[Tuple[int, int]],
    config: ExperimentConfig,
    trust_remote_code: bool = False,
):
    """Executes full suite of activation patching tests for a given model."""
    logger.info(f"Initializing causal experiment for {model_tag}...")
    os.makedirs(config.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    device = next(model.parameters()).device
    
    backbone = getattr(model, "model", model)
    if hasattr(backbone, "language_model"): backbone = backbone.language_model
    layer_stack = backbone.layers
    
    intermediate_size = _get_mlp_down_proj(layer_stack[0]).weight.shape[1]
    choice_ids_map = _get_choice_token_ids(tokenizer)
    choice_token_ids = torch.tensor([choice_ids_map[l] for l in "ABCD"], dtype=torch.long, device=device)

    anchors_by_layer = {}
    for (l, c) in anchors:
        anchors_by_layer.setdefault(l, []).append(c)

    datasets = {}
    for ds_name, n_req in [("medqa", config.n_medqa), ("medmcqa", config.n_medmcqa), ("pubmedqa", config.n_pubmedqa)]:
        logger.info(f"Building {ds_name} contrastive pairs...")
        datasets[ds_name] = build_dataset_pairs(ds_name, model, tokenizer, n_req, device, config)

    all_results = {
        "metadata": {
            "model_path": model_path,
            "model_tag": model_tag,
            "n_anchors": len(anchors),
            "config": asdict(config),
            "timestamp": datetime.now().isoformat(),
        },
        "seeds": {},
    }

    for seed in config.seeds:
        logger.info(f"Running Seed {seed}...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        seed_results = {}
        for ds_name, pairs in datasets.items():
            shuffled = list(pairs)
            random.shuffle(shuffled)
            anchor_res = patch_run(model, layer_stack, anchors_by_layer, shuffled, config.n_patch_samples, choice_token_ids, device)
            seed_results[ds_name] = {"anchor": anchor_res}
        all_results["seeds"][str(seed)] = seed_results

    out_path = os.path.join(config.output_dir, f"{model_tag}_activation_patching.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Experiment complete. Results saved to {out_path}")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return all_results
