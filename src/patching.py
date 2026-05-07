"""
Activation Patching and Causal Validation Utilities.
"""

import gc
import json
import os
import random
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from datasets import load_dataset
from .core import get_backbone, get_mlp_down_proj

logger = logging.getLogger(__name__)

@dataclass
class PatchConfig:
    n_medqa: int = 300
    n_medmcqa: int = 300
    n_pubmedqa: int = 300
    n_patch_samples: int = 120
    n_random_trials: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123])
    output_dir: str = "results"

def patch_run(
    model: torch.nn.Module,
    layer_stack: torch.nn.ModuleList,
    anchors_by_layer: Dict[int, List[int]],
    pairs: List[Dict],
    n_samples: int,
    choice_token_ids: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Zero-then-restore sufficiency test."""
    degrade_shifts = []
    restore_shifts = []
    stats = {"correct_clean": 0, "correct_corrupt": 0, "correct_restored": 0}

    def _make_save_hooks(stored, targets):
        handles = []
        for l_idx, cols in targets.items():
            proj = get_mlp_down_proj(layer_stack[l_idx])
            def _hook(m, inp, _l=l_idx, _c=cols):
                h = inp[0]
                for c in _c: stored[(_l, c)] = h[0, -1, c].detach().clone()
            handles.append(proj.register_forward_pre_hook(_hook))
        return handles

    def _make_zero_hooks(targets):
        handles = []
        for l_idx, cols in targets.items():
            proj = get_mlp_down_proj(layer_stack[l_idx])
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
            proj = get_mlp_down_proj(layer_stack[l_idx])
            def _hook(m, inp, _l=l_idx, _c=cols):
                h = inp[0]
                out = h.clone()
                for c in _c:
                    if (_l, c) in stored: out[0, -1, c] = stored[(_l, c)].to(out.dtype)
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
        "mean_restore_shift": float(np.mean(restore_shifts)),
        "acc_clean": stats["correct_clean"] / n,
        "acc_corrupt": stats["correct_corrupt"] / n,
        "acc_restored": stats["correct_restored"] / n,
        "n_samples": n,
    }
