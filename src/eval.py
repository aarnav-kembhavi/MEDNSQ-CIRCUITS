"""
Mechanistic Evaluation Metrics.
"""

from typing import List, Dict, Any
import torch
import numpy as np
from .data import format_mcq_prompt, get_label_token_ids

def evaluate_model(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: List[Dict[str, Any]]
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    label_ids = get_label_token_ids(tokenizer).to(device)
    correct = 0
    margins = []
    if not dataset: return {"accuracy": 0.0, "mean_margin": 0.0}

    with torch.no_grad():
        for sample in dataset:
            prompt = format_mcq_prompt(sample["question"], sample["options"])
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
            logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[0, -1, :]
            label_probs = torch.softmax(logits[label_ids], dim=-1)
            pred_idx = int(torch.argmax(label_probs).item())
            correct_idx = int(sample["correct_index"])
            if pred_idx == correct_idx: correct += 1
            pos_id = int(label_ids[correct_idx].item())
            mask = torch.ones_like(label_ids, dtype=torch.bool)
            mask[correct_idx] = False
            incorrect_ids = label_ids[mask]
            neg_id = int(incorrect_ids[torch.argmax(logits[incorrect_ids])].item())
            margin = (logits[pos_id] - logits[neg_id]).item()
            margins.append(margin)

    return {
        "accuracy": float(correct / len(dataset)),
        "mean_margin": float(np.mean(margins)),
        "std_margin": float(np.std(margins))
    }
