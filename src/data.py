"""
Dataset and Contrastive Pair Utilities.
"""

from typing import List, Dict, Any
import torch
from datasets import load_dataset

def load_medqa_dataset(n_total: int = 1000, split: str = "train") -> List[Dict[str, Any]]:
    ds = load_dataset("openlifescienceai/medqa")[split]
    n_total = min(n_total, len(ds))
    samples = []
    for i in range(n_total):
        row = ds[i]
        q = row["data"]["Question"]
        opts = row["data"]["Options"]
        options = [opts["A"], opts["B"], opts["C"], opts["D"]]
        correct_idx = "ABCD".index(row["data"]["Correct Option"])
        samples.append({"question": q, "options": options, "correct_index": correct_idx})
    return samples

def format_mcq_prompt(question: str, options: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    option_block = "\n".join([f"{l}. {t}" for l, t in zip(letters, options)])
    return f"Question: {question}\nOptions:\n{option_block}\nAnswer:"

def get_label_token_ids(tokenizer, labels: str = "ABCD") -> torch.Tensor:
    ids = []
    for l in labels:
        enc = tokenizer(l, add_special_tokens=False)["input_ids"]
        ids.append(enc[0] if not isinstance(enc[0], list) else enc[0][0])
    return torch.tensor(ids, dtype=torch.long)

def build_adversarial_pairs(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: List[Dict[str, Any]],
    n_samples: int = 100
) -> List[Dict[str, Any]]:
    model.eval()
    device = next(model.parameters()).device
    label_ids = get_label_token_ids(tokenizer).to(device)
    n_samples = min(n_samples, len(dataset))
    pairs = []
    with torch.no_grad():
        for i in range(n_samples):
            sample = dataset[i]
            prompt = format_mcq_prompt(sample["question"], sample["options"])
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
            logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[0, -1, :]
            correct_idx = sample["correct_index"]
            pos_id = int(label_ids[correct_idx].item())
            mask = torch.ones_like(label_ids, dtype=torch.bool)
            mask[correct_idx] = False
            incorrect_logits = logits[label_ids].masked_fill(~mask, -float("inf"))
            neg_local_idx = int(torch.argmax(incorrect_logits).item())
            neg_id = int(label_ids[neg_local_idx].item())
            
            # Safe prompt
            safe_options = list(sample["options"])
            safe_options[correct_idx], safe_options[neg_local_idx] = safe_options[neg_local_idx], safe_options[correct_idx]
            safe_prompt = format_mcq_prompt(sample["question"], safe_options)
            safe_enc = tokenizer(safe_prompt, return_tensors="pt", add_special_tokens=True).to(device)

            pairs.append({
                "input_ids": enc.input_ids.cpu(),
                "attention_mask": enc.attention_mask.cpu(),
                "safe_input_ids": safe_enc.input_ids.cpu(),
                "safe_attention_mask": safe_enc.attention_mask.cpu(),
                "pos_id": pos_id,
                "neg_id": neg_id,
                "correct_label_index": correct_idx,
                "distractor_label_index": neg_local_idx
            })
    return pairs
