"""
Reference entry point for the EMS Discovery and Ablation pipeline.
"""

import argparse
import json
import os
import random
import sys
import torch
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from core import EMSProbe
from data import load_medqa_dataset, build_adversarial_pairs
from eval import evaluate_model

def setup_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="EMS Mechanistic Discovery Pipeline")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for results")
    parser.add_argument("--n_calib", type=int, default=100, help="Samples for discovery")
    parser.add_argument("--n_test", type=int, default=100, help="Samples for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    probe = EMSProbe(model)

    print("Loading datasets...")
    dataset = load_medqa_dataset(n_total=args.n_calib + args.n_test)
    calib_samples = dataset[:args.n_calib]
    test_samples = dataset[args.n_calib:]

    print("Building contrastive pairs...")
    calib_pairs = build_adversarial_pairs(model, tokenizer, calib_samples, n_samples=args.n_calib)

    print("Starting discovery on middle layers...")
    middle_layers = range(len(probe.layers)//4, 3*len(probe.layers)//4)
    all_anchors = []

    for l_idx in middle_layers:
        scores = probe.compute_taylor_scores(l_idx, calib_pairs)
        top_val, top_idx = torch.topk(scores, k=5)
        for val, idx in zip(top_val.tolist(), top_idx.tolist()):
            all_anchors.append({"layer": l_idx, "column": idx, "score": val})

    all_anchors.sort(key=lambda x: x["score"], reverse=True)
    top_anchors = all_anchors[:32]

    output = {
        "metadata": {
            "model_id": args.model_id,
            "timestamp": datetime.now().isoformat(),
            "n_anchors": len(top_anchors)
        },
        "anchors": top_anchors
    }

    out_path = os.path.join(args.output_dir, "discovered_anchors.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Discovery complete. Anchors saved to {out_path}")

if __name__ == "__main__":
    main()
