"""
Comprehensive Cross-Dataset Anchor Evaluation.
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
from analysis import compare_anchor_vs_random, perform_cluster_analysis

def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Mechanistic Evaluation")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--anchor_file", type=str, required=True, help="JSON file containing discovered anchors")
    parser.add_argument("--output_file", type=str, default="results/cross_dataset_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"Loading model {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    probe = EMSProbe(model)

    print("Loading anchors...")
    with open(args.anchor_file, "r") as f:
        anchor_data = json.load(f)
    anchors = anchor_data["anchors"]

    print("Loading datasets...")
    dataset = load_medqa_dataset(n_total=200)
    pairs = build_adversarial_pairs(model, tokenizer, dataset, n_samples=100)

    print("Evaluating anchors...")
    results = []
    for a in anchors:
        l, c = a["layer"], a["column"]
        orig = probe.simulate_column_crush(l, c)
        try:
            m_post = probe.compute_margins(pairs)
            m_pre = probe.compute_margins(pairs)
            drop = (m_pre - m_post).mean().item()
            results.append({"layer": l, "column": c, "drop": drop})
        finally:
            probe.restore_column(l, c, orig)

    print("Saving results...")
    with open(args.output_file, "w") as f:
        json.dump({"metadata": {"model_id": args.model_id}, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
