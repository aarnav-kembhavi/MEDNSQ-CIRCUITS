# Mechanistic Sabotage Analysis Toolkit

Official implementation of **Mechanistic Sabotage Analysis** and **Empirical Margin Sensitivity (EMS)** as described in the NeurIPS 2026 paper: 
*"From Behavioral Memorization to Mechanistic Sabotage: Neuron-Level Causal Analysis of SFT and RL in Medical LLMs"*.

## Overview

This repository provides a high-precision toolkit for identifying and validating "Causal Anchor" neurons in medical Large Language Models. Our pipeline enables researchers to:
1. **Discover**: Identify neurons with high causal influence on medical decision margins using the EMS operator.
2. **Validate**: Quantify the causal impact through zero-then-restore activation patching.
3. **Analyze**: Cluster neurons based on their cross-dataset effect profiles to reveal localized medical circuits.

## Repository Structure

```text
├── scripts/                # Entry point scripts for discovery and evaluation
│   ├── run_ems_pipeline.py # End-to-end anchor discovery and ablation
│   └── run_cross_dataset_eval.py # Multi-dataset analysis and clustering
├── src/                    # Core library
│   ├── core.py             # EMSProbe and architecture-agnostic intervention logic
│   ├── data.py             # Contrastive pair building for MedQA, PubMedQA, etc.
│   ├── eval.py             # Mechanistic performance metrics
│   ├── patching.py         # Activation patching and sufficiency testing
│   └── analysis.py         # Statistical tests and cluster analysis
└── results/                # Default directory for experiment outputs
```

## Installation

```bash
git clone https://github.com/aarnav-kembhavi/MEDNSQ-CIRCUITS.git
cd MEDNSQ-CIRCUITS
pip install -r requirements.txt
```

## Quick Start

### 1. Anchor Discovery
Identify the most causally significant neurons for a specific model:

```bash
python scripts/run_ems_pipeline.py --model_id google/medgemma-4b-it --n_calib 1000 --output_dir results/model_a
```

### 2. Cross-Dataset Evaluation
Analyze how discovered anchors perform across different medical benchmarks:

```bash
python scripts/run_cross_dataset_eval.py --model_id google/medgemma-4b-it --anchor_file results/model_a/discovered_anchors.json
```

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{kembhavi2026mechanistic,
  title={From Behavioral Memorization to Mechanistic Sabotage: Neuron-Level Causal Analysis of SFT and RL in Medical LLMs},
  author={Kembhavi, Aarnav and [Other Authors]},
  journal={NeurIPS},
  year={2026}
}
```
