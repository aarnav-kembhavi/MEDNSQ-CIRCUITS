# Mechanistic Sabotage Analysis Toolkit

Official implementation of the Mechanistic Sabotage Analysis and Empirical Margin Sensitivity (EMS) framework.

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
git clone [ANONYMOUS_REPO_URL]
cd [REPO_NAME]
pip install -r requirements.txt
```

## Quick Start

### 1. Anchor Discovery
Identify the most causally significant neurons for a specific model:

```bash
python scripts/run_ems_pipeline.py --model_id [MODEL_ID] --n_calib 1000 --output_dir results/model_a
```

### 2. Cross-Dataset Evaluation
Analyze how discovered anchors perform across different medical benchmarks:

```bash
python scripts/run_cross_dataset_eval.py --model_id [MODEL_ID] --anchor_file results/model_a/discovered_anchors.json
```

## Citation

```text
Under Review (NeurIPS 2026)
```
