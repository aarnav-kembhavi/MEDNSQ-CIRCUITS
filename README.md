# MEDQ

Experiments for **neural anchor discovery and ablation** on medical multiple-choice benchmarks, using Hugging Face models, MedQA-style prompts, and a column-crush intervention on MLP `down_proj` weights (`MedNSQProbe`).

## Requirements

- **Python** 3.10 or newer
- **GPU** strongly recommended (scripts load large causal LMs with `device_map="auto"` and `torch.bfloat16`)
- **Hugging Face** account / token as needed for gated models (e.g. Llama) and for dataset downloads

## Setup

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, if you need a specific CUDA build of PyTorch, install it from [pytorch.org](https://pytorch.org/get-started/locally/) before or instead of the generic `torch` line from pip.

## Project layout

| Path                           | Role                                                                        |
| ------------------------------ | --------------------------------------------------------------------------- |
| `notebooks/mednsq_probe.py`    | `MedNSQProbe`: column crush / restore and scoring                           |
| `notebooks/mednsq_data.py`     | MedQA loading, prompts, adversarial pairs                                   |
| `notebooks/mednsq_eval.py`     | 4-way accuracy and margin metrics                                           |
| `notebooks/discover_ablate.py` | Full EMS pipeline: anchor discovery + ablation (default: AFM OpenMed model) |
| `notebooks/cross_dataset.py`   | Cross-dataset anchor evaluation (MedQA, MedMCQA, PubMedQA)                  |

Scripts import each other as **local modules** (`mednsq_*`). Run them with the working directory set to `notebooks` so imports resolve:

```bash
cd notebooks
python discover_ablate.py
python cross_dataset.py
```

## Licenses

Respect the licenses of downloaded models (Hugging Face) and datasets (e.g. MedQA, MedMCQA, PubMedQA).
