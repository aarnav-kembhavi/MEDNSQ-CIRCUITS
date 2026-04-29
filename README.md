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

| Path | Role |
|------|------|
| `notebooks/mednsq_probe.py` | `MedNSQProbe`: column crush / restore and scoring |
| `notebooks/mednsq_data.py` | MedQA loading, prompts, adversarial pairs |
| `notebooks/mednsq_eval.py` | 4-way accuracy and margin metrics |
| `notebooks/discover_ablate.py` | Full EMS pipeline: anchor discovery + ablation (default: AFM OpenMed model) |
| `notebooks/cross_dataset.py` | Cross-dataset anchor evaluation (MedQA, MedMCQA, PubMedQA) |

Scripts import each other as **local modules** (`mednsq_*`). Run them with the working directory set to `notebooks` so imports resolve:

```bash
cd notebooks
python discover_ablate.py
python cross_dataset.py
```

Optional: if you copy the `mednsq_*.py` helpers elsewhere, set `MEDNSQ_LIB_DIR` to that folder so `cross_dataset.py` can add it to `sys.path` (see the top of `cross_dataset.py`).

## Outputs

- **Discovery / ablation** (`discover_ablate.py`): JSON files and a log file; names are set in the `Config` dataclass (e.g. `anchors_*.json`, `ablation_*.json`, `experiment_*.log`).
- **Cross-dataset** (`cross_dataset.py`): consolidated results JSON (default `afm_anchor_complete_results.json`) plus optional cached pair files as configured.

## Configuration

Edit the `Config` / `CONFIG` dataclass at the top of each entry script for model IDs, sample sizes, layer ranges, and output paths. For custom architectures (e.g. `trust_remote_code=True`), follow the notes in `discover_ablate.py` regarding layer counts and `middle_layers`.

## Licenses

Respect the licenses of downloaded models (Hugging Face) and datasets (e.g. MedQA, MedMCQA, PubMedQA).
