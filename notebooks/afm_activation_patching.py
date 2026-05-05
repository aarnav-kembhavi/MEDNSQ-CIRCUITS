"""Activation patching for AFM-4.5B-OpenMed-RL-CoT (SFT+DPO+GRPO)."""
from act_patch_common import run_patching_experiment, PatchConfig

MODEL_ID = "openmed-community/AFM-4.5B-OpenMed-RL-CoT"
MODEL_KEY = "afm_openmed"

# Top 38 anchors (same as used in cross-dataset eval)
ANCHORS = [
    (16, 12646), (19, 8084), (12, 4348), (13, 7617), (12, 1751),
    (13, 8577), (12, 4302), (18, 4213), (21, 4765), (16, 107),
    (12, 12143), (16, 2216), (17, 15709), (21, 9130), (22, 12414),
    (16, 6425), (19, 12145), (14, 17780), (21, 838), (23, 6331),
    (17, 12143), (19, 13024), (19, 6701), (20, 2825), (16, 17666),
    (17, 3481), (20, 15860), (13, 4321), (17, 9773), (18, 532),
    (20, 1056), (13, 12183), (16, 8091), (22, 7310), (18, 13172),
    (20, 3175), (19, 17809), (18, 13462),
]


if __name__ == "__main__":
    cfg = PatchConfig(
        n_medqa=300,
        n_medmcqa=300,
        n_pubmedqa=300,
        n_patch_samples=300,
        n_random_trials=10,
        seeds=[42, 123, 7, 13, 97],
        output_dir="patch_results",
    )
    run_patching_experiment(MODEL_ID, MODEL_KEY, ANCHORS, cfg, trust_remote_code=True)