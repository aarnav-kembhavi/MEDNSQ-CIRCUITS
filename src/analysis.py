"""
Statistical Analysis and Clustering for Mechanistic Interpretability.
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def compare_anchor_vs_random(
    anchor_drops: List[float],
    random_drops: List[float],
    dataset_name: str
) -> Dict[str, Any]:
    """
    Performs statistical comparison (Welch's t-test and Cohen's d) between anchor and random sets.
    """
    anchor_mean = np.mean(anchor_drops)
    anchor_std = np.std(anchor_drops, ddof=1)
    random_mean = np.mean(random_drops)
    random_std = np.std(random_drops, ddof=1)
    
    # Welch's t-test (unequal variance)
    t_stat, p_val = stats.ttest_ind(anchor_drops, random_drops, equal_var=False, alternative='greater')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((anchor_std**2 + random_std**2) / 2)
    cohens_d = (anchor_mean - random_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        "dataset": dataset_name,
        "anchor_mean": float(anchor_mean),
        "random_mean": float(random_mean),
        "difference": float(anchor_mean - random_mean),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d)
    }

def perform_cluster_analysis(
    effect_matrix: torch.Tensor,
    n_clusters: int = 4,
    seed: int = 42
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Clusters neurons based on their cross-dataset effect profiles.
    """
    X = effect_matrix.numpy()
    
    # Normalization (focus on pattern)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    X_norm = (X - mu) / sd
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    assignments = kmeans.fit_predict(X_norm)
    
    score = silhouette_score(X_norm, assignments) if X.shape[0] > n_clusters else 0.0
    
    metrics = {
        "silhouette_score": float(score),
        "inertia": float(kmeans.inertia_),
        "centers": kmeans.cluster_centers_.tolist()
    }
    
    return torch.tensor(assignments), metrics
