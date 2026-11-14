"""Cluster-aware uncertainty estimates and paired tests for trajectory windows."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _group_values(values, cluster_ids):
    values = np.asarray(values, dtype=float).reshape(-1)
    cluster_ids = np.asarray(cluster_ids)
    if len(values) != len(cluster_ids):
        raise ValueError("values and cluster_ids must have the same length")
    if not len(values):
        raise ValueError("at least one observation is required")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    grouped = {}
    for cluster in dict.fromkeys(cluster_ids.tolist()):
        mask = cluster_ids == cluster
        grouped[cluster] = values[mask]
    return grouped


def cluster_bootstrap_ci(values, cluster_ids, n_boot=2000, confidence=0.95, seed=42):
    """Percentile CI by resampling whole trajectory clusters with replacement.

    Each draw samples K clusters, where K is the observed number of clusters,
    then includes all windows belonging to each sampled cluster. The point
    estimate is the original window-weighted mean, matching ordinary ADE/FDE.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    grouped = _group_values(values, cluster_ids)
    clusters = list(grouped)
    point = float(np.mean(np.asarray(values, dtype=float)))
    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot, dtype=float)

    for index in range(n_boot):
        selected = rng.choice(len(clusters), size=len(clusters), replace=True)
        chunks = [grouped[clusters[selected_index]] for selected_index in selected]
        boot[index] = np.concatenate(chunks).mean()

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(boot, [alpha, 1.0 - alpha])
    return {
        "estimate": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_observations": int(len(values)),
        "n_clusters": int(len(clusters)),
        "n_boot": int(n_boot),
        "confidence": float(confidence),
        "bootstrap_unit": "trajectory_id",
        "cluster_weighting": "resample whole clusters; pool all windows in sampled clusters",
    }


def per_cluster_means(values, cluster_ids):
    grouped = _group_values(values, cluster_ids)
    clusters = list(grouped)
    means = np.asarray([grouped[key].mean() for key in clusters], dtype=float)
    return clusters, means


def paired_cluster_test(base_values, model_values, cluster_ids):
    """Paired tests on per-cluster means, not on correlated overlapping windows."""
    base_clusters, base_means = per_cluster_means(base_values, cluster_ids)
    model_clusters, model_means = per_cluster_means(model_values, cluster_ids)
    if base_clusters != model_clusters:
        raise ValueError("paired models must contain identical ordered cluster IDs")
    differences = base_means - model_means
    n_clusters = len(differences)
    if n_clusters < 2:
        raise ValueError("at least two independent trajectory clusters are required")

    if np.allclose(differences, 0.0):
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    else:
        result = stats.wilcoxon(base_means, model_means, zero_method="pratt", method="auto")
        wilcoxon_stat, wilcoxon_p = float(result.statistic), float(result.pvalue)

    t_result = stats.ttest_rel(base_means, model_means)
    std_diff = float(np.std(differences, ddof=1))
    effect = float(np.mean(differences) / std_diff) if std_diff > 0 else 0.0
    return {
        "n_clusters": n_clusters,
        "mean_difference_base_minus_model": float(np.mean(differences)),
        "median_difference_base_minus_model": float(np.median(differences)),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p,
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_pvalue": float(t_result.pvalue),
        "cohens_d_paired": effect,
        "test_unit": "per-trajectory mean",
    }


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values in original input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a finite 1-D sequence in [0, 1]")
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    adjusted_sorted = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (n - rank) * p[original_index])
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty(n, dtype=float)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted.tolist()
