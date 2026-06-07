"""Evaluation metrics for comparing pure AlertBERT vs. hybrid pipeline.

Computes clustering quality metrics (purity, completeness, V-measure, ARI, NMI)
and operational metrics (LLM call rate, cost per alert, workload reduction).
"""

import logging

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger(__name__)


def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute cluster purity.

    Purity measures the extent to which each cluster contains only one class.
    Higher is better (1.0 = perfect purity).

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels

    Returns:
        Purity score in [0, 1]
    """
    # Build contingency matrix
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Map to contiguous integers
    class_map = {c: i for i, c in enumerate(classes)}
    cluster_map = {c: i for i, c in enumerate(clusters)}

    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    for true_label, pred_label in zip(labels_true, labels_pred):
        contingency[class_map[true_label], cluster_map[pred_label]] += 1

    # Purity = sum of max class in each cluster / total
    return float(np.sum(np.max(contingency, axis=0)) / len(labels_true))


def cluster_completeness(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute cluster completeness.

    Completeness measures whether all members of a given class are assigned
    to the same cluster. Higher is better (1.0 = perfect completeness).

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels

    Returns:
        Completeness score in [0, 1]
    """
    # Symmetric of purity: swap true and pred
    return cluster_purity(labels_pred, labels_true)


def v_measure(labels_true: np.ndarray, labels_pred: np.ndarray, beta: float = 1.0) -> float:
    """Compute V-measure (harmonic mean of homogeneity and completeness).

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
        beta: Weight for completeness vs homogeneity (1.0 = equal weight)

    Returns:
        V-measure score in [0, 1]
    """
    h = cluster_purity(labels_true, labels_pred)  # homogeneity
    c = cluster_completeness(labels_true, labels_pred)  # completeness

    if h == 0 and c == 0:
        return 0.0

    return (1 + beta) * h * c / (beta * h + c) if (beta * h + c) > 0 else 0.0


def evaluate_scenario(
    baseline_labels: np.ndarray,
    refined_labels: np.ndarray,
    true_labels: np.ndarray,
    scenario_name: str = "",
    labels: np.ndarray | None = None,
) -> dict:
    result = {}

    if true_labels is not None and isinstance(true_labels, np.ndarray):
        valid_mask = ~np.isnan(true_labels.astype(float))
        baseline = baseline_labels[valid_mask] if np.any(valid_mask) else baseline_labels
        refined = refined_labels[valid_mask] if np.any(valid_mask) else refined_labels
        truth = true_labels[valid_mask] if np.any(valid_mask) else true_labels
    else:
        truth = true_labels
        baseline = baseline_labels
        refined = refined_labels

    baseline_metrics = _compute_all_metrics(truth, baseline)
    refined_metrics = _compute_all_metrics(truth, refined)

    delta = {}
    for key in baseline_metrics:
        if baseline_metrics[key] is not None and refined_metrics[key] is not None:
            delta[key] = refined_metrics[key] - baseline_metrics[key]
        else:
            delta[key] = None

    result = {
        "baseline": baseline_metrics,
        "refined": refined_metrics,
        "delta": delta,
        "scenario_name": scenario_name,
    }

    return result


def _compute_all_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """Compute all clustering metrics for a single set of labels."""
    try:
        purity = cluster_purity(true_labels, pred_labels)
    except Exception:
        purity = None

    try:
        completeness = cluster_completeness(true_labels, pred_labels)
    except Exception:
        completeness = None

    try:
        vm = v_measure(true_labels, pred_labels)
    except Exception:
        vm = None

    try:
        ari = adjusted_rand_score(true_labels, pred_labels)
    except Exception:
        ari = None

    try:
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
    except Exception:
        nmi = None

    n_clusters = len(np.unique(pred_labels))

    return {
        "purity": purity,
        "completeness": completeness,
        "v_measure": vm,
        "ari": ari,
        "nmi": nmi,
        "n_clusters": n_clusters,
        "n_samples": len(pred_labels),
    }


def evaluate_hybrid(
    results: dict,
    label_vocabs: dict,
    config,
) -> dict:
    scenario_evals = {}
    all_baseline_metrics = {}
    all_refined_metrics = {}
    all_metrics = {}

    for scenario_name, scenario_data in results.items():
        true_labels = scenario_data.get("true_labels")
        if true_labels is None:
            logger.warning(f"Skipping {scenario_name}: no ground truth labels")
            continue

        baseline_labels = scenario_data["baseline_labels"]
        refined_labels = scenario_data["refined_labels"]
        labels = scenario_data.get("labels")

        min_len = min(len(true_labels), len(baseline_labels), len(refined_labels))
        true_labels = true_labels[:min_len]
        baseline_labels = baseline_labels[:min_len]
        refined_labels = refined_labels[:min_len]
        if labels is not None:
            labels = labels[:min_len]

        eval_result = evaluate_scenario(
            baseline_labels, refined_labels, true_labels, scenario_name,
            labels=labels,
        )
        scenario_evals[scenario_name] = eval_result

        for metric_key in ["purity", "completeness", "v_measure", "ari", "nmi"]:
            if eval_result["baseline"][metric_key] is not None:
                all_baseline_metrics.setdefault(metric_key, []).append(
                    eval_result["baseline"][metric_key]
                )
                all_refined_metrics.setdefault(metric_key, []).append(
                    eval_result["refined"][metric_key]
                )
    summary = {}
    for metric_key in all_baseline_metrics:
        baseline_vals = np.array(all_baseline_metrics[metric_key])
        refined_vals = np.array(all_refined_metrics[metric_key])
        summary[f"baseline_{metric_key}"] = {
            "mean": float(np.mean(baseline_vals)),
            "std": float(np.std(baseline_vals)),
        }
        summary[f"refined_{metric_key}"] = {
            "mean": float(np.mean(refined_vals)),
            "std": float(np.std(refined_vals)),
        }
        summary[f"delta_{metric_key}"] = {
            "mean": float(np.mean(refined_vals - baseline_vals)),
            "std": float(np.std(refined_vals - baseline_vals)),
        }

    total_clusters = sum(
        len(s["cluster_infos"]) for s in results.values()
    )
    total_low_conf = sum(
        s["summary"]["flagged_clusters"] for s in results.values()
    )
    total_refinements = sum(
        len(s.get("refinement_results", [])) for s in results.values()
    )

    n_baseline = sum(
        s.get("n_baseline_clusters", 0) for s in results.values()
    )
    n_clusters = sum(
        s.get("n_clusters", 0) for s in results.values()
    )
    n_refined = sum(
        s.get("n_refined_clusters", 0) for s in results.values()
    )

    ops = {
        "total_clusters": total_clusters,
        "total_flagged": total_low_conf,
        "total_refinements": total_refinements,
        "llm_call_rate": total_refinements / max(total_clusters, 1),
        "flagged_pct": total_low_conf / max(total_clusters, 1) * 100,
    }

    summary["operational"] = ops

    return {
        "summary": summary,
        "scenarios": scenario_evals,
    }


def print_evaluation_report(eval_results: dict) -> str:
    summary = eval_results.get("summary", {})

    lines = []
    lines.append("=" * 80)
    lines.append("HYBRID PIPELINE EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Clustering Quality Comparison:")
    lines.append("-" * 60)
    lines.append(f"{'Metric':<20} {'Baseline':>15} {'Refined':>15} {'Delta':>15}")
    lines.append("-" * 60)

    for metric in ["purity", "completeness", "v_measure", "ari", "nmi"]:
        baseline_key = f"baseline_{metric}"
        refined_key = f"refined_{metric}"
        delta_key = f"delta_{metric}"

        if baseline_key in summary and refined_key in summary:
            b_mean = summary[baseline_key]["mean"]
            r_mean = summary[refined_key]["mean"]
            d_mean = summary[delta_key]["mean"]

            lines.append(
                f"{metric:<20} {b_mean:>14.4f} {r_mean:>14.4f} {d_mean:>+14.4f}"
            )

    lines.append("-" * 60)
    lines.append("")

    if "operational" in summary:
        ops = summary["operational"]
        lines.append("Operational Metrics:")
        lines.append("-" * 60)
        lines.append(f"  Total clusters:         {ops['total_clusters']}")
        lines.append(f"  Flagged:         {ops['total_flagged']} ({ops['flagged_pct']:.1f}%)")
        lines.append(f"  LLM refinements:        {ops['total_refinements']}")
        lines.append(f"  LLM call rate:          {ops['llm_call_rate']:.3f}")

        lines.append("")

    lines.append("Per-Scenario Results:")
    lines.append("-" * 60)
    for scenario_name, scenario_eval in eval_results.get("scenarios", {}).items():
        lines.append(f"\n  Scenario: {scenario_name}")
        lines.append(f"    Baseline: purity={scenario_eval['baseline'].get('purity', 'N/A'):.4f}, "
                     f"completeness={scenario_eval['baseline'].get('completeness', 'N/A'):.4f}, "
                     f"ARI={scenario_eval['baseline'].get('ari', 'N/A'):.4f}")
        lines.append(f"    Refined:  purity={scenario_eval['refined'].get('purity', 'N/A'):.4f}, "
                     f"completeness={scenario_eval['refined'].get('completeness', 'N/A'):.4f}, "
                     f"ARI={scenario_eval['refined'].get('ari', 'N/A'):.4f}")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)