#!/usr/bin/env python3
"""Evaluate the newly trained AlertBERT model and compare with pre-trained baseline.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n alertbert python3 eval_trained_model.py

This script:
1. Runs compute_roc_trajectories for the newly trained model
2. Compares results with the pre-trained baseline
3. Prints summary metrics tables
"""
import sys
import os
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alertbert.eval_grouping import (
    compute_roc_trajectories,
    alertbert_deltas,
    alertbert_theta_roc_traj_primary,
    alertbert_theta_roc_traj_secondary,
    alertbert_theta_roc_traj_tertiary,
    alertbert_theta_roc_traj_quartary,
    load_results,
    get_eval_file_name,
    get_grouping_model_params,
    all_delta_theta_vals,
    load_roc_results,
    get_relevant_roc_results,
    compute_auc_score,
    get_low_level_labels,
)
from alertbert.preprocessing import Vocabulary

# ============================================================
# Configuration
# ============================================================
NEW_MODEL_ID = "mlm_1l_4h_16d_original_default_params"
PRETRAINED_MODEL_ID = "mlm_1l_4h_16d_original_1_60k"
AITADS_A_CONFIG = "original"
PATH = "saved_models"

# ============================================================
# Step 1: Run ROC trajectory computation
# ============================================================
def run_roc_evaluation():
    """Compute ROC trajectories for the newly trained model."""
    print("=" * 70)
    print(f"Evaluating model: {NEW_MODEL_ID}")
    print(f"Data config: {AITADS_A_CONFIG}")
    print("=" * 70)

    # Run ROC trajectory computation in phases (like the original script)
    theta_trajectories = [
        ("primary", alertbert_theta_roc_traj_primary),
        ("secondary", alertbert_theta_roc_traj_secondary),
        ("tertiary", alertbert_theta_roc_traj_tertiary),
        ("quartary", alertbert_theta_roc_traj_quartary),
    ]

    for name, thetas in theta_trajectories:
        print(f"\n--- Computing ROC trajectories: {name} ({len(thetas)} thetas x {len(alertbert_deltas)} deltas) ---")
        for delta in alertbert_deltas:
            print(f"  delta={delta}, {len(thetas)} thetas...")
            compute_roc_trajectories(
                model_id=NEW_MODEL_ID,
                aitads_a_config=AITADS_A_CONFIG,
                deltas=[delta],
                thetas=thetas,
                path=PATH,
                test_mode=False,
            )

    print("\n✓ ROC trajectory computation complete!")

# ============================================================
# Step 2: Load and compare results
# ============================================================
def load_model_summary_metrics(model_id, path="saved_models"):
    """Load all available result files and compute summary metrics for a model."""
    print(f"\nLoading results for {model_id}...")

    results_found = []
    for delta in all_delta_theta_vals:
        for theta in all_delta_theta_vals:
            grouping_params = get_grouping_model_params(
                model_id, delta, theta, data_split="val"
            )
            file_name = get_eval_file_name(grouping_params, AITADS_A_CONFIG, noise=True)
            try:
                result = load_results(path=path, model_id=model_id, name=file_name, split="val")
                results_found.append(result)
            except FileNotFoundError:
                pass
            # Also try clean results
            file_name_clean = get_eval_file_name(grouping_params, AITADS_A_CONFIG, noise=False)
            try:
                result = load_results(path=path, model_id=model_id, name=file_name_clean, split="val")
                results_found.append(result)
            except FileNotFoundError:
                pass

    print(f"  Found {len(results_found)} result files")
    return results_found


def print_metrics_comparison(new_model_id, pretrained_model_id, path="saved_models"):
    """Print a comparison table between the new and pre-trained models."""
    print("\n" + "=" * 80)
    print("COMPARISON: New Model vs Pre-trained Baseline")
    print("=" * 80)

    # Pre-trained baseline (theta=2.0, delta=2.0, val, noise)
    # These are the values we extracted earlier from the pre-trained model
    pretrained_baseline = {
        "val_noise": {
            "accuracy": (0.9998, 0.0003),
            "precision": (0.7181, 0.2217),
            "recall": (0.8445, 0.2376),
            "tnr": (0.9996, 0.0010),
            "f1": (0.7011, 0.2142),
            "mcc": (0.7359, 0.1881),
        },
        "val_clean": {
            "accuracy": (0.9996, 0.0009),
            "precision": (1.0000, 0.0000),
            "recall": (0.8445, 0.2376),
            "tnr": (1.0000, 0.0000),
            "f1": (0.8913, 0.1694),
            "mcc": (0.9056, 0.1452),
        },
    }

    # Load new model results
    try:
        grouping_params = get_grouping_model_params(
            new_model_id, delta=2.0, theta=2.0, data_split="val"
        )
        file_name = get_eval_file_name(grouping_params, AITADS_A_CONFIG, noise=True)

        val_noise = load_results(path=path, model_id=new_model_id, name=file_name, split="val")
        macro_noise = val_noise["summary"]["macro"]["macro"]

        file_name_clean = get_eval_file_name(grouping_params, AITADS_A_CONFIG, noise=False)
        val_clean = load_results(path=path, model_id=new_model_id, name=file_name_clean, split="val")
        macro_clean = val_clean["summary"]["macro"]["macro"]

        print("\n--- Validation (with noise) --- macro metrics ---")
        print(f"{'Metric':<15} {'New Model':>20} {'Pre-trained':>20}")
        for metric in ["accuracy", "precision", "recall", "tnr", "f1", "mcc"]:
            new_val = macro_noise[metric]
            pre_val = pretrained_baseline["val_noise"][metric]
            print(f"{metric:<15} {new_val[0]:>8.4f} ± {new_val[1]:.4f}  {pre_val[0]:>8.4f} ± {pre_val[1]:.4f}")

        print("\n--- Validation (clean) --- macro metrics ---")
        print(f"{'Metric':<15} {'New Model':>20} {'Pre-trained':>20}")
        for metric in ["accuracy", "precision", "recall", "tnr", "f1", "mcc"]:
            new_val = macro_clean[metric]
            pre_val = pretrained_baseline["val_clean"][metric]
            print(f"{metric:<15} {new_val[0]:>8.4f} ± {new_val[1]:.4f}  {pre_val[0]:>8.4f} ± {pre_val[1]:.4f}")

    except FileNotFoundError as e:
        print(f"Could not load new model results: {e}")
        print("The model may need more evaluation to be run first.")

    # Also try to find best result across all delta/theta combinations
    print("\n--- Searching for best F1 across all delta/theta combinations ---")
    try:
        val_results = load_roc_results(
            new_model_id,
            layers=("embedding", "encoder"),
            dim_reduction=2,
            aitds_a_config=AITADS_A_CONFIG,
            noise=True,
            split="val",
            path=path,
        )
        if val_results:
            best_f1 = 0
            best_config = None
            for r in val_results:
                f1_mean = r["summary"]["macro"]["macro"]["f1"][0]
                if f1_mean > best_f1:
                    best_f1 = f1_mean
                    best_config = r
            print(f"Best F1 (noise): {best_f1:.4f}")
            print(f"  Config: delta={best_config['model_params']['delta']}, theta={best_config['model_params']['theta']}")
            for metric in ["accuracy", "precision", "recall", "tnr", "f1", "mcc"]:
                m = best_config["summary"]["macro"]["macro"][metric]
                print(f"  {metric}: {m[0]:.4f} ± {m[1]:.4f}")
    except Exception as e:
        print(f"Could not compute best F1: {e}")


# ============================================================
# Step 3: Compute AUC for ROC curves
# ============================================================
def compute_roc_auc(model_id, path="saved_models"):
    """Compute and print AUC scores for the ROC curve."""
    print("\n" + "=" * 80)
    print(f"ROC AUC Scores for {model_id}")
    print("=" * 80)

    for noise in [True, False]:
        noise_str = "incl" if noise else "excl"
        for split in ["train", "val"]:
            try:
                results = load_roc_results(
                    model_id,
                    layers=("embedding", "encoder"),
                    dim_reduction=2,
                    aitds_a_config=AITADS_A_CONFIG,
                    noise=noise,
                    split=split,
                    path=path,
                )
                if len(results) == 0:
                    continue

                relevant_idx = get_relevant_roc_results(results)
                if len(relevant_idx) == 0:
                    continue

                tpr_all = np.array([r["summary"]["macro"]["macro"]["recall"][0] for r in results])
                tnr_all = np.array([r["summary"]["macro"]["macro"]["tnr"][0] for r in results])

                tpr_relevant = tpr_all[relevant_idx]
                tnr_relevant = tnr_all[relevant_idx]

                sort_idx = np.lexsort((tnr_relevant, -1 * tpr_relevant))
                tnr_relevant = tnr_relevant[sort_idx]
                tpr_relevant = tpr_relevant[sort_idx]

                auc = compute_auc_score(
                    np.array(list(get_relevant_roc_results.__code__.co_consts and [1] or [])),
                    tpr_relevant
                ) if len(tnr_relevant) == 0 else None

                # Proper AUC computation
                auc = compute_auc_score(tnr_relevant, tpr_relevant)

                print(f"\n  {split} | {noise_str} noise | {len(results)} data points | {len(relevant_idx)} relevant | AUC = {auc:.4f}")

            except Exception as e:
                print(f"  {split} | {noise_str} noise: Error - {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AlertBERT trained model")
    parser.add_argument("--eval-only", action="store_true", help="Skip ROC computation, only load and compare results")
    parser.add_argument("--compare-only", action="store_true", help="Only print comparison, skip ROC computation")
    args = parser.parse_args()

    if not args.compare_only:
        print("Step 1: Computing ROC trajectories...")
        run_roc_evaluation()

    print("\nStep 2: Comparing results with pre-trained baseline...")
    print_metrics_comparison(NEW_MODEL_ID, PRETRAINED_MODEL_ID, PATH)