#!/usr/bin/env python3
"""Precision evaluation: Zero-shot vs Few-shot on ALL clusters of a test scenario.

Analyzes every cluster in the scenario (not just attack-phase clusters),
then computes precision, recall, and F1 for both modes.

Usage:
    python evaluate_precision.py \
        --scenario 7 \
        --paradigm experiments/attack_paradigm/attack_paradigm.json \
        --output-dir experiments/precision_s7

The script runs in two phases:
  Phase 1: Zero-shot analysis (all clusters)
  Phase 2: Few-shot analysis (all clusters, with paradigm)

If a phase result already exists in output-dir, it is skipped (resumable).
"""

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_pipeline.config import LLMConfig, PipelineConfig
from hybrid_pipeline.cluster_sampler import ClusterSampler
from hybrid_pipeline.llm_attack_analyzer import Tier1Analyzer
from hybrid_pipeline.pipeline import load_model_and_data


def load_paradigm(paradigm_path: Path) -> dict:
    with open(paradigm_path, encoding="utf-8") as f:
        return json.load(f)


def build_paradigm_prompt(paradigm: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("已知攻击范式（从历史攻击场景学习）")
    lines.append("=" * 60)
    lines.append(f"\n攻击链名称：{paradigm.get('attack_chain_name', '未知')}")
    lines.append(f"链概述：{paradigm.get('chain_summary', '')}\n")
    for i, stage in enumerate(paradigm.get("stages", []), 1):
        lines.append(f"\n--- 阶段 {i}: {stage.get('stage_name', '未知')} ---")
        lines.append(f"MITRE战术：{stage.get('mitre_tactic', 'N/A')}")
        lines.append(f"典型告警类型：{', '.join(stage.get('typical_alert_types', []))}")
        lines.append(f"关键指标：{'; '.join(stage.get('key_indicators', []))}")
        if stage.get("benign_looking_but_actually_attack"):
            lines.append(f"【重点】看起来良性但实际是攻击：")
            for item in stage["benign_looking_but_actually_attack"]:
                lines.append(f"  - {item}")
    lines.append("\n" + "=" * 60)
    lines.append("使用上述范式辅助判断当前告警簇")
    lines.append("=" * 60)
    return "\n".join(lines)


def run_tier1_all_clusters(
    cluster_infos,
    alert_data,
    labels,
    embeddings,
    paradigm_text: str | None = None,
    max_cost: float = float("inf"),
):
    """Run Tier 1 analysis on all clusters."""
    llm_config = LLMConfig(
        model="deepseek-v4-flash",
        max_cost=max_cost,
        max_alerts_per_prompt=5,
    )
    analyzer = Tier1Analyzer(llm_config, paradigm_text=paradigm_text)
    summaries = analyzer.analyze_all_clusters(
        cluster_infos=cluster_infos,
        alert_data=alert_data,
        labels=labels,
        embeddings=embeddings,
        max_clusters=None,
    )
    return summaries, analyzer.get_stats()


def compute_precision_recall(summaries, labels, true_labels):
    """Compute precision, recall, and F1.

    Definitions:
    - Actual positive: cluster whose dominant true label != "-"
    - Predicted positive: cluster with risk_level in {medium, high, critical}
    """
    unique_labels = np.unique(labels)
    benign_token = None

    # Find benign token (most frequent label)
    flat_true = true_labels[true_labels != ""]
    if len(flat_true) > 0:
        counter = Counter(flat_true)
        benign_token = counter.most_common(1)[0][0]
    else:
        benign_token = "-"

    # Map cluster_id -> summary
    summary_map = {s.cluster_id: s for s in summaries}

    tp = 0  # true positive: predicted attack, actually attack
    fp = 0  # false positive: predicted attack, actually benign
    fn = 0  # false negative: predicted benign, actually attack
    tn = 0  # true negative: predicted benign, actually benign

    total_actual_attack_clusters = 0

    for cid in unique_labels:
        # Determine actual label
        mask = labels == cid
        cluster_true = true_labels[mask]
        if len(cluster_true) == 0:
            continue
        true_counter = Counter(cluster_true)
        dominant_true = true_counter.most_common(1)[0][0]
        is_actual_attack = dominant_true != benign_token

        if is_actual_attack:
            total_actual_attack_clusters += 1

        # Determine predicted label
        summary = summary_map.get(cid)
        if summary is None:
            predicted_attack = False
        else:
            predicted_attack = summary.risk_level in ("medium", "high", "critical")

        if predicted_attack and is_actual_attack:
            tp += 1
        elif predicted_attack and not is_actual_attack:
            fp += 1
        elif not predicted_attack and is_actual_attack:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "total_clusters": len(unique_labels),
        "total_actual_attack_clusters": total_actual_attack_clusters,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Precision evaluation: ZS vs FS")
    parser.add_argument("--scenario", type=int, required=True, help="Scenario index to evaluate")
    parser.add_argument("--paradigm", type=str, required=True, help="Path to attack_paradigm.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max-cost", type=float, default=float("inf"), help="Max total cost")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load paradigm
    paradigm = load_paradigm(Path(args.paradigm))
    paradigm_text = build_paradigm_prompt(paradigm)
    print(f"Loaded paradigm: {paradigm.get('attack_chain_name', 'N/A')}")
    print(f"Paradigm stages: {len(paradigm.get('stages', []))}")

    # Load model and data (once)
    print("\n[Loading model and data...]")
    config = PipelineConfig()
    grouping_model, data, label_vocabs, _, _ = load_model_and_data(config)
    scenario = data.scenarios[args.scenario]

    # Run AlertBERT clustering
    print(f"\n[Running AlertBERT on scenario {args.scenario}...]")
    result = grouping_model(scenario, return_details=True)
    labels = result["labels"]
    embeddings = result["embeddings"]
    pre_cluster_ids = result["pre_cluster_ids"]
    n_clusters = len(np.unique(labels))
    print(f"  Produced {n_clusters} clusters")

    # Ground truth labels
    target = "hierarchical_event_label"
    if target in scenario.data:
        true_labels = label_vocabs[target]([scenario.data[target]]).numpy().squeeze()
    else:
        print("ERROR: No ground truth labels found")
        sys.exit(1)

    # Alert data
    alert_data = {}
    for key in ["short", "host", "name", "ip", "raw_time"]:
        if key in scenario.data:
            alert_data[key] = scenario.data[key]

    # Score clusters
    cluster_sampler = ClusterSampler()
    cluster_infos = cluster_sampler.score_clusters(
        labels=labels,
        embeddings=embeddings,
        pre_cluster_ids=pre_cluster_ids,
        alert_types=alert_data.get("short"),
        hosts=alert_data.get("host"),
        raw_time=alert_data.get("raw_time"),
    )
    print(f"  Scored {len(cluster_infos)} clusters")

    # Phase 1: Zero-shot
    zs_path = output_dir / "zs_summaries.json"
    if zs_path.exists():
        print(f"\n[Phase 1] Found existing {zs_path}, loading...")
        with open(zs_path, encoding="utf-8") as f:
            raw = json.load(f)
            zs_summaries = [Tier1Analyzer.__new__(Tier1Analyzer)]  # dummy
            # Reconstruct from dict
            from hybrid_pipeline.llm_attack_analyzer import Tier1ClusterSummary
            zs_summaries = [Tier1ClusterSummary(**item) for item in raw]
        zs_stats = {"total_calls": len(zs_summaries), "total_cost": 0.0}
    else:
        print(f"\n[Phase 1] Running Zero-shot on all {n_clusters} clusters...")
        t0 = time.time()
        zs_summaries, zs_stats = run_tier1_all_clusters(
            cluster_infos, alert_data, labels, embeddings,
            paradigm_text=None,
            max_cost=args.max_cost,
        )
        t1 = time.time()
        print(f"  Done: {zs_stats['total_calls']} calls, ${zs_stats['total_cost']:.4f}, {t1-t0:.1f}s")
        with open(zs_path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in zs_summaries], f, ensure_ascii=False, indent=2)

    # Phase 2: Few-shot
    fs_path = output_dir / "fs_summaries.json"
    if fs_path.exists():
        print(f"\n[Phase 2] Found existing {fs_path}, loading...")
        with open(fs_path, encoding="utf-8") as f:
            raw = json.load(f)
            from hybrid_pipeline.llm_attack_analyzer import Tier1ClusterSummary
            fs_summaries = [Tier1ClusterSummary(**item) for item in raw]
        fs_stats = {"total_calls": len(fs_summaries), "total_cost": 0.0}
    else:
        print(f"\n[Phase 2] Running Few-shot on all {n_clusters} clusters...")
        t0 = time.time()
        fs_summaries, fs_stats = run_tier1_all_clusters(
            cluster_infos, alert_data, labels, embeddings,
            paradigm_text=paradigm_text,
            max_cost=args.max_cost,
        )
        t1 = time.time()
        print(f"  Done: {fs_stats['total_calls']} calls, ${fs_stats['total_cost']:.4f}, {t1-t0:.1f}s")
        with open(fs_path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in fs_summaries], f, ensure_ascii=False, indent=2)

    # Compute metrics
    print("\n[Computing metrics...]")
    zs_metrics = compute_precision_recall(zs_summaries, labels, true_labels)
    fs_metrics = compute_precision_recall(fs_summaries, labels, true_labels)

    report = {
        "scenario": args.scenario,
        "total_clusters": n_clusters,
        "zero_shot": {
            "calls": zs_stats["total_calls"],
            "cost": zs_stats["total_cost"],
            **zs_metrics,
        },
        "few_shot": {
            "calls": fs_stats["total_calls"],
            "cost": fs_stats["total_cost"],
            **fs_metrics,
        },
    }

    report_path = output_dir / "precision_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print report
    print("\n" + "=" * 60)
    print("PRECISION EVALUATION REPORT")
    print("=" * 60)
    print(f"\nScenario: {args.scenario}")
    print(f"Total clusters: {n_clusters}")
    print(f"Actual attack-phase clusters: {zs_metrics['total_actual_attack_clusters']}")
    print(f"")
    print(f"{'Metric':<20} {'Zero-shot':>15} {'Few-shot':>15} {'Delta':>10}")
    print("-" * 60)
    for metric in ["precision", "recall", "f1", "accuracy"]:
        zs_val = zs_metrics[metric]
        fs_val = fs_metrics[metric]
        delta = fs_val - zs_val
        print(f"{metric.capitalize():<20} {zs_val*100:>14.2f}% {fs_val*100:>14.2f}% {delta*100:>+9.2f}%")
    print("-" * 60)
    print(f"{'TP':<20} {zs_metrics['tp']:>15} {fs_metrics['tp']:>15}")
    print(f"{'FP':<20} {zs_metrics['fp']:>15} {fs_metrics['fp']:>15}")
    print(f"{'FN':<20} {zs_metrics['fn']:>15} {fs_metrics['fn']:>15}")
    print(f"{'TN':<20} {zs_metrics['tn']:>15} {fs_metrics['tn']:>15}")
    print("-" * 60)
    print(f"{'Cost':<20} ${zs_stats['total_cost']:.4f} ${fs_stats['total_cost']:.4f}")
    print("=" * 60)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
