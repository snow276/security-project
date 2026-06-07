#!/usr/bin/env python3
"""Standalone runner for two-tier LLM attack analysis.

Loads pre-computed cluster labels from an experiment directory,
runs Tier 1 (per-cluster summaries) and Tier 2 (cross-cluster reasoning),
and saves all LLM-generated outputs.

Usage:
    python run_llm_analysis.py \
        --experiment-dir experiments/baseline_original \
        --scenario 0 \
        --output-dir experiments/llm_analysis_s0

Quick test mode (first 5 clusters only):
    python run_llm_analysis.py \
        --experiment-dir experiments/baseline_original \
        --scenario 0 \
        --output-dir /tmp/llm_test \
        --max-clusters-tier1 5 \
        --max-clusters-tier2 3 \
        --max-cost 0.02
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_pipeline.config import LLMConfig, PipelineConfig
from hybrid_pipeline.llm_attack_analyzer import ProblematicClusterConfig
from hybrid_pipeline.cluster_sampler import ClusterSampler
from hybrid_pipeline.llm_attack_analyzer import Tier1Analyzer, Tier2Analyzer
from hybrid_pipeline.pipeline import load_model_and_data

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_experiment_labels(experiment_dir: Path, scenario_idx: int) -> np.ndarray:
    """Load cluster labels from a previous experiment's npz file.

    Args:
        experiment_dir: Path to experiment directory containing *_labels.npz
        scenario_idx: Scenario index

    Returns:
        Cluster labels array
    """
    npz_file = experiment_dir / f"scenario_{scenario_idx}_labels.npz"
    if not npz_file.exists():
        # Try alternative naming
        candidates = list(experiment_dir.glob("*_labels.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"No *_labels.npz found in {experiment_dir}. "
                "Please run an experiment first (e.g., hybrid_pipeline.run_hybrid)."
            )
        npz_file = candidates[0]

    data = np.load(npz_file)
    if "labels" in data:
        labels = data["labels"]
    elif "refined_labels" in data:
        labels = data["refined_labels"]
    else:
        raise KeyError(f"No 'labels' or 'refined_labels' in {npz_file}")

    logger.info(f"Loaded labels from {npz_file}: {len(np.unique(labels))} clusters")
    return labels


def save_results(
    output_dir: Path,
    tier1_summaries: list,
    tier2_result,
    tier1_stats: dict,
    tier2_stats: dict,
    config: dict,
    timing: dict,
):
    """Save all LLM analysis results to output directory.

    Args:
        output_dir: Directory to save results
        tier1_summaries: List of Tier1ClusterSummary
        tier2_result: Tier2AttackChain
        tier1_stats: Tier1 usage stats
        tier2_stats: Tier2 usage stats
        config: Configuration dict
        timing: Timing dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Tier 1 summaries
    tier1_data = []
    for summary in tier1_summaries:
        d = asdict(summary)
        # Convert numpy arrays to lists for JSON serialization
        for key, val in d.items():
            if isinstance(val, np.ndarray):
                d[key] = val.tolist()
        tier1_data.append(d)

    with open(output_dir / "tier1_summaries.json", "w", encoding="utf-8") as f:
        json.dump(tier1_data, f, indent=2, ensure_ascii=False)

    # Save Tier 2 result
    tier2_data = asdict(tier2_result)
    for key, val in tier2_data.items():
        if isinstance(val, np.ndarray):
            tier2_data[key] = val.tolist()

    with open(output_dir / "tier2_reasoning.json", "w", encoding="utf-8") as f:
        json.dump(tier2_data, f, indent=2, ensure_ascii=False)

    # Save metadata
    metadata = {
        "config": config,
        "timing": timing,
        "tier1_stats": tier1_stats,
        "tier2_stats": tier2_stats,
        "total_cost": tier1_stats.get("total_cost", 0.0) + tier2_stats.get("total_cost", 0.0),
        "total_llm_calls": tier1_stats.get("total_calls", 0) + tier2_stats.get("total_calls", 0),
        "num_tier1_clusters": len(tier1_summaries),
        "num_tier2_clusters": tier2_result.num_clusters_analyzed,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  Tier 1: {len(tier1_summaries)} summaries")
    logger.info(f"  Tier 2: {tier2_result.num_clusters_analyzed} clusters analyzed")
    logger.info(f"  Total cost: ${metadata['total_cost']:.4f}")


def run_analysis(args: argparse.Namespace):
    """Run the full two-tier LLM analysis pipeline.

    Args:
        args: Parsed CLI arguments
    """
    start_time = time.time()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("TWO-TIER LLM ATTACK ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Experiment dir: {experiment_dir}")
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max cost: ${args.max_cost:.2f}")

    # ------------------------------------------------------------------
    # Step 1: Load model and data
    # ------------------------------------------------------------------
    logger.info("\n[Step 1] Loading AlertBERT model and dataset...")
    step1_start = time.time()

    # Use default pipeline config for model loading
    pipeline_config = PipelineConfig(
        model_id=args.model_id,
        aitads_a_config=args.aitads_a_config,
    )

    grouping_model, data, label_vocabs, data_tools, model_param_dicts = load_model_and_data(
        pipeline_config
    )

    scenario = data.scenarios[args.scenario]
    scenario_name = getattr(scenario, "name", f"scenario_{args.scenario}")
    logger.info(f"Loaded scenario: {scenario_name}")
    logger.info(f"  Total alerts: {len(scenario.data.get('short', []))}")

    step1_time = time.time() - step1_start
    logger.info(f"  Step 1 took {step1_time:.2f}s")

    # ------------------------------------------------------------------
    # Step 2: Load cluster labels from experiment
    # ------------------------------------------------------------------
    logger.info("\n[Step 2] Loading cluster labels...")
    step2_start = time.time()

    labels = load_experiment_labels(experiment_dir, args.scenario)

    # Also get ground truth labels
    target = "hierarchical_event_label"
    if target in scenario.data:
        true_labels = label_vocabs[target]([scenario.data[target]]).numpy().squeeze()
    else:
        true_labels = None
        logger.warning("No ground truth labels found")

    step2_time = time.time() - step2_start
    logger.info(f"  Step 2 took {step2_time:.2f}s")

    # ------------------------------------------------------------------
    # Step 3: Run AlertBERT to get embeddings
    # ------------------------------------------------------------------
    logger.info("\n[Step 3] Running AlertBERT clustering with details...")
    step3_start = time.time()

    result = grouping_model(scenario, return_details=True)
    embeddings = result["embeddings"]
    pre_cluster_ids = result["pre_cluster_ids"]

    logger.info(f"  Embeddings shape: {embeddings.shape}")
    step3_time = time.time() - step3_start
    logger.info(f"  Step 3 took {step3_time:.2f}s")

    # ------------------------------------------------------------------
    # Step 4: Score clusters
    # ------------------------------------------------------------------
    logger.info("\n[Step 4] Scoring clusters...")
    step4_start = time.time()

    alert_data = {}
    for key in ["short", "host", "name", "ip", "raw_time"]:
        if key in scenario.data:
            alert_data[key] = scenario.data[key]

    cluster_sampler = ClusterSampler()
    cluster_infos = cluster_sampler.score_clusters(
        labels=labels,
        embeddings=embeddings,
        pre_cluster_ids=pre_cluster_ids,
        alert_types=alert_data.get("short"),
        hosts=alert_data.get("host"),
        raw_time=alert_data.get("raw_time"),
    )

    summary = cluster_sampler.summarize(cluster_infos)
    logger.info(
        f"  {summary['total_clusters']} clusters: "
        f"{summary['flagged_clusters']} flagged"
    )

    step4_time = time.time() - step4_start
    logger.info(f"  Step 4 took {step4_time:.2f}s")

    # ------------------------------------------------------------------
    # Step 5: Tier 1 — Per-cluster attack summaries
    # ------------------------------------------------------------------
    logger.info("\n[Step 5] Running Tier 1: Per-cluster attack summaries...")
    step5_start = time.time()

    llm_config = LLMConfig(
        model=args.model,
        max_cost=args.max_cost,
        max_alerts_per_prompt=5,
    )

    tier1_analyzer = Tier1Analyzer(llm_config)
    tier1_summaries = tier1_analyzer.analyze_all_clusters(
        cluster_infos=cluster_infos,
        alert_data=alert_data,
        labels=labels,
        embeddings=embeddings,
        max_clusters=args.max_clusters_tier1,
    )

    tier1_stats = tier1_analyzer.get_stats()
    logger.info(
        f"  Tier 1 complete: {tier1_stats['total_calls']} calls, "
        f"${tier1_stats['total_cost']:.4f} cost"
    )

    # Print sample summaries
    high_risk = [s for s in tier1_summaries if s.risk_level in ("high", "critical")]
    logger.info(f"  High/critical risk clusters: {len(high_risk)}")
    for s in high_risk[:3]:
        logger.info(f"    Cluster {s.cluster_id}: {s.attack_summary} (risk={s.risk_level}, certainty={s.certainty:.2f})")

    step5_time = time.time() - step5_start
    logger.info(f"  Step 5 took {step5_time:.2f}s")

    # ------------------------------------------------------------------
    # Step 6: Tier 2 — Cross-cluster attack chain reasoning
    # ------------------------------------------------------------------
    logger.info("\n[Step 6] Running Tier 2: Cross-cluster attack chain reasoning...")
    step6_start = time.time()

    if true_labels is not None:
        prob_config = ProblematicClusterConfig(
            max_clusters_tier2=args.max_clusters_tier2,
        )
        tier2_analyzer = Tier2Analyzer(llm_config, prob_config)
        tier2_result = tier2_analyzer.analyze(
            tier1_summaries=tier1_summaries,
            cluster_infos=cluster_infos,
            true_labels=true_labels,
            labels=labels,
        )
        tier2_stats = tier2_analyzer.get_stats()
    else:
        logger.warning("Skipping Tier 2 — no ground truth labels available")
        tier2_result = Tier2Analyzer.__new__(Tier2Analyzer)
        # Create a minimal result
        from hybrid_pipeline.llm_attack_analyzer import Tier2AttackChain
        tier2_result = Tier2AttackChain(
            attack_timeline="无真实标签，跳过 Tier 2 分析",
            num_clusters_analyzed=0,
        )
        tier2_stats = {"total_calls": 0, "total_cost": 0.0}

    logger.info(
        f"  Tier 2 complete: {tier2_stats['total_calls']} calls, "
        f"${tier2_stats['total_cost']:.4f} cost"
    )
    if tier2_result.overall_attack_narrative:
        logger.info(f"  Attack narrative: {tier2_result.overall_attack_narrative[:200]}...")

    step6_time = time.time() - step6_start
    logger.info(f"  Step 6 took {step6_time:.2f}s")

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    logger.info("\n[Step 7] Saving results...")

    config_dict = {
        "experiment_dir": str(experiment_dir),
        "scenario": args.scenario,
        "model": args.model,
        "max_cost": args.max_cost,
        "max_clusters_tier1": args.max_clusters_tier1,
        "max_clusters_tier2": args.max_clusters_tier2,
    }

    timing = {
        "step1_load_model": step1_time,
        "step2_load_labels": step2_time,
        "step3_embeddings": step3_time,
        "step4_score_clusters": step4_time,
        "step5_tier1": step5_time,
        "step6_tier2": step6_time,
        "total": time.time() - start_time,
    }

    save_results(
        output_dir=output_dir,
        tier1_summaries=tier1_summaries,
        tier2_result=tier2_result,
        tier1_stats=tier1_stats,
        tier2_stats=tier2_stats,
        config=config_dict,
        timing=timing,
    )

    total_time = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"ANALYSIS COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Total LLM calls: {tier1_stats['total_calls'] + tier2_stats['total_calls']}")
    logger.info(f"Total cost: ${tier1_stats['total_cost'] + tier2_stats['total_cost']:.4f}")
    logger.info(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Two-tier LLM attack analysis for AlertBERT clusters"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory containing pre-computed experiment labels (e.g., experiments/baseline_original)",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=0,
        help="Scenario index to analyze (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for LLM analysis results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v4-flash",
        help="LLM model to use (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mlm_1l_4h_16d_original_default_params_60k",
        help="AlertBERT model ID",
    )
    parser.add_argument(
        "--aitads-a-config",
        type=str,
        default="original",
        help="AIT-ADS-A configuration",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=float("inf"),
        help="Maximum LLM API cost in dollars (default: unlimited)",
    )
    parser.add_argument(
        "--max-clusters-tier1",
        type=int,
        default=None,
        help="Maximum clusters to analyze in Tier 1 (default: all)",
    )
    parser.add_argument(
        "--max-clusters-tier2",
        type=int,
        default=20,
        help="Maximum clusters to analyze in Tier 2 (default: 20)",
    )

    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
