"""End-to-end hybrid pipeline orchestration.

Runs AlertBERT clustering → cluster sampling → LLM refinement → evaluation.
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from hybrid_pipeline.config import PipelineConfig
from hybrid_pipeline.cluster_sampler import ClusterSampler
from hybrid_pipeline.llm_refine import LLMRefiner
from hybrid_pipeline.evaluate import evaluate_hybrid

logger = logging.getLogger(__name__)


def load_model_and_data(config: PipelineConfig):
    """Load the trained AlertBERT model and dataset.

    Args:
        config: Pipeline configuration

    Returns:
        Tuple of (model, collate_fn, data_tools, scenarios, label_vocabs)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "AlertBERT"))

    from alertbert.aitads import AITAlertDataset, MultiAlertDataset
    from alertbert.model_eval_utils import load_data_tools, load_ground_truth_label_vocabs, load_models, load_reports
    from alertbert.models import AlertBERT, MaskedLangModelInferenceWrapper
    from alertbert.preprocessing import BaseSequenceCollate

    saved_models_path = os.path.join(
        str(Path(__file__).parent.parent / "AlertBERT"), config.saved_models_path
    )
    alertbert_path = str(Path(__file__).parent.parent / "AlertBERT")

    # Load reports and model params
    reports, model_param_dicts = load_reports([config.model_id], saved_models_path)

    # Load ground truth label vocabs
    label_vocabs = load_ground_truth_label_vocabs(
        saved_models_path, config.aitads_a_config
    )

    # Load data tools
    data_tools = load_data_tools(
        [config.model_id], model_param_dicts, saved_models_path, label_vocabs
    )

    # Load model
    device = torch.device(config.device)
    models = load_models(model_param_dicts, saved_models_path, data_tools, device)

    # Use CPU for inference by default (matching original eval code)
    # The GPU is mainly needed for training; inference works on CPU
    if config.device == "cuda:0":
        # If GPU is requested, model is already on GPU from load_models
        pass
    model = models[config.model_id]

    # Create inference wrapper
    layers = model_param_dicts[config.model_id].get("layers", ("embedding", "encoder"))
    inf_wrapper = MaskedLangModelInferenceWrapper(model, layers)

    # Load dataset using AITAlertDataset factory
    # AITAlertDataset returns AITAlertDatasetAugmented for "augmented" flavour
    # with configuration parameter
    data = AITAlertDataset(
        flavour="augmented",
        split="all",
        configuration=config.aitads_a_config,
        path=os.path.join(alertbert_path, "aitads_augmented"),
    )

    # Create collate function
    collate_fn = data_tools[config.model_id]["inf_coll_fn"]

    # Create AlertBERT grouping model
    grouping_model = AlertBERT(
        model=inf_wrapper,
        collate_fn=collate_fn,
        dim_reduction=config.dim_reduction,
        delta=config.delta,
        theta=config.theta,
    )

    return grouping_model, data, label_vocabs, data_tools, model_param_dicts


def run_pipeline(config: PipelineConfig, scenarios: list | None = None):
    """Run the full hybrid pipeline.

    Args:
        config: Pipeline configuration
        scenarios: List of scenario indices to run (None = all)

    Returns:
        Dict with results for each scenario
    """
    logger.info("Loading model and data...")
    grouping_model, data, label_vocabs, data_tools, model_param_dicts = load_model_and_data(
        config
    )

    # Determine pipeline mode
    do_llm = config.llm_enabled and config.mode in ("full", "llm_only")

    logger.info(f"Pipeline mode: {config.mode} (llm_refine={do_llm})")

    # Initialize processors
    cluster_sampler = ClusterSampler(config.sampler)
    llm_refiner = LLMRefiner(config.llm) if do_llm else None

    results = {}

    # Run on specified scenarios (or all)
    scenario_indices = scenarios if scenarios is not None else range(len(data.scenarios))

    for scenario_idx in tqdm(scenario_indices, desc="Scenarios", unit="scenario"):
        scenario = data.scenarios[scenario_idx]
        scenario_name = getattr(scenario, "name", f"scenario_{scenario_idx}")
        logger.info(f"Processing {scenario_name} (index {scenario_idx})...")

        # Step 1: Run AlertBERT clustering with details
        logger.info(f"  Step 1: Running AlertBERT clustering...")
        start_time = time.time()
        result = grouping_model(scenario, return_details=True)
        clustering_time = time.time() - start_time

        labels = result["labels"]
        embeddings = result["embeddings"]
        pre_cluster_ids = result["pre_cluster_ids"]
        model_params = result["model_params"]

        n_clusters = len(np.unique(labels))
        logger.info(f"  Produced {n_clusters} clusters in {clustering_time:.2f}s")

        # Step 2: Score clusters for quality
        logger.info(f"  Step 2: Scoring clusters for quality...")
        alert_data = {}
        for key in ["short", "host", "name", "raw_time"]:
            if key in scenario.data:
                alert_data[key] = scenario.data[key]

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
            f"{summary['flagged_clusters']} flagged "
            f"({summary['flagged_pct']:.1f}%)"
        )
        logger.info(
            f"  Routing: skip={summary['routing']['skip']}, "
            f"cheap={summary['routing']['cheap']}, "
            f"expensive={summary['routing']['expensive']}"
        )

        # Store baseline labels (before any modifications)
        baseline_labels = labels.copy()

        # Step 3: LLM refinement for flagged clusters
        refined_labels = labels.copy()
        refinement_results = []

        if do_llm and llm_refiner is not None:
            low_conf_clusters = cluster_sampler.get_flagged_clusters(cluster_infos)

            if config.quick and low_conf_clusters:
                total_before = len(low_conf_clusters)
                low_conf_clusters = low_conf_clusters[::3]
                logger.info(
                    f"  Quick mode: sampling every 3rd cluster → "
                    f"{len(low_conf_clusters)} of {total_before} to refine"
                )
            logger.info(f"  Step 4: Refining {len(low_conf_clusters)} flagged clusters with LLM...")

            for cluster_info in tqdm(low_conf_clusters, desc=f"  LLM refine", unit="cluster", leave=False):
                logger.debug(
                    f"    Refining cluster {cluster_info.cluster_id} "
                    f"(size={cluster_info.size}, tier={cluster_info.routing_tier})"
                )
                refinement = llm_refiner.refine_cluster(
                    cluster_info=cluster_info,
                    alert_data=alert_data,
                    cluster_labels=refined_labels,
                    all_cluster_infos=cluster_infos,
                )
                refinement_results.append(refinement)

                refined_labels = llm_refiner.apply_refinement(
                    refined_labels, refinement, cluster_info
                )

            logger.info(f"  LLM refinement complete: {len(refinement_results)} clusters processed")
            if llm_refiner.total_calls > 0:
                stats = llm_refiner.get_stats()
                logger.info(
                    f"  LLM stats: {stats['total_calls']} calls, "
                    f"${stats['total_cost']:.4f} cost, "
                    f"{stats['parse_failures']} parse failures"
                )
        else:
            logger.info("  Step 4: LLM refinement disabled, skipping")

        # Step 5: Evaluate
        logger.info(f"  Step 5: Evaluating results...")
        n_baseline_clusters = len(np.unique(baseline_labels))
        n_refined_clusters = len(np.unique(refined_labels))
        logger.info(
            f"  Baseline: {n_baseline_clusters} clusters → "
            f"Final: {n_refined_clusters} clusters"
        )

        # Get ground truth
        target = "hierarchical_event_label"
        if target in scenario.data:
            true_labels = label_vocabs[target]([scenario.data[target]]).numpy().squeeze()
        else:
            true_labels = None
            logger.warning(f"  No ground truth labels found for {scenario_name}")

        results[scenario_name] = {
            "baseline_labels": baseline_labels,
            "labels": labels,
            "refined_labels": refined_labels,
            "true_labels": true_labels,
            "cluster_infos": cluster_infos,
            "refinement_results": refinement_results,
            "summary": summary,
            "clustering_time": clustering_time,
            "n_baseline_clusters": n_baseline_clusters,
            "n_clusters": len(np.unique(labels)),
            "n_refined_clusters": n_refined_clusters,
        }

    # Step 6: Aggregate evaluation
    logger.info("Aggregating results across scenarios...")
    eval_results = evaluate_hybrid(results, label_vocabs, config)

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "config": {
            "mode": config.mode,
            "model_id": config.model_id,
            "aitads_a_config": config.aitads_a_config,
            "delta": config.delta,
            "theta": config.theta,
            "sampler": {
                "min_cluster_size": config.sampler.min_cluster_size,
                "max_intra_cluster_variance": config.sampler.max_intra_cluster_variance,
                "boundary_alpha": config.sampler.boundary_alpha,
                "boundary_fraction_threshold": config.sampler.boundary_fraction_threshold,
                "max_time_span": config.sampler.max_time_span,
                "tau_high": config.sampler.tau_high,
                "tau_low": config.sampler.tau_low,
            },
            "llm_enabled": config.llm_enabled,
            "llm_model": config.llm.model if config.llm_enabled else None,
        },
        "eval_summary": eval_results.get("summary", {}),
        "llm_stats": llm_refiner.get_stats() if do_llm and llm_refiner else {},
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    # Save numpy arrays (include labels for comparison)
    for scenario_name, scenario_data in results.items():
        np.savez(
            output_dir / f"{scenario_name}_labels.npz",
            baseline_labels=scenario_data["baseline_labels"],
            labels=scenario_data["labels"],
            refined_labels=scenario_data["refined_labels"],
            true_labels=scenario_data["true_labels"] if scenario_data["true_labels"] is not None else np.array([]),
        )

    logger.info(f"Results saved to {output_dir}")

    return results, eval_results