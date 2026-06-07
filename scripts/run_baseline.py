#!/usr/bin/env python3
"""Run baseline dry-run across all 8 AIT-ADS-A scenarios."""

import time, json, numpy as np
from pathlib import Path
from hybrid_pipeline.config import PipelineConfig, SamplerConfig
from hybrid_pipeline.pipeline import load_model_and_data
from hybrid_pipeline.cluster_sampler import ClusterSampler
from hybrid_pipeline.evaluate import evaluate_hybrid, print_evaluation_report

config = PipelineConfig(
    model_id='mlm_1l_4h_16d_original_default_params_60k',
    aitads_a_config='original',
    saved_models_path='saved_models',
    llm_enabled=False,
    output_dir='experiments/baseline_original',
)

print('Loading model and data...', flush=True)
start = time.time()
grouping_model, data, label_vocabs, _, _ = load_model_and_data(config)
scorer = ClusterSampler(config.sampler)
print(f'Model loaded in {time.time()-start:.1f}s', flush=True)

results = {}
for i in range(len(data.scenarios)):
    scenario = data.scenarios[i]
    scenario_name = getattr(scenario, 'name', f'scenario_{i}')
    print(f'Processing {scenario_name} ({i+1}/{len(data.scenarios)})...', flush=True)

    result = grouping_model(scenario, return_details=True)
    labels = result['labels']
    embeddings = result['embeddings']
    pre_cluster_ids = result['pre_cluster_ids']

    alert_data = {}
    for key in ['short', 'host', 'name', 'raw_time']:
        if key in scenario.data:
            alert_data[key] = scenario.data[key]

    cluster_infos = scorer.score_clusters(
        labels=labels, embeddings=embeddings,
        pre_cluster_ids=pre_cluster_ids,
        alert_types=alert_data.get('short'),
        hosts=alert_data.get('host'),
        raw_time=alert_data.get('raw_time'),
    )
    summary = scorer.summarize(cluster_infos)

    baseline_labels = labels.copy()
    target = 'hierarchical_event_label'
    true_labels = label_vocabs[target]([scenario.data[target]]).numpy().squeeze() if target in scenario.data else None

    results[scenario_name] = {
        'baseline_labels': baseline_labels,
        'refined_labels': baseline_labels.copy(),
        'true_labels': true_labels,
        'cluster_infos': cluster_infos,
        'refinement_results': [],
        'summary': summary,
        'n_baseline_clusters': len(np.unique(labels)),
        'n_refined_clusters': len(np.unique(labels)),
    }
    print(f'  Done: {summary["total_clusters"]} clusters, low-conf={summary["flagged_pct"]:.1f}%', flush=True)

eval_results = evaluate_hybrid(results, label_vocabs, config)
report = print_evaluation_report(eval_results)
print(report)

# Save results
output_dir = Path('experiments/baseline_original')
output_dir.mkdir(parents=True, exist_ok=True)
summary_data = {
    'config': {
        'model_id': config.model_id, 'aitads_a_config': config.aitads_a_config,
        'sampler': {
            'min_cluster_size': config.sampler.min_cluster_size,
            'max_intra_cluster_variance': config.sampler.max_intra_cluster_variance,
            'boundary_alpha': config.sampler.boundary_alpha,
            'boundary_fraction_threshold': config.sampler.boundary_fraction_threshold,
            'max_time_span': config.sampler.max_time_span,
            'tau_high': config.sampler.tau_high,
            'tau_low': config.sampler.tau_low,
        },
        'llm_enabled': False,
    },
    'eval_summary': eval_results.get('summary', {}),
}
with open(output_dir / 'summary.json', 'w') as f:
    json.dump(summary_data, f, indent=2, default=str)
for sn, sd in results.items():
    np.savez(output_dir / f'{sn}_labels.npz',
              baseline_labels=sd['baseline_labels'],
              refined_labels=sd['refined_labels'],
              true_labels=sd['true_labels'] if sd['true_labels'] is not None else np.array([]))
print(f'\nResults saved to {output_dir}')