#!/usr/bin/env python3
"""Batch evaluate zero-shot vs few-shot LLM analysis across ALL 8 scenarios.

For each scenario:
1. Extract attack-phase clusters from baseline AlertBERT output
2. Run zero-shot LLM analysis
3. Run few-shot LLM analysis (with same attack paradigm)
4. Compare recognition rates

Usage:
    python evaluate_all_scenarios.py --paradigm experiments/attack_paradigm/attack_paradigm.json --output-dir experiments/all_scenarios_eval
"""

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from hybrid_pipeline.config import LLMConfig, PipelineConfig
from hybrid_pipeline.cluster_sampler import ClusterSampler
from hybrid_pipeline.llm_attack_analyzer import Tier1Analyzer
from hybrid_pipeline.llm_refine import LLMRefiner
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


def analyze_clusters(cluster_infos, alert_data, labels, embeddings, use_paradigm=False, paradigm_text=""):
    """Run Tier 1 LLM analysis on given clusters."""
    llm_config = LLMConfig(
        model="deepseek-v4-flash",
        max_cost=5.00,
        max_alerts_per_prompt=5,
    )
    analyzer = Tier1Analyzer(llm_config)

    if use_paradigm and paradigm_text:
        import hybrid_pipeline.llm_attack_analyzer as ata_module
        original_system = ata_module.TIER1_SYSTEM_PROMPT
        ata_module.TIER1_SYSTEM_PROMPT = f"""你是一名资深 SOC（安全运营中心）分析师，擅长从告警数据中识别攻击模式并生成事件摘要。

你接受过以下攻击范式的训练，请在分析时参考：

{paradigm_text}

关键规则：
- 仅基于提供的告警信息进行判断，不要臆测不存在的数据
- 参考上述攻击范式，识别告警簇是否符合已知攻击阶段
- 如果告警文本看起来良性但符合范式中的攻击阶段特征，请提高风险等级
- 始终输出有效的 JSON，不要包含 markdown 代码块标记"""
        try:
            summaries = analyzer.analyze_all_clusters(
                cluster_infos=cluster_infos,
                alert_data=alert_data,
                labels=labels,
                embeddings=embeddings,
            )
        finally:
            ata_module.TIER1_SYSTEM_PROMPT = original_system
    else:
        summaries = analyzer.analyze_all_clusters(
            cluster_infos=cluster_infos,
            alert_data=alert_data,
            labels=labels,
            embeddings=embeddings,
        )

    return summaries, analyzer.get_stats()


def evaluate_scenario(scenario_idx, paradigm, paradigm_text, output_dir):
    """Evaluate a single scenario: zero-shot vs few-shot."""
    print(f"\n{'=' * 60}")
    print(f"EVALUATING SCENARIO {scenario_idx}")
    print(f"{'=' * 60}")

    config = PipelineConfig()
    grouping_model, data, _, _, _ = load_model_and_data(config)
    scenario = data.scenarios[scenario_idx]
    true_labels = scenario.data["hierarchical_event_label"]

    # Run AlertBERT
    result = grouping_model(scenario, return_details=True)
    labels = result["labels"]
    embeddings = result["embeddings"]
    pre_cluster_ids = result["pre_cluster_ids"]

    n_clusters = len(np.unique(labels))
    print(f"Total baseline clusters: {n_clusters}")

    # Identify attack-phase clusters
    unique_phases = sorted(set(true_labels) - {"-"})
    print(f"Attack phases: {len(unique_phases)} -> {unique_phases}")

    clusters_of_interest = set()
    for phase in unique_phases:
        mask = true_labels == phase
        cids = set(labels[mask])
        clusters_of_interest.update(cids)

    print(f"Attack-phase clusters: {len(clusters_of_interest)}")

    # Add benign samples
    benign_mask = true_labels == '-'
    benign_indices = np.where(benign_mask)[0]
    benign_cids = set(labels[benign_indices])
    import random
    random.seed(42)
    sampled_benign = random.sample(sorted(benign_cids), min(20, len(benign_cids)))
    clusters_of_interest.update(sampled_benign)

    print(f"Total to analyze: {len(clusters_of_interest)} ({len(clusters_of_interest)-20} attack + 20 benign)")

    # Score clusters
    alert_data = {}
    for key in ["short", "host", "name", "ip", "raw_time"]:
        if key in scenario.data:
            alert_data[key] = scenario.data[key]

    cluster_sampler = ClusterSampler()
    all_cluster_infos = cluster_sampler.score_clusters(
        labels=labels, embeddings=embeddings, pre_cluster_ids=pre_cluster_ids,
        alert_types=alert_data.get("short"), hosts=alert_data.get("host"), raw_time=alert_data.get("raw_time"),
    )

    target_cids = {float(cid) for cid in clusters_of_interest}
    target_infos = [c for c in all_cluster_infos if float(c.cluster_id) in target_cids]

    # Zero-shot
    print("\nRunning zero-shot...")
    t0 = time.time()
    summaries_zs, stats_zs = analyze_clusters(target_infos, alert_data, labels, embeddings, use_paradigm=False)
    t_zs = time.time() - t0

    # Few-shot
    print("Running few-shot...")
    t0 = time.time()
    summaries_fs, stats_fs = analyze_clusters(target_infos, alert_data, labels, embeddings, use_paradigm=True, paradigm_text=paradigm_text)
    t_fs = time.time() - t0

    # Evaluate
    summary_map_zs = {s.cluster_id: s for s in summaries_zs}
    summary_map_fs = {s.cluster_id: s for s in summaries_fs}

    results = {
        "scenario": scenario_idx,
        "total_clusters": n_clusters,
        "attack_phases": unique_phases,
        "attack_clusters": len(clusters_of_interest) - 20,
        "zero_shot": {"cost": stats_zs["total_cost"], "calls": stats_zs["total_calls"], "time": t_zs},
        "few_shot": {"cost": stats_fs["total_cost"], "calls": stats_fs["total_calls"], "time": t_fs},
        "phases": {},
    }

    correct_zs = 0
    correct_fs = 0
    missed_zs = 0
    missed_fs = 0

    for phase in unique_phases:
        mask = true_labels == phase
        cids = set(labels[mask])
        phase_results = []

        for cid in sorted(cids):
            cid_int = int(cid)
            sz = summary_map_zs.get(cid_int)
            sf = summary_map_fs.get(cid_int)

            is_attack_zs = sz.risk_level in ('high', 'critical', 'medium') if sz else False
            is_attack_fs = sf.risk_level in ('high', 'critical', 'medium') if sf else False

            phase_results.append({
                "cluster_id": cid_int,
                "size": sz.size if sz else 0,
                "zero_shot": {"risk": sz.risk_level if sz else "N/A", "summary": sz.attack_summary if sz else ""},
                "few_shot": {"risk": sf.risk_level if sf else "N/A", "summary": sf.attack_summary if sf else ""},
            })

            if is_attack_zs:
                correct_zs += 1
            else:
                missed_zs += 1
            if is_attack_fs:
                correct_fs += 1
            else:
                missed_fs += 1

        results["phases"][phase] = phase_results

    results["zero_shot"]["correct"] = correct_zs
    results["zero_shot"]["missed"] = missed_zs
    results["few_shot"]["correct"] = correct_fs
    results["few_shot"]["missed"] = missed_fs

    # Save
    scenario_dir = output_dir / f"scenario_{scenario_idx}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    with open(scenario_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nScenario {scenario_idx} complete:")
    print(f"  Zero-shot: {correct_zs}/{correct_zs+missed_zs} correct, ${stats_zs['total_cost']:.4f}")
    print(f"  Few-shot:  {correct_fs}/{correct_fs+missed_fs} correct, ${stats_fs['total_cost']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paradigm", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--scenarios", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--max-cost-per-scenario", type=float, default=3.00)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paradigm = load_paradigm(Path(args.paradigm))
    paradigm_text = build_paradigm_prompt(paradigm)

    print("=" * 60)
    print("BATCH EVALUATION: ZERO-SHOT VS FEW-SHOT ACROSS ALL SCENARIOS")
    print("=" * 60)
    print(f"Paradigm: {paradigm.get('attack_chain_name', 'N/A')}")
    print(f"Scenarios to evaluate: {args.scenarios}")

    all_results = []
    total_cost = 0.0

    for scenario_idx in args.scenarios:
        try:
            result = evaluate_scenario(scenario_idx, paradigm, paradigm_text, output_dir)
            all_results.append(result)
            total_cost += result["zero_shot"]["cost"] + result["few_shot"]["cost"]
        except Exception as e:
            print(f"ERROR on scenario {scenario_idx}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n{'Scenario':<10} {'Attack Phases':<15} {'ZS Correct':<12} {'FS Correct':<12} {'Improvement':<12}")
    print("-" * 65)

    total_zs_correct = 0
    total_zs_total = 0
    total_fs_correct = 0
    total_fs_total = 0

    for r in all_results:
        zs_c = r["zero_shot"]["correct"]
        zs_t = r["zero_shot"]["correct"] + r["zero_shot"]["missed"]
        fs_c = r["few_shot"]["correct"]
        fs_t = r["few_shot"]["correct"] + r["few_shot"]["missed"]
        improvement = f"+{fs_c - zs_c}" if fs_c > zs_c else "=" if fs_c == zs_c else f"{fs_c - zs_c}"

        print(f"Scenario {r['scenario']:<3} {len(r['attack_phases']):<15} {zs_c}/{zs_t:<8} {fs_c}/{fs_t:<8} {improvement}")

        total_zs_correct += zs_c
        total_zs_total += zs_t
        total_fs_correct += fs_c
        total_fs_total += fs_t

    print(f"\n{'TOTAL':<10} {'':<15} {total_zs_correct}/{total_zs_total:<8} {total_fs_correct}/{total_fs_total:<8}")
    print(f"\nTotal cost: ${total_cost:.4f}")
    print(f"Zero-shot accuracy: {total_zs_correct/max(total_zs_total,1)*100:.1f}%")
    print(f"Few-shot accuracy:  {total_fs_correct/max(total_fs_total,1)*100:.1f}%")

    # Save summary
    summary = {
        "all_results": all_results,
        "total_cost": total_cost,
        "zero_shot_accuracy": total_zs_correct / max(total_zs_total, 1),
        "few_shot_accuracy": total_fs_correct / max(total_fs_total, 1),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
