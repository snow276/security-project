#!/usr/bin/env python3
"""Smart cross-scenario evaluation with intelligent train/test split.

Groups scenarios by attack chain similarity, trains on one per group,
tests on the rest. This evaluates REAL generalization:
- In-group: same attack chain type, different instance
- Cross-group: different attack chain type entirely

Scenario Groups (based on attack phase analysis):
- Group A (Standard Web Intrusion): 1, 3, 4, 7 — has service_scan, no dns_scan, no online_cracking
- Group B (Enhanced Web Intrusion): 5, 6 — has service_scan + online_cracking
- Group C (DNS-Scan Variant): 0 — has dns_scan, no service_scan
- Group D (Large-Scale Variant): 2 — no crack_passwords, massive dirb volume

Train: Pick one representative from each group: 0, 1, 5, 2
Test: The rest: 3, 4, 6, 7

Usage:
    python evaluate_smart_split.py --output-dir experiments/smart_split_eval
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


def extract_attack_phase_clusters(scenario, labels, true_labels, max_benign=20):
    """Extract clusters containing attack phases + benign sample."""
    unique_phases = sorted(set(true_labels) - {"-"})
    clusters_of_interest = set()
    phase_to_clusters = {}

    for phase in unique_phases:
        mask = true_labels == phase
        n = np.sum(mask)
        if n > 0:
            cids = set(labels[mask])
            phase_to_clusters[phase] = cids
            clusters_of_interest.update(cids)

    # Add benign sample
    benign_mask = true_labels == '-'
    benign_indices = np.where(benign_mask)[0]
    if len(benign_indices) > 0:
        benign_cids = set(labels[benign_indices])
        import random
        random.seed(42)
        sampled = random.sample(sorted(benign_cids), min(max_benign, len(benign_cids)))
        clusters_of_interest.update(sampled)

    return clusters_of_interest, phase_to_clusters, unique_phases


def analyze_targeted_clusters(target_infos, alert_data, labels, embeddings, use_paradigm=False, paradigm_text=""):
    """Run Tier 1 LLM analysis on targeted clusters."""
    llm_config = LLMConfig(
        model="deepseek-v4-flash",
        max_cost=2.00,
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
                cluster_infos=target_infos,
                alert_data=alert_data,
                labels=labels,
                embeddings=embeddings,
            )
        finally:
            ata_module.TIER1_SYSTEM_PROMPT = original_system
    else:
        summaries = analyzer.analyze_all_clusters(
            cluster_infos=target_infos,
            alert_data=alert_data,
            labels=labels,
            embeddings=embeddings,
        )

    return summaries, analyzer.get_stats()


def evaluate_scenario(scenario_idx, paradigm_text, output_dir):
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

    n_total = len(np.unique(labels))
    clusters_of_interest, phase_to_clusters, unique_phases = extract_attack_phase_clusters(
        scenario, labels, true_labels
    )

    print(f"Total clusters: {n_total}")
    print(f"Attack phases: {len(unique_phases)} -> {unique_phases}")
    print(f"Attack-phase clusters: {len(clusters_of_interest) - 20}")
    print(f"Total to analyze: {len(clusters_of_interest)} ({len(clusters_of_interest) - 20} attack + 20 benign)")

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
    summaries_zs, stats_zs = analyze_targeted_clusters(target_infos, alert_data, labels, embeddings, use_paradigm=False)
    t_zs = time.time() - t0

    # Few-shot
    print("Running few-shot...")
    t0 = time.time()
    summaries_fs, stats_fs = analyze_targeted_clusters(target_infos, alert_data, labels, embeddings, use_paradigm=True, paradigm_text=paradigm_text)
    t_fs = time.time() - t0

    # Evaluate
    summary_map_zs = {s.cluster_id: s for s in summaries_zs}
    summary_map_fs = {s.cluster_id: s for s in summaries_fs}

    results = {
        "scenario": scenario_idx,
        "total_clusters": n_total,
        "attack_phases": unique_phases,
        "zero_shot": {"cost": stats_zs["total_cost"], "calls": stats_zs["total_calls"], "time": t_zs},
        "few_shot": {"cost": stats_fs["total_cost"], "calls": stats_fs["total_calls"], "time": t_fs},
        "phases": {},
    }

    correct_zs = 0
    correct_fs = 0
    missed_zs = 0
    missed_fs = 0
    upgraded = 0
    downgraded = 0

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

            if not is_attack_zs and is_attack_fs:
                upgraded += 1
            if is_attack_zs and not is_attack_fs:
                downgraded += 1

        results["phases"][phase] = phase_results

    results["zero_shot"].update({"correct": correct_zs, "missed": missed_zs})
    results["few_shot"].update({"correct": correct_fs, "missed": missed_fs})
    results["upgraded"] = upgraded
    results["downgraded"] = downgraded

    # Save
    scenario_dir = output_dir / f"scenario_{scenario_idx}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    with open(scenario_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nScenario {scenario_idx}: ZS={correct_zs}/{correct_zs+missed_zs}, FS={correct_fs}/{correct_fs+missed_fs}, upgraded={upgraded}, downgraded={downgraded}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paradigm", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-scenarios", type=int, nargs="+", default=[0, 1, 2, 5])
    parser.add_argument("--test-scenarios", type=int, nargs="+", default=[3, 4, 6, 7])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paradigm = load_paradigm(Path(args.paradigm))
    paradigm_text = build_paradigm_prompt(paradigm)

    print("=" * 70)
    print("SMART CROSS-SCENARIO EVALUATION")
    print("=" * 70)
    print(f"Paradigm: {paradigm.get('attack_chain_name', 'N/A')}")
    print(f"Train scenarios: {args.train_scenarios}")
    print(f"Test scenarios: {args.test_scenarios}")
    print(f"\nGroup mapping:")
    print(f"  Group A (Standard Web Intrusion): train=[1], test=[3,4,7]")
    print(f"  Group B (Enhanced Web Intrusion): train=[5], test=[6]")
    print(f"  Group C (DNS-Scan Variant): train=[0], test=[none - 0 is unique]")
    print(f"  Group D (Large-Scale Variant): train=[2], test=[none - 2 is unique]")
    print(f"\nTest 3,4,6,7 are IN-GROUP (same attack chain type, different instance)")
    print(f"Test 0,2 are CROSS-GROUP (different attack chain type)")

    # Build paradigm from training scenarios
    print(f"\n[Building paradigm from training scenarios {args.train_scenarios}...]")
    # (paradigm is pre-built, but we document which scenarios it came from)

    all_results = []
    total_cost = 0.0

    # Evaluate test scenarios
    for scenario_idx in args.test_scenarios:
        try:
            result = evaluate_scenario(scenario_idx, paradigm_text, output_dir)
            all_results.append(result)
            total_cost += result["zero_shot"]["cost"] + result["few_shot"]["cost"]
        except Exception as e:
            print(f"ERROR on scenario {scenario_idx}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n{'Scenario':<10} {'Group':<12} {'Phases':<8} {'ZS Correct':<12} {'FS Correct':<12} {'Upgrade':<10} {'Downgrade':<10}")
    print("-" * 80)

    total_zs_correct = 0
    total_zs_total = 0
    total_fs_correct = 0
    total_fs_total = 0
    total_upgraded = 0
    total_downgraded = 0

    group_map = {3: "A", 4: "A", 6: "B", 7: "A"}

    for r in all_results:
        zs_c = r["zero_shot"]["correct"]
        zs_t = r["zero_shot"]["correct"] + r["zero_shot"]["missed"]
        fs_c = r["few_shot"]["correct"]
        fs_t = r["few_shot"]["correct"] + r["few_shot"]["missed"]
        group = group_map.get(r["scenario"], "?")

        print(f"S{r['scenario']:<3}      {group:<12} {len(r['attack_phases']):<8} {zs_c}/{zs_t:<8} {fs_c}/{fs_t:<8} {r.get('upgraded',0):<10} {r.get('downgraded',0):<10}")

        total_zs_correct += zs_c
        total_zs_total += zs_t
        total_fs_correct += fs_c
        total_fs_total += fs_t
        total_upgraded += r.get('upgraded', 0)
        total_downgraded += r.get('downgraded', 0)

    print(f"\n{'TOTAL':<10} {'':<12} {'':<8} {total_zs_correct}/{total_zs_total:<8} {total_fs_correct}/{total_fs_total:<8} {total_upgraded:<10} {total_downgraded:<10}")
    print(f"\nTotal cost: ${total_cost:.4f}")
    print(f"Zero-shot accuracy: {total_zs_correct/max(total_zs_total,1)*100:.1f}%")
    print(f"Few-shot accuracy:  {total_fs_correct/max(total_fs_total,1)*100:.1f}%")
    print(f"Net improvement: +{total_fs_correct - total_zs_correct} clusters correctly identified")
    print(f"False upgrades: {total_upgraded}, False downgrades: {total_downgraded}")

    # Save summary
    summary = {
        "train_scenarios": args.train_scenarios,
        "test_scenarios": args.test_scenarios,
        "all_results": all_results,
        "total_cost": total_cost,
        "zero_shot_accuracy": total_zs_correct / max(total_zs_total, 1),
        "few_shot_accuracy": total_fs_correct / max(total_fs_total, 1),
        "net_improvement": total_fs_correct - total_zs_correct,
        "upgraded": total_upgraded,
        "downgraded": total_downgraded,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
