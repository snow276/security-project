#!/usr/bin/env python3
"""Build attack paradigm knowledge base from training scenarios.

Extracts representative alerts per attack phase from training scenarios,
uses LLM to summarize attack patterns, and saves as a JSON knowledge base.

Usage:
    python build_attack_paradigm.py --output-dir experiments/attack_paradigm
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from hybrid_pipeline.config import LLMConfig, PipelineConfig
from hybrid_pipeline.llm_attack_analyzer import Tier1Analyzer
from hybrid_pipeline.pipeline import load_model_and_data


def extract_phase_samples(scenario, labels, true_labels, phase_name, max_samples=10):
    """Extract representative alert texts for a given attack phase.

    Returns list of alert dicts with key fields.
    """
    mask = true_labels == phase_name
    n = np.sum(mask)
    if n == 0:
        return []

    indices = np.where(mask)[0]
    # Sample up to max_samples evenly distributed
    if len(indices) > max_samples:
        step = len(indices) // max_samples
        indices = indices[::step][:max_samples]

    samples = []
    for idx in indices:
        sample = {
            "short": str(scenario.data.get("short", [""])[idx]),
            "name": str(scenario.data.get("name", [""])[idx]),
            "host": str(scenario.data.get("host", [""])[idx]),
            "ip": str(scenario.data.get("ip", [""])[idx]),
            "raw_time": float(scenario.data.get("raw_time", [0.0])[idx]),
        }
        samples.append(sample)

    return samples


def build_paradigm_with_llm(phase_samples_dict, model="deepseek-v4-flash"):
    """Use LLM to summarize attack patterns from collected samples.

    Args:
        phase_samples_dict: {scenario_name: {phase_name: [samples]}}
        model: LLM model to use

    Returns:
        Dict with attack_paradigm string
    """
    # Build a rich prompt describing all attack phases across training scenarios
    lines = []
    for scenario_name, phases in phase_samples_dict.items():
        lines.append(f"\n=== {scenario_name} ===")
        for phase_name, samples in phases.items():
            if not samples:
                continue
            lines.append(f"\n攻击阶段: {phase_name}")
            lines.append(f"样本数量: {len(samples)}")
            lines.append("典型告警:")
            for i, s in enumerate(samples[:5]):
                lines.append(f"  [{i}] type={s['short']}, host={s['host']}, desc=\"{s['name'][:80]}\"")

    prompt_content = "\n".join(lines)

    system_prompt = """你是一名高级威胁猎手（Threat Hunter）。请根据提供的多个攻击场景样本，总结出一种典型的 Web 应用攻击链范式。

要求：
1. 描述攻击链的完整阶段（从 reconnaissance 到 privilege escalation）
2. 每个阶段列出典型的告警类型（short code）和告警描述模式
3. 指出哪些告警虽然文本看起来良性，但实际上属于攻击的一部分
4. 给出各阶段的时间顺序关系
5. 用中文输出，格式清晰"""

    user_prompt = f"""以下是 3 个攻击场景的样本告警数据。请总结攻击范式。

{prompt_content}

请输出 JSON 格式：
{{
    "attack_chain_name": "攻击链名称",
    "stages": [
        {{
            "stage_name": "阶段名称（中文）",
            "mitre_tactic": "MITRE ATT&CK 战术",
            "typical_alert_types": ["典型告警类型1", "类型2"],
            "alert_text_patterns": ["告警文本模式1", "模式2"],
            "duration_minutes": "典型持续时间",
            "key_indicators": ["关键指标1", "指标2"],
            "benign_looking_but_actually_attack": ["看起来良性但实际是攻击的告警类型"]
        }}
    ],
    "chain_summary": "整个攻击链的简要描述（100字以内）"
}}"""

    # Call LLM
    from hybrid_pipeline.llm_refine import LLMRefiner
    llm_config = LLMConfig(model=model, max_tokens=4096)
    refiner = LLMRefiner(llm_config)
    client = refiner._get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content
    # Try to parse JSON
    try:
        # Direct JSON parse
        paradigm = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Extract from markdown
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            try:
                paradigm = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                paradigm = {"raw_response": raw, "parse_error": True}
        else:
            # Find first JSON object
            brace_start = raw.find("{")
            brace_end = raw.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                try:
                    paradigm = json.loads(raw[brace_start:brace_end + 1])
                except json.JSONDecodeError:
                    paradigm = {"raw_response": raw, "parse_error": True}
            else:
                paradigm = {"raw_response": raw, "parse_error": True}

    return paradigm, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-scenarios", type=int, nargs="+", default=[1, 5, 7])
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--max-samples-per-phase", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BUILDING ATTACK PARADIGM FROM TRAINING SCENARIOS")
    print("=" * 60)

    # Load model and data
    print("\n[Step 1] Loading model and data...")
    config = PipelineConfig()
    grouping_model, data, _, _, _ = load_model_and_data(config)

    # Collect samples from training scenarios
    print(f"\n[Step 2] Collecting attack phase samples from scenarios {args.train_scenarios}...")
    phase_samples = {}

    for scenario_idx in args.train_scenarios:
        scenario = data.scenarios[scenario_idx]
        scenario_name = f"scenario_{scenario_idx}"
        true_labels = scenario.data["hierarchical_event_label"]

        # Run AlertBERT to get clusters
        result = grouping_model(scenario, return_details=True)
        labels = result["labels"]

        # Identify attack phases
        unique_phases = sorted(set(true_labels) - {"-"})
        print(f"\n  {scenario_name}: {len(unique_phases)} attack phases")

        phase_samples[scenario_name] = {}
        for phase in unique_phases:
            samples = extract_phase_samples(
                scenario, labels, true_labels, phase,
                max_samples=args.max_samples_per_phase
            )
            phase_samples[scenario_name][phase] = samples
            print(f"    {phase}: {len(samples)} samples")

    # Save raw samples
    with open(output_dir / "phase_samples.json", "w", encoding="utf-8") as f:
        json.dump(phase_samples, f, indent=2, ensure_ascii=False)

    # Build paradigm with LLM
    print("\n[Step 3] Using LLM to summarize attack paradigm...")
    paradigm, raw_response = build_paradigm_with_llm(phase_samples, model=args.model)

    # Save paradigm
    with open(output_dir / "attack_paradigm.json", "w", encoding="utf-8") as f:
        json.dump(paradigm, f, indent=2, ensure_ascii=False)

    with open(output_dir / "paradigm_raw.txt", "w", encoding="utf-8") as f:
        f.write(raw_response)

    print(f"\nParadigm built and saved to {output_dir}")
    print(f"  phase_samples.json: raw alert samples")
    print(f"  attack_paradigm.json: structured paradigm")
    print(f"  paradigm_raw.txt: LLM raw output")

    # Print summary
    if "stages" in paradigm:
        print(f"\nAttack chain stages identified: {len(paradigm['stages'])}")
        for stage in paradigm["stages"]:
            print(f"  - {stage.get('stage_name', 'unknown')}: {', '.join(stage.get('typical_alert_types', [])[:3])}")


if __name__ == "__main__":
    main()
