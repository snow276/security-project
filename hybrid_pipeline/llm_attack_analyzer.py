"""Two-tier LLM attack analysis system for SOC alert clustering.

Tier 1: Per-cluster attack summary generation using representative alerts.
Tier 2: Cross-cluster attack chain reasoning for problematic clusters.

Reuses LLMRefiner's client infrastructure from llm_refine.py.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from hybrid_pipeline.config import LLMConfig
from hybrid_pipeline.cluster_sampler import ClusterInfo
from hybrid_pipeline.llm_refine import LLMRefiner

logger = logging.getLogger(__name__)


@dataclass
class Tier1ClusterSummary:
    """Result of Tier 1 analysis for a single cluster."""

    cluster_id: int
    size: int
    time_span: float
    time_min: float
    time_max: float
    alert_types: list[str]
    hosts: list[str]

    # LLM-generated fields
    attack_summary: str = ""
    certainty: float = 0.0
    risk_level: str = "unknown"  # "low", "medium", "high", "critical", "benign"
    recommended_action: str = ""
    key_iocs: list[str] = field(default_factory=list)

    # Metadata
    llm_model_used: str = ""
    llm_call_cost: float = 0.0
    parse_success: bool = True
    raw_response: str = ""
    representative_alert_indices: list[int] = field(default_factory=list)


@dataclass
class Tier2AttackChain:
    """Result of Tier 2 cross-cluster attack chain reasoning."""

    # LLM-generated fields
    attack_timeline: str = ""
    cluster_relationships: list[dict] = field(default_factory=list)
    overall_attack_narrative: str = ""
    suspicious_indicators: list[str] = field(default_factory=list)

    # Metadata
    llm_model_used: str = ""
    llm_call_cost: float = 0.0
    parse_success: bool = True
    raw_response: str = ""
    num_clusters_analyzed: int = 0


@dataclass
class ProblematicClusterConfig:
    """Configuration for identifying clusters that need Tier 2 analysis."""

    # Purity threshold: clusters with purity < this are flagged as mixed
    purity_threshold: float = 0.9

    # Minimum size to flag as potentially interesting
    min_size_for_flag: int = 100

    # Maximum clusters to feed into Tier 2
    max_clusters_tier2: int = 20

    # Critical attack phases that automatically flag a cluster
    critical_phases: list[str] = field(default_factory=lambda: [
        "webshell_cmd",
        "crack_passwords",
        "escalated_sudo_command",
        "dnsteal",
        "attacker_change_user",
    ])


# =============================================================================
# Prompt Templates
# =============================================================================

TIER1_SYSTEM_PROMPT = """你是一名资深 SOC（安全运营中心）分析师，擅长从告警数据中识别攻击模式并生成事件摘要。

你的任务是根据提供的告警簇信息，分析该簇代表的安全事件，并输出结构化的攻击摘要。

关键规则：
- 仅基于提供的告警信息进行判断，不要臆测不存在的数据
- 如果告警全部是良性/管理性事件，明确返回风险等级为 "benign"
- 如果告警涉及多个攻击阶段，在摘要中按时间顺序描述
- 提取关键 IoC（IP、域名、用户名、进程名等）
- 始终输出有效的 JSON，不要包含 markdown 代码块标记"""

TIER1_USER_PROMPT = """告警簇信息：
- 簇 ID: {cluster_id}
- 告警数量: {n_alerts}
- 时间窗口: {time_min:.1f}s 至 {time_max:.1f}s (跨度: {time_span:.1f}s)
- 告警类型: {alert_types}
- 涉及主机: {hosts}

代表性告警（按与簇中心语义距离排序，前 {max_alerts} 条）：
{alert_texts}

请分析该告警簇，判断其是否代表一次安全攻击或异常行为。输出 JSON 格式：
{{
    "attack_summary": "20-50 字的中文攻击描述，说明这是什么类型的攻击或异常行为",
    "certainty": 0.0-1.0,
    "risk_level": "benign" 或 "low" 或 "medium" 或 "high" 或 "critical",
    "recommended_action": "一条具体的处置建议",
    "key_iocs": ["IP地址", "用户名", "域名", "进程名"]
}}

注意：
- benign 仅用于完全没有攻击迹象的簇（如系统更新、常规日志）
- low 用于扫描、探测等低危行为
- medium 用于可疑但不确定的行为
- high 用于明确的攻击行为（暴力破解、webshell、横向移动等）
- critical 用于可能造成严重损害的活跃攻击（勒索软件、数据窃取等）"""

TIER2_SYSTEM_PROMPT = """你是一名资深威胁猎手（Threat Hunter），擅长从多个相关告警簇中还原完整的攻击链条。

你的任务是根据多个告警簇的摘要，分析它们之间的关联关系，还原攻击的时间线和战术意图。

关键规则：
- 仅基于提供的簇摘要进行推理
- 如果簇之间没有明显关联，明确指出
- 按 MITRE ATT&CK 框架描述攻击阶段
- 输出结构化的 JSON"""

TIER2_USER_PROMPT = """以下是 {n_clusters} 个可疑告警簇的摘要信息。请分析它们是否属于同一攻击战役（campaign）的不同阶段，并还原攻击时间线。

簇摘要列表：
{cluster_summaries}

请输出 JSON 格式：
{{
    "attack_timeline": "按时间顺序描述攻击各阶段的中文文本（100-200字）",
    "cluster_relationships": [
        {{
            "from_cluster_id": 123,
            "to_cluster_id": 456,
            "relationship": "reconnaissance_leads_to_initial_access",
            "certainty": 0.85,
            "reasoning": "简要说明为什么这两个簇有关联"
        }}
    ],
    "overall_attack_narrative": "对整个攻击战役的综合描述（50-100字）",
    "suspicious_indicators": ["可疑指标1", "可疑指标2"]
}}

注意：
- cluster_relationships 描述簇之间的因果关系（如扫描→入侵→提权→横向移动）
- 如果某些簇之间没有关联，不要强行建立关系
- 对于 benign 簇，说明它们为何被标记为可疑（可能是误报）"""


# =============================================================================
# Tier 1 Analyzer
# =============================================================================

class Tier1Analyzer:
    """Generates per-cluster attack summaries using LLM.

    For each cluster, selects representative alerts closest to the centroid,
    builds a prompt, calls the LLM, and parses the structured response.
    """

    def __init__(self, config: LLMConfig | None = None, paradigm_text: str | None = None):
        self.config = config or LLMConfig()
        self.paradigm_text = paradigm_text
        self._client = None
        self.total_calls = 0
        self.total_cost = 0.0
        self.total_tokens = 0
        self.parse_failures = 0

    def _get_client(self):
        """Lazy-initialize LLM client (reuses LLMRefiner's pattern)."""
        if self._client is not None:
            return self._client
        # Reuse LLMRefiner's client initialization logic
        refiner = LLMRefiner(self.config)
        self._client = refiner._get_client()
        return self._client

    def _select_representative_alerts(
        self,
        cluster_id: int,
        labels: np.ndarray,
        embeddings: np.ndarray,
        max_alerts: int = 5,
    ) -> list[int]:
        """Select alerts closest to cluster centroid.

        Args:
            cluster_id: The cluster ID to select from
            labels: Cluster labels for all alerts
            embeddings: Alert embeddings (N, D)
            max_alerts: Maximum number of representative alerts

        Returns:
            List of alert indices (global) closest to centroid
        """
        mask = labels == cluster_id
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return []

        if len(indices) <= max_alerts:
            return indices.tolist()

        # Compute centroid (mean of embeddings in this cluster)
        cluster_embeddings = embeddings[indices]
        centroid = np.mean(cluster_embeddings, axis=0)

        # Compute cosine similarity to centroid
        # Normalize vectors for cosine similarity
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        embeddings_norm = cluster_embeddings / (np.linalg.norm(cluster_embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = embeddings_norm @ centroid_norm

        # Select top-k most similar
        top_k = min(max_alerts, len(indices))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return indices[top_indices].tolist()

    def _format_alert_text(
        self,
        alert_indices: list[int],
        alert_data: dict,
        max_alerts: int,
    ) -> str:
        """Format alert data for the LLM prompt."""
        indices = alert_indices[:max_alerts]
        lines = []
        for i, idx in enumerate(indices):
            parts = [f"[{i}]"]
            if "short" in alert_data:
                parts.append(f"type={alert_data['short'][idx]}")
            if "host" in alert_data:
                parts.append(f"host={alert_data['host'][idx]}")
            if "ip" in alert_data:
                parts.append(f"ip={alert_data['ip'][idx]}")
            if "raw_time" in alert_data:
                parts.append(f"t={alert_data['raw_time'][idx]:.1f}s")
            if "name" in alert_data:
                parts.append(f'desc="{alert_data["name"][idx]}"')
            lines.append("  " + " ".join(parts))

        if len(alert_indices) > max_alerts:
            lines.append(f"  ... and {len(alert_indices) - max_alerts} more alerts")

        return "\n".join(lines)

    def _build_tier1_prompt(
        self,
        cluster_info: ClusterInfo,
        alert_data: dict,
        representative_indices: list[int],
    ) -> tuple[str, str]:
        """Build system and user prompts for Tier 1 analysis."""
        alert_texts = self._format_alert_text(
            representative_indices, alert_data, self.config.max_alerts_per_prompt
        )

        alert_types_str = ", ".join(cluster_info.alert_types[:10]) if cluster_info.alert_types else "unknown"
        hosts_str = ", ".join(cluster_info.hosts[:10]) if cluster_info.hosts else "unknown"

        user_prompt = TIER1_USER_PROMPT.format(
            cluster_id=cluster_info.cluster_id,
            n_alerts=cluster_info.size,
            time_min=cluster_info.time_min,
            time_max=cluster_info.time_max,
            time_span=cluster_info.time_span,
            alert_types=alert_types_str,
            hosts=hosts_str,
            alert_texts=alert_texts,
            max_alerts=self.config.max_alerts_per_prompt,
        )

        if self.paradigm_text:
            user_prompt = f"""{self.paradigm_text}

---

{user_prompt}"""

        return TIER1_SYSTEM_PROMPT, user_prompt

    def _parse_tier1_response(self, raw_response: str) -> dict:
        """Parse Tier 1 LLM response with self-healing fallbacks."""
        # Strategy 1: Direct JSON parse
        try:
            return json.loads(raw_response.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_response)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find first { ... } JSON object
        brace_start = raw_response.find("{")
        brace_end = raw_response.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(raw_response[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse Tier 1 LLM response")
        return {
            "attack_summary": "解析失败",
            "certainty": 0.0,
            "risk_level": "unknown",
            "recommended_action": "",
            "key_iocs": [],
        }

    def _validate_tier1_response(self, parsed: dict) -> dict:
        """Validate and normalize Tier 1 response fields."""
        # Normalize attack_summary
        attack_summary = str(parsed.get("attack_summary", "")).strip()
        if not attack_summary:
            attack_summary = "未提供攻击摘要"

        # Validate certainty
        try:
            certainty = float(parsed.get("certainty", 0.0))
            certainty = max(0.0, min(1.0, certainty))
        except (TypeError, ValueError):
            certainty = 0.0

        # Validate risk_level
        risk_level = parsed.get("risk_level", "unknown").lower().strip()
        valid_levels = {"benign", "low", "medium", "high", "critical", "unknown"}
        if risk_level not in valid_levels:
            risk_level = "unknown"

        # Normalize recommended_action
        recommended_action = str(parsed.get("recommended_action", "")).strip()

        # Normalize key_iocs
        key_iocs = parsed.get("key_iocs", [])
        if not isinstance(key_iocs, list):
            key_iocs = []
        key_iocs = [str(ioc).strip() for ioc in key_iocs if str(ioc).strip()]

        return {
            "attack_summary": attack_summary,
            "certainty": certainty,
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "key_iocs": key_iocs,
        }

    def analyze_cluster(
        self,
        cluster_info: ClusterInfo,
        alert_data: dict,
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> Tier1ClusterSummary:
        """Analyze a single cluster and return its summary.

        Args:
            cluster_info: Metadata for the cluster
            alert_data: Dict with per-alert fields
            labels: Cluster labels
            embeddings: Alert embeddings

        Returns:
            Tier1ClusterSummary with LLM-generated attack description
        """
        # Select representative alerts
        rep_indices = self._select_representative_alerts(
            cluster_info.cluster_id, labels, embeddings, max_alerts=5
        )

        # Build prompt
        system_prompt, user_prompt = self._build_tier1_prompt(
            cluster_info, alert_data, rep_indices
        )

        # Check budget
        if self.total_cost >= self.config.max_cost:
            logger.info(
                f"Budget cap ${self.config.max_cost:.2f} exceeded. "
                f"Skipping cluster {cluster_info.cluster_id}."
            )
            return Tier1ClusterSummary(
                cluster_id=cluster_info.cluster_id,
                size=cluster_info.size,
                time_span=cluster_info.time_span,
                time_min=cluster_info.time_min,
                time_max=cluster_info.time_max,
                alert_types=cluster_info.alert_types,
                hosts=cluster_info.hosts,
                attack_summary="预算超限，跳过 LLM 分析",
                certainty=0.0,
                risk_level="unknown",
                parse_success=False,
                representative_alert_indices=rep_indices,
            )

        # Call LLM
        model = self.config.model
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            raw_response = response.choices[0].message.content
            parsed = self._parse_tier1_response(raw_response)
            validated = self._validate_tier1_response(parsed)

            # Track usage
            self.total_calls += 1
            if hasattr(response, "usage") and response.usage:
                self.total_tokens += response.usage.total_tokens
                if "deepseek" in model:
                    if "v4-pro" in model:
                        self.total_cost += (
                            response.usage.prompt_tokens * 1.00 / 1_000_000
                            + response.usage.completion_tokens * 4.00 / 1_000_000
                        )
                    else:
                        self.total_cost += (
                            response.usage.prompt_tokens * 0.10 / 1_000_000
                            + response.usage.completion_tokens * 0.40 / 1_000_000
                        )

            return Tier1ClusterSummary(
                cluster_id=cluster_info.cluster_id,
                size=cluster_info.size,
                time_span=cluster_info.time_span,
                time_min=cluster_info.time_min,
                time_max=cluster_info.time_max,
                alert_types=cluster_info.alert_types,
                hosts=cluster_info.hosts,
                attack_summary=validated["attack_summary"],
                certainty=validated["certainty"],
                risk_level=validated["risk_level"],
                recommended_action=validated["recommended_action"],
                key_iocs=validated["key_iocs"],
                llm_model_used=model,
                llm_call_cost=self.total_cost,
                parse_success=True,
                raw_response=raw_response,
                representative_alert_indices=rep_indices,
            )

        except Exception as e:
            logger.error(f"LLM call failed for cluster {cluster_info.cluster_id}: {e}")
            return Tier1ClusterSummary(
                cluster_id=cluster_info.cluster_id,
                size=cluster_info.size,
                time_span=cluster_info.time_span,
                time_min=cluster_info.time_min,
                time_max=cluster_info.time_max,
                alert_types=cluster_info.alert_types,
                hosts=cluster_info.hosts,
                attack_summary=f"LLM 调用失败: {e}",
                certainty=0.0,
                risk_level="unknown",
                parse_success=False,
                representative_alert_indices=rep_indices,
            )

    def analyze_all_clusters(
        self,
        cluster_infos: list[ClusterInfo],
        alert_data: dict,
        labels: np.ndarray,
        embeddings: np.ndarray,
        max_clusters: int | None = None,
            max_workers: int = 10,
    ) -> list[Tier1ClusterSummary]:
        """Analyze all clusters and collect summaries in parallel.

        Args:
            cluster_infos: List of ClusterInfo for all clusters
            alert_data: Dict with per-alert fields
            labels: Cluster labels
            embeddings: Alert embeddings
            max_clusters: If set, only analyze first N clusters (for testing)
            max_workers: Number of parallel API workers (default 20 for fast I/O)

        Returns:
            List of Tier1ClusterSummary
        """
        import concurrent.futures
        import threading

        clusters_to_analyze = cluster_infos[:max_clusters] if max_clusters else cluster_infos
        _lock = threading.Lock()
        summaries = []

        def analyze_one(cluster_info: ClusterInfo) -> Tier1ClusterSummary:
            summary = self.analyze_cluster(cluster_info, alert_data, labels, embeddings)
            with _lock:
                summaries.append(summary)
            return summary

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(
                executor.map(analyze_one, clusters_to_analyze),
                total=len(clusters_to_analyze),
                desc="Tier 1 analysis",
                unit="cluster",
            ))

        logger.info(
            f"Tier 1 complete: {len(summaries)} clusters analyzed, "
            f"{self.total_calls} LLM calls, ${self.total_cost:.4f} cost, "
            f"{self.parse_failures} parse failures"
        )
        return summaries

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "parse_failures": self.parse_failures,
            "avg_cost_per_call": self.total_cost / max(self.total_calls, 1),
        }


# =============================================================================
# Tier 2 Analyzer
# =============================================================================

class Tier2Analyzer:
    """Performs cross-cluster attack chain reasoning for problematic clusters.

    Identifies suspicious clusters based on purity, size, and critical attack phases,
    then feeds their Tier 1 summaries to the LLM for holistic attack chain analysis.
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        problematic_config: ProblematicClusterConfig | None = None,
    ):
        self.llm_config = llm_config or LLMConfig()
        self.prob_config = problematic_config or ProblematicClusterConfig()
        self._client = None
        self.total_calls = 0
        self.total_cost = 0.0
        self.total_tokens = 0
        self.parse_failures = 0

    def _get_client(self):
        """Lazy-initialize LLM client."""
        if self._client is not None:
            return self._client
        refiner = LLMRefiner(self.llm_config)
        self._client = refiner._get_client()
        return self._client

    def _compute_cluster_purity(
        self,
        cluster_id: int,
        labels: np.ndarray,
        true_labels: np.ndarray,
    ) -> float:
        """Compute purity of a single cluster against ground truth.

        Args:
            cluster_id: The cluster ID
            labels: Predicted cluster labels
            true_labels: Ground truth labels

        Returns:
            Purity score in [0, 1]
        """
        mask = labels == cluster_id
        cluster_true_labels = true_labels[mask]
        if len(cluster_true_labels) == 0:
            return 0.0

        from collections import Counter
        counts = Counter(cluster_true_labels)
        max_count = max(counts.values())
        return max_count / len(cluster_true_labels)

    def _identify_problematic_clusters(
        self,
        cluster_infos: list[ClusterInfo],
        tier1_summaries: list[Tier1ClusterSummary],
        true_labels: np.ndarray,
        labels: np.ndarray,
    ) -> list[int]:
        """Identify clusters that need Tier 2 cross-cluster analysis.

        Criteria (OR logic):
        1. Cluster purity < threshold (mixed ground-truth labels)
        2. Cluster size >= min_size_for_flag
        3. Cluster contains critical attack phase (from ground truth)
        4. LLM flagged risk_level as "high" or "critical"

        Returns:
            List of cluster IDs (at most max_clusters_tier2)
        """
        problematic_ids = set()
        summary_map = {s.cluster_id: s for s in tier1_summaries}

        for cluster_info in cluster_infos:
            cid = cluster_info.cluster_id
            is_problematic = False
            reason = []

            # Criterion 1: Low purity
            purity = self._compute_cluster_purity(cid, labels, true_labels)
            if purity < self.prob_config.purity_threshold:
                is_problematic = True
                reason.append(f"purity={purity:.2f}")

            # Criterion 2: Large size
            if cluster_info.size >= self.prob_config.min_size_for_flag:
                is_problematic = True
                reason.append(f"size={cluster_info.size}")

            # Criterion 3: Contains critical attack phase
            mask = labels == cid
            cluster_true = true_labels[mask]
            for phase in self.prob_config.critical_phases:
                if any(phase in str(lbl) for lbl in cluster_true):
                    is_problematic = True
                    reason.append(f"contains_{phase}")
                    break

            # Criterion 4: High risk from Tier 1
            summary = summary_map.get(cid)
            if summary and summary.risk_level in ("high", "critical"):
                is_problematic = True
                reason.append(f"risk={summary.risk_level}")

            if is_problematic:
                logger.debug(f"Cluster {cid} flagged: {', '.join(reason)}")
                problematic_ids.add(cid)

        # Cap at max_clusters_tier2
        # Sort by priority: critical/high risk first, then by size
        prioritized = []
        for cid in problematic_ids:
            summary = summary_map.get(cid)
            priority = 0
            if summary:
                if summary.risk_level == "critical":
                    priority = 4
                elif summary.risk_level == "high":
                    priority = 3
                elif summary.risk_level == "medium":
                    priority = 2
            # Get cluster info for size
            cinfo = next((c for c in cluster_infos if c.cluster_id == cid), None)
            size = cinfo.size if cinfo else 0
            prioritized.append((priority, size, cid))

        prioritized.sort(reverse=True)
        selected = [cid for _, _, cid in prioritized[: self.prob_config.max_clusters_tier2]]

        logger.info(
            f"Tier 2: {len(problematic_ids)} problematic clusters identified, "
            f"selecting top {len(selected)} for analysis"
        )
        return selected

    def _build_tier2_prompt(
        self,
        problematic_summaries: list[Tier1ClusterSummary],
    ) -> tuple[str, str]:
        """Build system and user prompts for Tier 2 analysis."""
        lines = []
        for i, summary in enumerate(problematic_summaries):
            lines.append(f"\n=== 簇 {summary.cluster_id} ===")
            lines.append(f"告警数量: {summary.size}")
            lines.append(f"时间窗口: {summary.time_min:.1f}s - {summary.time_max:.1f}s")
            lines.append(f"风险等级: {summary.risk_level}")
            lines.append(f"评分: {summary.certainty:.2f}")
            lines.append(f"攻击摘要: {summary.attack_summary}")
            lines.append(f"关键 IoC: {', '.join(summary.key_iocs[:5]) if summary.key_iocs else '无'}")
            lines.append(f"建议处置: {summary.recommended_action}")

        cluster_summaries = "\n".join(lines)

        user_prompt = TIER2_USER_PROMPT.format(
            n_clusters=len(problematic_summaries),
            cluster_summaries=cluster_summaries,
        )

        return TIER2_SYSTEM_PROMPT, user_prompt

    def _parse_tier2_response(self, raw_response: str) -> dict:
        """Parse Tier 2 LLM response with self-healing fallbacks."""
        # Strategy 1: Direct JSON parse
        try:
            return json.loads(raw_response.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_response)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find first { ... } JSON object
        brace_start = raw_response.find("{")
        brace_end = raw_response.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(raw_response[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse Tier 2 LLM response")
        return {
            "attack_timeline": "解析失败",
            "cluster_relationships": [],
            "overall_attack_narrative": "",
            "suspicious_indicators": [],
        }

    def _validate_tier2_response(self, parsed: dict) -> dict:
        """Validate and normalize Tier 2 response fields."""
        attack_timeline = str(parsed.get("attack_timeline", "")).strip()
        if not attack_timeline:
            attack_timeline = "未提供攻击时间线"

        overall_attack_narrative = str(parsed.get("overall_attack_narrative", "")).strip()

        cluster_relationships = parsed.get("cluster_relationships", [])
        if not isinstance(cluster_relationships, list):
            cluster_relationships = []

        # Validate each relationship
        validated_relationships = []
        for rel in cluster_relationships:
            if not isinstance(rel, dict):
                continue
            try:
                validated_relationships.append({
                    "from_cluster_id": int(rel.get("from_cluster_id", -1)),
                    "to_cluster_id": int(rel.get("to_cluster_id", -1)),
                    "relationship": str(rel.get("relationship", "unknown")),
                    "certainty": float(rel.get("certainty", 0.0)),
                    "reasoning": str(rel.get("reasoning", "")),
                })
            except (TypeError, ValueError):
                continue

        suspicious_indicators = parsed.get("suspicious_indicators", [])
        if not isinstance(suspicious_indicators, list):
            suspicious_indicators = []
        suspicious_indicators = [str(s).strip() for s in suspicious_indicators if str(s).strip()]

        return {
            "attack_timeline": attack_timeline,
            "cluster_relationships": validated_relationships,
            "overall_attack_narrative": overall_attack_narrative,
            "suspicious_indicators": suspicious_indicators,
        }

    def analyze(
        self,
        tier1_summaries: list[Tier1ClusterSummary],
        cluster_infos: list[ClusterInfo],
        true_labels: np.ndarray,
        labels: np.ndarray,
    ) -> Tier2AttackChain:
        """Run Tier 2 cross-cluster attack chain reasoning.

        Args:
            tier1_summaries: Results from Tier 1 analysis
            cluster_infos: Cluster metadata
            true_labels: Ground truth labels
            labels: Predicted cluster labels

        Returns:
            Tier2AttackChain with cross-cluster reasoning
        """
        # Identify problematic clusters
        problematic_ids = self._identify_problematic_clusters(
            cluster_infos, tier1_summaries, true_labels, labels
        )

        if not problematic_ids:
            logger.info("No problematic clusters identified, skipping Tier 2")
            return Tier2AttackChain(
                attack_timeline="未发现可疑簇，无需跨簇分析",
                num_clusters_analyzed=0,
            )

        # Get summaries for problematic clusters
        summary_map = {s.cluster_id: s for s in tier1_summaries}
        problematic_summaries = [
            summary_map[cid] for cid in problematic_ids if cid in summary_map
        ]

        if not problematic_summaries:
            logger.warning("No Tier 1 summaries found for problematic clusters")
            return Tier2AttackChain(
                attack_timeline="无法找到 Tier 1 摘要",
                num_clusters_analyzed=0,
            )

        # Build prompt
        system_prompt, user_prompt = self._build_tier2_prompt(problematic_summaries)

        # Check budget
        if self.total_cost >= self.llm_config.max_cost:
            logger.info(f"Budget cap ${self.llm_config.max_cost:.2f} exceeded. Skipping Tier 2.")
            return Tier2AttackChain(
                attack_timeline="预算超限，跳过 Tier 2 分析",
                num_clusters_analyzed=len(problematic_summaries),
                parse_success=False,
            )

        # Call LLM
        model = self.llm_config.model
        try:
            client = self._get_client()
            tier2_max_tokens = min(self.llm_config.max_tokens * 4, 4096)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.llm_config.temperature,
                max_tokens=tier2_max_tokens,
            )

            raw_response = response.choices[0].message.content
            parsed = self._parse_tier2_response(raw_response)
            validated = self._validate_tier2_response(parsed)

            # Track usage
            self.total_calls += 1
            if hasattr(response, "usage") and response.usage:
                self.total_tokens += response.usage.total_tokens
                if "deepseek" in model:
                    if "v4-pro" in model:
                        self.total_cost += (
                            response.usage.prompt_tokens * 1.00 / 1_000_000
                            + response.usage.completion_tokens * 4.00 / 1_000_000
                        )
                    else:
                        self.total_cost += (
                            response.usage.prompt_tokens * 0.10 / 1_000_000
                            + response.usage.completion_tokens * 0.40 / 1_000_000
                        )

            return Tier2AttackChain(
                attack_timeline=validated["attack_timeline"],
                cluster_relationships=validated["cluster_relationships"],
                overall_attack_narrative=validated["overall_attack_narrative"],
                suspicious_indicators=validated["suspicious_indicators"],
                llm_model_used=model,
                llm_call_cost=self.total_cost,
                parse_success=True,
                raw_response=raw_response,
                num_clusters_analyzed=len(problematic_summaries),
            )

        except Exception as e:
            logger.error(f"Tier 2 LLM call failed: {e}")
            return Tier2AttackChain(
                attack_timeline=f"LLM 调用失败: {e}",
                num_clusters_analyzed=len(problematic_summaries),
                parse_success=False,
            )

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "parse_failures": self.parse_failures,
            "avg_cost_per_call": self.total_cost / max(self.total_calls, 1),
        }
