"""LLM refinement module for flagged cluster validation.

Routes flagged clusters to an LLM for semantic analysis and
potential restructuring (split/keep decisions). Supports OpenAI and
Ollama backends with structured JSON output parsing and self-healing.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hybrid_pipeline.config import LLMConfig
from hybrid_pipeline.cluster_sampler import ClusterInfo

logger = logging.getLogger(__name__)

# Category normalization map for LLM output
ALERT_CATEGORY_ALIASES = {
    "intrusion": "intrusion_attempt",
    "privesc": "privilege_escalation",
    "c2": "command_and_control",
    "recon": "reconnaissance",
    "exfil": "data_exfiltration",
    "exfiltration": "data_exfiltration",
    "bruteforce": "brute_force",
    "brute-force": "brute_force",
    "malware": "malicious_activity",
    "scan": "network_scan",
    "dos": "denial_of_service",
    "phish": "phishing",
}


@dataclass
class RefinementResult:
    """Result of LLM refinement for a single cluster."""

    cluster_id: int
    verdict: str  # "keep" or "split"
    certainty: float  # 0.0-1.0
    reasoning: str
    sub_groups: list[dict] = field(default_factory=list)
    suggested_label: str = ""
    llm_model_used: str = ""
    llm_call_cost: float = 0.0
    parse_success: bool = True
    raw_response: str = ""


REFINEMENT_SYSTEM_PROMPT = """You are an expert SOC (Security Operations Center) analyst specializing in alert triage and clustering.

Your task is to evaluate whether a group of security alerts should remain clustered together or be split into separate groups.

You will receive:
- A cluster of alerts produced by an automated density-based clustering algorithm (AlertBERT)
- The algorithm uses a combined time-cosine metric to group alerts
- Some clusters may contain alerts that are semantically unrelated despite temporal proximity

CRITICAL RULES:
- Base your assessment ONLY on the provided alert information
- If all alerts share a clear common theme, return "keep"
- If alerts represent clearly different themes, return "split" with sub-groups
- If uncertain, return "keep" with flagged
- Do NOT hallucinate alert details not present in the data
- Always output valid JSON matching the specified schema"""

REFINEMENT_USER_PROMPT = """CLUSTER INFO:
- Cluster ID: {cluster_id}
- Size: {n_alerts} alerts
- Time window: {time_min:.1f}s to {time_max:.1f}s (span: {time_span:.1f}s)
- Alert types in cluster: {alert_types}
- Source hosts in cluster: {hosts}
- Inter-cluster context: {context}

REPRESENTATIVE ALERTS (up to {max_alerts} most central by embedding distance):
{alert_texts}

ANALYSIS STEPS:
1. Review the alert types and hosts for semantic coherence
2. Check if all alerts share a common theme or represent mixed topics
3. Determine if the cluster should be kept as-is or split

OUTPUT (JSON only, no markdown):
{{
    "verdict": "keep" or "split",
    "certainty": 0.0-1.0,
    "reasoning": "string explaining your decision",
    "sub_groups": [{{"alert_indices": [0, 2, 5], "theme": "description"}}],
    "suggested_label": "concise 3-5 word cluster description"
}}"""


class LLMRefiner:
    """Routes flagged clusters to LLM for refinement.

    Supports OpenAI API and Ollama backends. Uses structured JSON output
    with self-healing parse fallbacks.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None
        self.total_calls = 0
        self.total_cost = 0.0
        self.total_tokens = 0
        self.parse_failures = 0

    def _get_client(self):
        """Lazy-initialize the LLM client.

        DeepSeek API is OpenAI-compatible, so we use the OpenAI SDK
        with base_url='https://api.deepseek.com'.
        """
        if self._client is not None:
            return self._client

        if self.config.provider in ("deepseek", "openai"):
            try:
                from openai import OpenAI

                kwargs = {"api_key": self.config.api_key}
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
        elif self.config.provider == "ollama":
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key="ollama",
                    base_url=self.config.base_url or "http://localhost:11434/v1",
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed for Ollama backend. Install with: pip install openai"
                )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

        return self._client

    def _select_model(self, routing_tier: str) -> str:
        """Select model based on routing tier."""
        if routing_tier == "cheap":
            return self.config.cheap_model
        elif routing_tier == "expensive":
            return self.config.expensive_model
        else:
            return self.config.model

    def _format_alert_text(
        self,
        alert_indices: list[int],
        alert_data: dict,
        max_alerts: int,
    ) -> str:
        """Format alert data for the LLM prompt.

        Args:
            alert_indices: Indices of alerts in this cluster
            alert_data: Dict with keys 'short', 'host', 'raw_time', etc.
            max_alerts: Maximum number of alerts to include

        Returns:
            Formatted string of alert descriptions.
        """
        # Truncate if too many alerts
        indices = alert_indices[:max_alerts]
        lines = []
        for i, idx in enumerate(indices):
            parts = [f"[{i}]"]
            if "short" in alert_data:
                parts.append(f"type={alert_data['short'][idx]}")
            if "host" in alert_data:
                parts.append(f"host={alert_data['host'][idx]}")
            if "raw_time" in alert_data:
                parts.append(f"t={alert_data['raw_time'][idx]:.1f}s")
            if "name" in alert_data:
                parts.append(f'name="{alert_data["name"][idx]}"')
            lines.append(" ".join(parts))

        if len(alert_indices) > max_alerts:
            lines.append(
                f"... and {len(alert_indices) - max_alerts} more alerts (truncated)"
            )

        return "\n".join(lines)

    def _build_prompt(
        self,
        cluster_info: ClusterInfo,
        alert_data: dict,
        cluster_labels: np.ndarray,
        all_cluster_infos: list[ClusterInfo],
    ) -> tuple[str, str]:
        """Build the system and user prompts for LLM refinement.

        Args:
            cluster_info: Metadata for the cluster being refined
            alert_data: Dict with per-alert fields ('short', 'host', 'raw_time', etc.)
            cluster_labels: Cluster labels for the full dataset
            all_cluster_infos: All cluster infos for context

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Get alert indices for this cluster
        mask = cluster_labels == cluster_info.cluster_id
        alert_indices = np.where(mask)[0].tolist()

        # Build alert text representation
        alert_texts = self._format_alert_text(
            alert_indices, alert_data, self.config.max_alerts_per_prompt
        )

        # Build context about neighboring clusters
        neighbor_context = self._build_neighbor_context(
            cluster_info, all_cluster_infos
        )

        alert_types_str = ", ".join(cluster_info.alert_types[:10]) if cluster_info.alert_types else "unknown"
        hosts_str = ", ".join(cluster_info.hosts[:10]) if cluster_info.hosts else "unknown"

        user_prompt = REFINEMENT_USER_PROMPT.format(
            cluster_id=cluster_info.cluster_id,
            n_alerts=cluster_info.size,
            time_min=cluster_info.time_min,
            time_max=cluster_info.time_max,
            time_span=cluster_info.time_span,
            alert_types=alert_types_str,
            hosts=hosts_str,
            context=neighbor_context,
            alert_texts=alert_texts,
            max_alerts=self.config.max_alerts_per_prompt,
        )

        return REFINEMENT_SYSTEM_PROMPT, user_prompt

    def _build_neighbor_context(
        self,
        cluster_info: ClusterInfo,
        all_cluster_infos: list[ClusterInfo],
        max_neighbors: int = 3,
    ) -> str:
        """Build context string about temporally adjacent clusters."""
        # Find clusters that overlap in time
        neighbors = []
        for other in all_cluster_infos:
            if other.cluster_id == cluster_info.cluster_id:
                continue
            # Check temporal overlap
            if other.time_min <= cluster_info.time_max and other.time_max >= cluster_info.time_min:
                neighbors.append(other)
            if len(neighbors) >= max_neighbors:
                break

        if not neighbors:
            return "No temporally adjacent clusters found"

        lines = [f"{len(neighbors)} other alerts nearby"]
        for n in neighbors:
            types = ", ".join(n.alert_types[:3]) if n.alert_types else "unknown"
            lines.append(
                f"  - Cluster {n.cluster_id}: {n.size} alerts, "
                f"types=[{types}], time=[{n.time_min:.0f}s-{n.time_max:.0f}s]"
            )
        return "\n".join(lines)

    def _parse_llm_response(self, raw_response: str) -> dict:
        """Parse LLM response with self-healing fallbacks.

        Strategies:
        1. Direct JSON parse
        2. Extract from markdown code blocks
        3. Extract from inline code
        4. Return safe default

        Args:
            raw_response: Raw text output from LLM

        Returns:
            Parsed dict with verdict, certainty, reasoning, etc.
        """
        # Strategy 1: Direct JSON parse
        try:
            return json.loads(raw_response.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
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

        # Strategy 4: Return safe default (keep original cluster)
        logger.warning("Failed to parse LLM response, using safe default")
        return {
            "verdict": "keep",
            "certainty": 0.0,
            "reasoning": f"Failed to parse LLM response: {raw_response[:200]}",
            "sub_groups": [],
            "suggested_label": "",
        }

    def _validate_response(self, parsed: dict) -> dict:
        """Validate and normalize LLM response fields."""
        # Validate verdict
        verdict = parsed.get("verdict", "keep").lower().strip()
        if verdict not in ("keep", "split", "reassign"):
            logger.warning(f"Unknown verdict '{verdict}', defaulting to 'keep'")
            verdict = "keep"

        # Validate certainty
        try:
            certainty = float(parsed.get("certainty", 0.0))
            certainty = max(0.0, min(1.0, certainty))
        except (TypeError, ValueError):
            certainty = 0.0

        # Normalize sub_groups
        sub_groups = parsed.get("sub_groups", [])
        if not isinstance(sub_groups, list):
            sub_groups = []
        # Ensure each sub_group has alert_indices and theme
        validated_groups = []
        for sg in sub_groups:
            if not isinstance(sg, dict):
                continue
            indices = sg.get("alert_indices", [])
            if not isinstance(indices, list):
                continue
            # Reject empty sub-groups (no alert indices is meaningless)
            if len(indices) == 0:
                continue
            # Ensure all indices are integers
            try:
                indices = [int(i) for i in indices]
            except (TypeError, ValueError):
                continue
            validated_groups.append(
                {
                    "alert_indices": indices,
                    "theme": str(sg.get("theme", "unknown")),
                }
            )

        # Normalize suggested label
        suggested_label = str(parsed.get("suggested_label", ""))

        return {
            "verdict": verdict,
            "certainty": certainty,
            "reasoning": str(parsed.get("reasoning", "")),
            "sub_groups": validated_groups,
            "suggested_label": suggested_label,
        }

    def refine_cluster(
        self,
        cluster_info: ClusterInfo,
        alert_data: dict,
        cluster_labels: np.ndarray,
        all_cluster_infos: list[ClusterInfo],
        max_retries: int | None = None,
    ) -> RefinementResult:
        """Refine a single cluster using LLM.

        Args:
            cluster_info: Metadata for the cluster to refine
            alert_data: Dict with per-alert fields
            cluster_labels: Full cluster label array
            all_cluster_infos: All cluster infos for context
            max_retries: Override config max_retries

        Returns:
            RefinementResult with verdict and optional sub-groups
        """
        max_retries = max_retries if max_retries is not None else self.config.max_retries

        if self.total_cost >= self.config.max_cost:
            logger.info(
                f"Budget cap ${self.config.max_cost:.2f} exceeded "
                f"(${self.total_cost:.4f} spent). Skipping LLM call for cluster {cluster_info.cluster_id}."
            )
            return RefinementResult(
                cluster_id=cluster_info.cluster_id,
                verdict="keep",
                certainty=0.0,
                reasoning=f"Budget cap ${self.config.max_cost:.2f} exceeded, skipping LLM refinement",
                parse_success=True,
            )
        if self.config.max_calls is not None and self.total_calls >= self.config.max_calls:
            logger.info(
                f"Max calls ({self.config.max_calls}) reached "
                f"({self.total_calls} made). Skipping LLM call for cluster {cluster_info.cluster_id}."
            )
            return RefinementResult(
                cluster_id=cluster_info.cluster_id,
                verdict="keep",
                certainty=0.0,
                reasoning=f"Max calls ({self.config.max_calls}) reached, skipping LLM refinement",
                parse_success=True,
            )

        model = self._select_model(cluster_info.routing_tier)
        system_prompt, user_prompt = self._build_prompt(
            cluster_info, alert_data, cluster_labels, all_cluster_infos
        )

        last_error = None
        for attempt in range(max_retries + 1):
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
                parsed = self._parse_llm_response(raw_response)
                validated = self._validate_response(parsed)

                # Track usage
                self.total_calls += 1
                if hasattr(response, "usage") and response.usage:
                    self.total_tokens += response.usage.total_tokens
                    # Estimate cost based on provider/model
                    if "deepseek" in model:
                        # DeepSeek pricing (approximate, as of 2026-06)
                        # deepseek-v4-flash: ~$0.10/1M input, ~$0.40/1M output
                        # deepseek-v4-pro: ~$1.00/1M input, ~$4.00/1M output
                        if "v4-pro" in model:
                            self.total_cost += (
                                response.usage.prompt_tokens * 1.00 / 1_000_000
                                + response.usage.completion_tokens * 4.00 / 1_000_000
                            )
                        else:  # v4-flash or other
                            self.total_cost += (
                                response.usage.prompt_tokens * 0.10 / 1_000_000
                                + response.usage.completion_tokens * 0.40 / 1_000_000
                            )
                    elif "gpt-4o-mini" in model:
                        self.total_cost += (
                            response.usage.prompt_tokens * 0.15 / 1_000_000
                            + response.usage.completion_tokens * 0.60 / 1_000_000
                        )
                    elif "gpt-4o" in model:
                        self.total_cost += (
                            response.usage.prompt_tokens * 2.50 / 1_000_000
                            + response.usage.completion_tokens * 10.00 / 1_000_000
                        )

                return RefinementResult(
                    cluster_id=cluster_info.cluster_id,
                    verdict=validated["verdict"],
                    certainty=validated["certainty"],
                    reasoning=validated["reasoning"],
                    sub_groups=validated["sub_groups"],
                    suggested_label=validated["suggested_label"],
                    llm_model_used=model,
                    llm_call_cost=self.total_cost,
                    parse_success=True,
                    raw_response=raw_response,
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call attempt {attempt + 1}/{max_retries + 1} failed: {e}"
                )
                self.parse_failures += 1
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))  # Exponential-ish backoff

        # All retries failed — return safe default
        logger.error(f"All LLM retries failed for cluster {cluster_info.cluster_id}: {last_error}")
        return RefinementResult(
            cluster_id=cluster_info.cluster_id,
            verdict="keep",
            certainty=0.0,
            reasoning=f"LLM call failed after {max_retries + 1} attempts: {last_error}",
            sub_groups=[],
            suggested_label="",
            llm_model_used=model,
            parse_success=False,
            raw_response="",
        )

    def apply_refinement(
        self,
        labels: np.ndarray,
        refinement: RefinementResult,
        cluster_info: ClusterInfo,
    ) -> np.ndarray:
        """Apply an LLM refinement result to the cluster labels.

        If verdict is "keep", no change. If verdict is "split", creates
        new cluster IDs for each sub-group.

        Args:
            labels: Current cluster labels, shape (N,)
            refinement: LLM refinement result
            cluster_info: Cluster info for this cluster

        Returns:
            Updated labels array (may have new cluster IDs)
        """
        if refinement.verdict != "split" or not refinement.sub_groups:
            return labels

        # Get the maximum cluster ID to avoid collisions
        next_label = int(np.max(labels)) + 1

        # Get indices of alerts in this cluster
        mask = labels == cluster_info.cluster_id
        cluster_indices = np.where(mask)[0]

        # Apply each sub-group
        new_labels = labels.copy()
        for i, sg in enumerate(refinement.sub_groups):
            if i == 0:
                # First sub-group keeps the original cluster ID
                continue
            # Convert local alert_indices to global indices
            global_indices = cluster_indices[sg["alert_indices"]]
            new_labels[global_indices] = next_label
            next_label += 1

        return new_labels

    def get_stats(self) -> dict:
        """Get usage statistics for LLM calls."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "parse_failures": self.parse_failures,
            "avg_cost_per_call": self.total_cost / max(self.total_calls, 1),
        }