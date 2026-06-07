"""Configuration for the hybrid AlertBERT + LLM refinement pipeline."""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SamplerConfig:
    """Thresholds for flagged cluster detection.

    A cluster is flagged as flagged if ANY criterion is met (union policy).

    Thresholds tuned on AIT-ADS-A 'original' configuration (7702 clusters):
    - min_cluster_size=3 flags ~15.5% of clusters (singletons + pairs)
    - max_intra_cluster_variance=0.08 flags ~12-15% (heterogeneous clusters)
    - boundary_alpha=0.5 controls silhouette-inspired detection sensitivity
    - boundary_fraction_threshold=0.5 flags clusters with >50% boundary alerts
    - max_time_span=5.0s flags only true outliers (P99=3.3s, max=12.5s)
    - tau_high=0.70 and tau_low=0.45 produce ~55-65% skip, ~20-30% cheap, ~10-20% expensive
    """

    # Clusters with fewer alerts than this are flagged
    min_cluster_size: int = 3

    # Max intra-cluster cosine variance (higher = more heterogeneous = less confident)
    # Embedding vectors are PCA-reduced; variance > this threshold triggers flagged
    max_intra_cluster_variance: float = 0.08

    # Boundary detection: silhouette-inspired approach
    # An alert is "boundary" if its distance to the nearest OTHER centroid < alpha * own_cluster_mean_distance
    # alpha=1.0: flag if any other centroid is closer than own cluster's typical spread
    # alpha<1.0: stricter, requires other centroid to be even closer relatively
    boundary_alpha: float = 0.5
    # A cluster is flagged if > boundary_fraction_threshold of its alerts are boundary
    boundary_fraction_threshold: float = 0.5

    # Clusters spanning a time range longer than this (in seconds) are flagged
    max_time_span: float = 5.0  # Catches true outliers (P99=3.3s)

    # Quality score thresholds for routing (Muric & Minton double-threshold policy)
    # Clusters with score >= tau_high skip LLM entirely
    tau_high: float = 0.70
    # Clusters with score between tau_low and tau_high go to cheap LLM
    tau_low: float = 0.45
    # Clusters with score < tau_low go to expensive LLM


@dataclass
class LLMConfig:
    """LLM provider configuration for cluster refinement.

    Default configuration uses DeepSeek API (OpenAI-compatible).
    Set DEEPSEEK_API_KEY via environment variable or deepseek-apikey.txt.
    """

    # Provider: "deepseek" (default), "openai", or "ollama"
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek"))

    # API key: reads from env var, then from deepseek-apikey.txt
    api_key: str = field(default_factory=lambda: _load_api_key())

    # Base URL: DeepSeek API is OpenAI-compatible
    base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    )

    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 1024

    # Max number of alerts per cluster sent to LLM (truncated if larger)
    max_alerts_per_prompt: int = 15

    # Max retries on parse failure
    max_retries: int = 2

    # Model tier routing (DeepSeek models)
    # Medium quality → cheap/fast model
    cheap_model: str = field(
        default_factory=lambda: os.getenv("LLM_CHEAP_MODEL", "deepseek-v4-flash")
    )
    # Low quality → expensive/powerful model
    expensive_model: str = field(
        default_factory=lambda: os.getenv("LLM_EXPENSIVE_MODEL", "deepseek-v4-pro")
    )

    # Default model (used for non-tiered calls)
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-v4-flash")
    )

    # Budget and rate limiting for batch experiments
    # Maximum total cost ($) before stopping LLM calls (None or inf = unlimited)
    max_cost: float = float("inf")
    # Maximum number of LLM calls across all scenarios (None = unlimited)
    max_calls: int | None = None


def _load_api_key() -> str:
    """Load API key from environment variable or deepseek-apikey.txt file."""
    # Try environment variable first
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if key:
        return key

    # Try loading from file
    key_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deepseek-apikey.txt")
    if os.path.exists(key_file):
        with open(key_file) as f:
            return f.read().strip()

    return ""


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    # AlertBERT model settings
    model_id: str = "mlm_1l_4h_16d_original_default_params_60k"
    aitads_a_config: str = "original"
    saved_models_path: str = "saved_models"

    # Clustering hyperparameters
    delta: float = 2.0
    theta: float = 2.0
    dim_reduction: int = 2

    # Sampler configuration for cluster feature extraction and routing
    sampler: SamplerConfig = field(default_factory=SamplerConfig)

    # LLM refinement (split/keep verdicts on individual clusters)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Whether to actually call LLM (set False to simulate/dry-run)
    llm_enabled: bool = True

    # Pipeline mode: "full" (LLM refine), "llm_only", or "baseline"
    mode: str = "full"

    # GPU device
    device: str = "cuda:0"

    # Quick mode: sample every 3rd flagged cluster for fast iteration
    quick: bool = False

    # Output directory for results
    output_dir: str = "hybrid_pipeline/results"