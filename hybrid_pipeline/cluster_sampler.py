"""Cluster sampling and feature extraction module.

Extracts per-cluster metadata from AlertBERT output and flags clusters that
warrant LLM refinement based on size, variance, and boundary characteristics.
Uses a union policy: ANY criterion flags a cluster.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from hybrid_pipeline.config import SamplerConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Per-cluster metadata produced by the cluster sampler."""

    cluster_id: int
    size: int
    time_span: float
    time_min: float
    time_max: float
    alert_types: list[str]
    hosts: list[str]

    # Flagging criteria
    is_small: bool
    is_high_variance: bool
    is_boundary_heavy: bool
    is_long_span: bool

    # Computed score and routing
    quality_score: float
    is_flagged: bool
    routing_tier: str  # "skip", "cheap", "expensive"


class ClusterSampler:
    """Scores AlertBERT clusters for quality and routes them to appropriate LLM tiers.

    A cluster is flagged as flagged if ANY of the following criteria are met:
    1. Cluster size < min_cluster_size
    2. Intra-cluster embedding variance > max_intra_cluster_variance
    3. Boundary fraction > boundary_fraction_threshold (silhouette-inspired detection)
    4. Time span > max_time_span
    """

    def __init__(self, config: SamplerConfig | None = None):
        self.config = config or SamplerConfig()

    def score_clusters(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
        pre_cluster_ids: np.ndarray,
        alert_types: np.ndarray | None = None,
        hosts: np.ndarray | None = None,
        raw_time: np.ndarray | None = None,
    ) -> list[ClusterInfo]:
        """Score all clusters and determine routing.

        Args:
            labels: Cluster labels from AlertBERT, shape (N,)
            embeddings: PCA+time embeddings from AlertBERT, shape (N, dim_reduction+1)
                Last column is raw_time. Cosine distance is computed on first dim_reduction columns.
            pre_cluster_ids: Time-delta pre-cluster IDs, shape (N,)
            alert_types: Alert type strings (e.g. "short" field), shape (N,), optional
            hosts: Host name strings, shape (N,), optional
            raw_time: Raw timestamps, shape (N,), optional (extracted from embeddings if not given)

        Returns:
            List of ClusterInfo for each unique cluster.
        """
        unique_labels = np.unique(labels)
        cluster_infos = []

        # Extract embedding dimensions (all columns except last which is time)
        emb_dims = embeddings[:, :-1]  # PCA dimensions only
        if raw_time is None:
            raw_time = embeddings[:, -1]

        # === Precompute centroids for all clusters (vectorized) ===
        # Single N×C alert-to-centroid distance matrix replaces per-pair loops
        all_centroids = np.zeros((len(unique_labels), emb_dims.shape[1]))
        label_to_idx = {int(label): idx for idx, label in enumerate(unique_labels)}
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            all_centroids[idx] = np.mean(emb_dims[mask], axis=0)

        # Precompute N × C alert-to-centroid distance matrix (single batch call)
        # Always computed — needed for intra-variance even with 1 cluster
        alert_centroid_dists = cosine_distances(emb_dims, all_centroids)

        # Build vectorized cluster membership index
        label_indices = np.array([label_to_idx[int(lab)] for lab in labels])

        # === Silhouette-inspired boundary detection (self-calibrating) ===
        # For each alert, compute distance to own cluster centroid vs nearest OTHER centroid.
        # An alert is "boundary" if it's closer to another cluster than its own cluster's
        # typical spread. This self-calibrates for local density in the embedding space.
        #
        # Step 1: Per-cluster intra-cluster mean distance (typical spread)
        n_clusters = len(unique_labels)
        intra_mean = np.zeros(n_clusters)
        for idx in range(n_clusters):
            cluster_mask = label_indices == idx
            if np.sum(cluster_mask) > 0:
                intra_mean[idx] = float(np.mean(alert_centroid_dists[cluster_mask, idx]))
            else:
                intra_mean[idx] = 0.0

        # Step 2: For each alert, minimum distance to any OTHER centroid
        # Mask own-cluster distance as inf so min() picks the nearest OTHER centroid
        masked_dists = alert_centroid_dists.copy()
        masked_dists[np.arange(len(labels)), label_indices] = np.inf
        min_other_dist = masked_dists.min(axis=1)  # shape [N]

        # Step 3: Per-alert boundary flag — is alert closer to another centroid
        # than its own cluster's typical spread (scaled by alpha)?
        # alpha < 1.0 makes boundary detection stricter (requires another centroid
        # to be even closer relative to intra-cluster spread)
        intra_mean_per_alert = intra_mean[label_indices]  # shape [N]
        # Handle single-alert clusters where intra_mean=0 (always boundary)
        is_singleton = intra_mean_per_alert == 0.0
        alpha = self.config.boundary_alpha
        is_boundary_alert = np.where(
            is_singleton,
            False,
            min_other_dist < alpha * intra_mean_per_alert,
        )

        # Step 4: Per-cluster boundary fraction
        boundary_fractions = np.zeros(n_clusters)
        for idx in range(n_clusters):
            cluster_mask = label_indices == idx
            if np.sum(cluster_mask) > 0:
                boundary_fractions[idx] = float(np.mean(is_boundary_alert[cluster_mask]))

        for idx, cluster_id in enumerate(unique_labels):
            mask = labels == cluster_id
            cluster_time = raw_time[mask]

            # Basic cluster stats
            n_alerts = int(np.sum(mask))
            time_min = float(np.min(cluster_time))
            time_max = float(np.max(cluster_time))
            time_span = time_max - time_min

            # Alert types and hosts
            if alert_types is not None:
                types_list = list(set(alert_types[mask]))
            else:
                types_list = []
            if hosts is not None:
                hosts_list = list(set(hosts[mask]))
            else:
                hosts_list = []

            # === Criterion 1: Small cluster ===
            is_small = n_alerts < self.config.min_cluster_size

            # === Criterion 2: Intra-cluster variance ===
            if n_alerts > 1:
                intra_var = float(np.mean(alert_centroid_dists[mask, idx]))
            else:
                intra_var = 0.0
            is_high_variance = intra_var > self.config.max_intra_cluster_variance

            # === Criterion 3: Boundary alerts (silhouette-inspired) ===
            # Uses precomputed boundary_fractions from silhouette analysis
            # An alert is "boundary" if another centroid is closer than alpha * own_cluster_spread
            # boundary_fraction = fraction of alerts in this cluster that are boundary
            boundary_fraction = boundary_fractions[idx]
            is_boundary_heavy = boundary_fraction > self.config.boundary_fraction_threshold

            # === Criterion 4: Long time span ===
            is_long_span = time_span > self.config.max_time_span

            # === Compute quality score ===
            # Higher score = more confident. Score is penalized by each criterion.
            score = 1.0
            if is_small:
                score *= 0.5  # Small clusters are suspicious
            if is_high_variance:
                score *= 0.4  # High variance is more suspicious
            if is_boundary_heavy:
                score *= 0.6  # Boundary alerts suggest ambiguity
            if is_long_span:
                score *= 0.8  # Long time span is mildly suspicious

            is_flagged = (
                is_small or is_high_variance or is_boundary_heavy or is_long_span
            )

            # === Routing tier (double-threshold policy) ===
            if score >= self.config.tau_high:
                routing_tier = "skip"
            elif score >= self.config.tau_low:
                routing_tier = "cheap"
            else:
                routing_tier = "expensive"

            cluster_infos.append(
                ClusterInfo(
                    cluster_id=int(cluster_id),
                    size=n_alerts,
                    time_span=time_span,
                    time_min=time_min,
                    time_max=time_max,
                    alert_types=types_list,
                    hosts=hosts_list,
                    is_small=is_small,
                    is_high_variance=is_high_variance,
                    is_boundary_heavy=is_boundary_heavy,
                    is_long_span=is_long_span,
                    quality_score=score,
                    is_flagged=is_flagged,
                    routing_tier=routing_tier,
                )
            )

        return cluster_infos

    def get_flagged_clusters(
        self, cluster_infos: list[ClusterInfo]
    ) -> list[ClusterInfo]:
        """Return only clusters flagged as flagged."""
        return [c for c in cluster_infos if c.is_flagged]

    def get_clusters_by_tier(
        self, cluster_infos: list[ClusterInfo], tier: str
    ) -> list[ClusterInfo]:
        """Return clusters routed to a specific tier ('skip', 'cheap', 'expensive')."""
        return [c for c in cluster_infos if c.routing_tier == tier]

    def summarize(self, cluster_infos: list[ClusterInfo]) -> dict:
        """Produce a summary of cluster sampling and flagging results."""
        total = len(cluster_infos)
        low_conf = sum(1 for c in cluster_infos if c.is_flagged)
        skip = sum(1 for c in cluster_infos if c.routing_tier == "skip")
        cheap = sum(1 for c in cluster_infos if c.routing_tier == "cheap")
        expensive = sum(1 for c in cluster_infos if c.routing_tier == "expensive")
        small = sum(1 for c in cluster_infos if c.is_small)
        high_var = sum(1 for c in cluster_infos if c.is_high_variance)
        boundary = sum(1 for c in cluster_infos if c.is_boundary_heavy)
        long_span = sum(1 for c in cluster_infos if c.is_long_span)

        return {
            "total_clusters": total,
            "flagged_clusters": low_conf,
            "flagged_pct": low_conf / max(total, 1) * 100,
            "routing": {"skip": skip, "cheap": cheap, "expensive": expensive},
            "criteria_triggered": {
                "small": small,
                "high_variance": high_var,
                "boundary_heavy": boundary,
                "long_span": long_span,
            },
            "avg_quality": np.mean([c.quality_score for c in cluster_infos]),
        }