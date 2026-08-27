"""
constants.py
Project-wide constants.

Runtime parameters live in configs/pipeline_config.yaml, not here. This module only
holds path anchors and the canonical artefact filenames the stages exchange, so that
renaming an output happens in exactly one place.
"""
from src.utils.paths import PROJECT_ROOT, resolve_dir

DEFAULT_RAW_DIR = resolve_dir("data/raw")
DEFAULT_PROCESSED_DIR = resolve_dir("data/processed")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline_config.yaml"

# Artefact filenames passed between pipeline stages.
CLEAN_STACK_FILE = "fossilfly_clean_stack.npz"
METADATA_FILE = "metadata.json"
SUMMARY_STATS_FILE = "01_summary_statistics.csv"
PROCESSED_STACK_FILE = "fossilfly_processed_stack.npz"
PCA_FILE = "02b_pca_components.npz"
NMF_FILE = "02b_nmf_components.npz"
UMAP_FILE = "02b_umap_embedding.npz"
FOREGROUND_MASK_FILE = "02b_foreground_mask.npy"
SEGMENTS_FILE = "fossilfly_segments.npz"
SEGMENTATION_METRICS_FILE = "segmentation_metrics.csv"
STATISTICAL_RESULTS_FILE = "04_statistical_results.csv"
COLOCALIZATION_FILE = "04_colocalization.csv"
POSTHOC_FILE = "04_dunn_posthoc.csv"

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_RAW_DIR",
    "DEFAULT_PROCESSED_DIR",
    "DEFAULT_CONFIG_PATH",
    "CLEAN_STACK_FILE",
    "METADATA_FILE",
    "SUMMARY_STATS_FILE",
    "PROCESSED_STACK_FILE",
    "PCA_FILE",
    "NMF_FILE",
    "UMAP_FILE",
    "FOREGROUND_MASK_FILE",
    "SEGMENTS_FILE",
    "SEGMENTATION_METRICS_FILE",
    "STATISTICAL_RESULTS_FILE",
    "COLOCALIZATION_FILE",
    "POSTHOC_FILE",
]
