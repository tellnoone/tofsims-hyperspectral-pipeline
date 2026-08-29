"""
pipeline.py
Stage orchestration.

Six stages, each independently runnable because each reads its inputs from disk and
writes its outputs there:

    load      raw .bmp files          -> clean stack, metadata, summary stats
    overview  clean stack             -> channel overview and correlation figures
    decompose clean stack             -> PCA / NMF / UMAP arrays and figures
    segment   decomposition arrays    -> segment labels, metrics and figures
    stats     labels + clean stack    -> significance tables and figures
    figures   everything above        -> publication-style master figures

Running `--stage stats` on its own therefore works, as long as the earlier stages
have been run at least once. Nothing in any stage is specific to one dataset: the
channels, the number of endmembers and the segment count are all derived from the
data or read from the config.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.features import decomposition as decomp
from src.features.selection import (
    ChannelSelection,
    build_cube,
    resample_to,
    select_analysis_channels,
)
from src.preprocessing.load_images import load_images_from_config
from src.preprocessing.metadata import attach_shapes, build_metadata_from_config
from src.preprocessing.stack_io import load_clean_stack, save_clean_stack
from src.segmentation import cluster
from src.stats import validation
from src.utils.constants import SEGMENTS_FILE
from src.viz import decomposition_plots, overview, report, segmentation_plots, stats_plots
from src.viz.style import segment_names

STAGES = ["load", "overview", "decompose", "segment", "stats", "figures"]


class StageResult(dict):
    """Plain dict of artefact name -> path, with a readable summary."""

    def summary(self) -> str:
        return "\n".join(f"  {name}: {Path(path).name}" for name, path in self.items())


def _log(message: str) -> None:
    print(message, flush=True)


def _merge(*parts: Dict[str, Any]) -> StageResult:
    """Combine artefact maps, failing loudly if two stages claim the same name."""
    merged = StageResult()
    for part in parts:
        for name, path in part.items():
            if name in merged:
                raise ValueError(f"Duplicate artefact key '{name}' in stage results")
            merged[name] = path
    return merged


def _load_selection(config: Config):
    """Load the clean stack and re-derive the channel selection."""
    images, metadata = load_clean_stack(config.processed_dir)
    selection = select_analysis_channels(images, metadata, config)
    return images, metadata, selection


# ------------------------------------------------------------------ stage: load
def stage_load(config: Config) -> StageResult:
    """Parse the raw exports into a clean, keyed image stack."""
    _log(f"[load] reading {config.get('dataset.file_pattern', '*.bmp')} from {config.raw_dir}")
    images = load_images_from_config(config)
    _log(f"[load] parsed {len(images)} image(s)")

    metadata = build_metadata_from_config(list(images), config)
    metadata = attach_shapes(metadata, images)

    written = save_clean_stack(images, metadata, config.processed_dir)

    figures = StageResult()
    dpi = config.get("figures.dpi", 200)
    figures["raw_overview"] = overview.plot_raw_overview(
        images, config.figure_path("01_raw_overview.png"), dpi=dpi,
        cmap=config.get("figures.intensity_cmap", "hot"),
    )
    figures["histograms"] = overview.plot_intensity_histograms(
        images, config.figure_path("01_intensity_histograms.png"), dpi=dpi
    )
    figures["preprocessing"] = overview.plot_preprocessing_comparison(
        images[list(images)[0]], config.figure_path("01_preprocessing_comparison.png"), dpi=dpi
    )

    result = _merge(written, figures)
    _log(f"[load] wrote {len(result)} artefact(s)")
    return result


# -------------------------------------------------------------- stage: overview
def stage_overview(config: Config) -> StageResult:
    """Channel-level overview: composites, correlation structure, polarity comparison."""
    images, metadata, selection = _load_selection(config)
    _log(f"[overview] {selection.describe()}")

    cube = build_cube(images, selection)
    dpi = config.get("figures.dpi", 200)
    result = StageResult()

    result["channel_series"] = overview.plot_channel_series(
        images, selection, config.figure_path("02a_channel_series.png"), config, dpi=dpi
    )
    matrix = overview.channel_correlation_matrix(cube)
    result["correlation_matrix"] = overview.plot_correlation_matrix(
        matrix, selection.labels, config.figure_path("02a_correlation_matrix.png"), dpi=dpi
    )
    result["rgb_composites"] = overview.plot_rgb_composites(
        images, selection, config.figure_path("02a_rgb_composites.png"), config, dpi=dpi
    )
    if selection.secondary_keys:
        result["polarity_comparison"] = overview.plot_polarity_comparison(
            images, selection, config.figure_path("02a_polarity_comparison.png"), dpi=dpi
        )
        result["mean_comparison"] = overview.plot_mean_comparison(
            images, selection, config.figure_path("02a_polarity_mean_comparison.png"), dpi=dpi
        )

    _log(f"[overview] wrote {len(result)} figure(s)")
    return result


# ------------------------------------------------------------- stage: decompose
def stage_decompose(config: Config) -> StageResult:
    """PCA, NMF and UMAP on the analysis cube."""
    images, metadata, selection = _load_selection(config)
    _log(f"[decompose] {selection.describe()}")

    cube = build_cube(images, selection)
    prepared = decomp.prepare_pixels(cube, eps=float(config.get("preprocessing.eps", 1e-8)))
    _log(f"[decompose] {prepared.n_foreground:,} foreground pixels of {len(prepared.fg_mask):,}")

    random_state = config.random_seed

    n_pcs = None if config.is_auto("decomposition.pca.n_components") else int(
        config.get("decomposition.pca.n_components")
    )
    pca_result = decomp.run_pca(prepared, n_components=n_pcs, random_state=random_state)
    _log(
        f"[decompose] PCA: {pca_result['n_components']} components, "
        f"{pca_result['explained_variance'][:3].sum() * 100:.1f}% variance in first 3"
    )

    sweep = decomp.run_nmf_sweep(
        prepared,
        sweep=config.get("decomposition.nmf.sweep", [2, 3, 4, 5]),
        max_iter=int(config.get("decomposition.nmf.max_iter", 1000)),
        init=config.get("decomposition.nmf.init", "nndsvda"),
        random_state=random_state,
    )
    if config.is_auto("decomposition.nmf.n_components"):
        chosen_k = decomp.select_nmf_rank(sweep)
        _log(f"[decompose] NMF rank selected from error elbow: k = {chosen_k}")
    else:
        chosen_k = int(config.get("decomposition.nmf.n_components"))
        _log(f"[decompose] NMF rank from config: k = {chosen_k}")

    nmf_result = decomp.run_nmf(
        prepared,
        n_components=chosen_k,
        max_iter=int(config.get("decomposition.nmf.max_iter", 1000)),
        init=config.get("decomposition.nmf.init", "nndsvda"),
        random_state=random_state,
        clip_percentile=float(config.get("decomposition.nmf.abundance_clip_percentile", 99.5)),
        sweep_results=sweep,
    )

    umap_result = None
    if config.get("decomposition.umap.enabled", True):
        try:
            umap_result = decomp.run_umap(
                prepared,
                n_neighbors=int(config.get("decomposition.umap.n_neighbors", 30)),
                min_dist=float(config.get("decomposition.umap.min_dist", 0.1)),
                n_components=int(config.get("decomposition.umap.n_components", 2)),
                metric=config.get("decomposition.umap.metric", "euclidean"),
                random_state=random_state,
            )
            _log(f"[decompose] UMAP embedded {umap_result['embedding'].shape[0]:,} pixels")
        except ImportError:
            _log("[decompose] umap-learn not installed; skipping UMAP")

    written = decomp.save_decomposition(
        config.processed_dir, prepared, pca_result, nmf_result, umap_result,
        selection.labels, selection.keys,
    )

    dpi = config.get("figures.dpi", 200)
    figures = StageResult()
    figures["pca_scree"] = decomposition_plots.plot_pca_scree(
        pca_result, selection.labels, config.figure_path("02b_pca_scree.png"), dpi=dpi
    )
    figures["pca_maps"] = decomposition_plots.plot_pca_component_maps(
        pca_result, prepared.fg_mask, config.figure_path("02b_pca_component_maps.png"), dpi=dpi
    )
    figures["pca_scatter"] = decomposition_plots.plot_pca_scatter(
        pca_result, prepared, config.figure_path("02b_pca_scatter.png"),
        subsample=int(config.get("decomposition.scatter_subsample", 20000)),
        random_state=random_state, dpi=dpi,
    )
    figures["nmf_selection"] = decomposition_plots.plot_nmf_model_selection(
        sweep, chosen_k, config.figure_path("02b_nmf_model_selection.png"), dpi=dpi
    )
    figures["nmf_endmembers"] = decomposition_plots.plot_nmf_endmembers(
        nmf_result, selection.labels, config.figure_path("02b_nmf_endmembers.png"), dpi=dpi
    )
    figures["nmf_abundance"] = decomposition_plots.plot_nmf_abundance_maps(
        nmf_result, config.figure_path("02b_nmf_abundance_maps.png"), dpi=dpi
    )
    figures["nmf_composite"] = decomposition_plots.plot_nmf_rgb_composite(
        nmf_result, config.figure_path("02b_nmf_rgb_composite.png"), dpi=dpi
    )
    if umap_result is not None:
        figures["umap_scatter"] = decomposition_plots.plot_umap_scatter(
            umap_result, prepared, nmf_result, config.figure_path("02b_umap_scatter.png"), dpi=dpi
        )
        figures["umap_spatial"] = decomposition_plots.plot_umap_spatial_maps(
            umap_result, config.figure_path("02b_umap_spatial_maps.png"), dpi=dpi
        )
    figures["method_comparison"] = decomposition_plots.plot_method_comparison(
        pca_result, nmf_result, umap_result, selection.labels,
        config.figure_path("02b_method_comparison.png"), dpi=dpi,
    )

    result = _merge(written, figures)
    _log(f"[decompose] wrote {len(result)} artefact(s)")
    return result


# --------------------------------------------------------------- stage: segment
def stage_segment(config: Config) -> StageResult:
    """Partition the foreground into chemically distinct regions and cross-check."""
    artefacts = decomp.load_decomposition(config.processed_dir)
    fg_mask = artefacts["fg_mask"].astype(bool)
    fg_idx = np.where(fg_mask)[0]

    abundance_maps = artefacts["nmf"]["abundance_maps"]
    score_maps = artefacts["pca"]["score_maps"]
    shape = abundance_maps.shape[:2]
    channel_labels = [str(x) for x in artefacts["nmf"]["channel_labels"]]

    X_nmf = cluster.nmf_feature_space(abundance_maps, fg_idx)
    X_pca = cluster.pca_feature_space(score_maps, fg_idx, n_components=3)

    n_clusters = (
        int(abundance_maps.shape[-1])
        if config.is_auto("segmentation.n_clusters")
        else int(config.get("segmentation.n_clusters"))
    )
    oversegment_k = (
        n_clusters + 1
        if config.is_auto("segmentation.oversegment_k")
        else int(config.get("segmentation.oversegment_k"))
    )
    random_state = config.random_seed
    _log(f"[segment] {len(fg_idx):,} foreground pixels, {n_clusters} segments")

    labels_dominant = cluster.dominant_endmember_labels(X_nmf)
    labels_kmeans, _ = cluster.run_kmeans(
        X_pca, n_clusters, n_init=int(config.get("segmentation.kmeans.n_init", 10)),
        random_state=random_state,
    )
    labels_kmeans_over, _ = cluster.run_kmeans(
        X_pca, oversegment_k, n_init=int(config.get("segmentation.kmeans.n_init", 10)),
        random_state=random_state,
    )
    gmm = cluster.run_gmm(
        X_pca, n_clusters,
        covariance_type=config.get("segmentation.gmm.covariance_type", "full"),
        random_state=random_state,
    )

    label_sets: Dict[str, np.ndarray] = {
        "dominant": labels_dominant,
        "kmeans_pca": labels_kmeans,
        "kmeans_pca_over": labels_kmeans_over,
        "gmm": gmm["labels"],
    }

    metrics_rows = [
        {
            "method": "Dominant NMF endmember",
            "silhouette": cluster.safe_silhouette(X_nmf, labels_dominant),
            "clusters": len(set(labels_dominant)),
            "noise_pct": 0.0,
        },
        {
            "method": f"K-Means k={n_clusters} (PCA)",
            "silhouette": cluster.safe_silhouette(X_pca, labels_kmeans),
            "clusters": n_clusters,
            "noise_pct": 0.0,
        },
        {
            "method": f"K-Means k={oversegment_k} (PCA)",
            "silhouette": cluster.safe_silhouette(X_pca, labels_kmeans_over),
            "clusters": oversegment_k,
            "noise_pct": 0.0,
        },
        {
            "method": f"GMM k={n_clusters} (PCA)",
            "silhouette": cluster.safe_silhouette(X_pca, gmm["labels"]),
            "clusters": n_clusters,
            "noise_pct": 0.0,
        },
    ]

    hdbscan_sweep: Optional[pd.DataFrame] = None
    umap_available = "umap" in artefacts
    if config.get("segmentation.hdbscan.enabled", True) and umap_available:
        X_umap = artefacts["umap"]["embedding"]
        min_cluster_size = cluster.resolve_min_cluster_size(
            config.get("segmentation.hdbscan.min_cluster_size", "auto"), len(fg_idx)
        )
        _log(f"[segment] HDBSCAN min_cluster_size = {min_cluster_size}")
        hdbscan = cluster.run_hdbscan(
            X_umap,
            min_cluster_size=min_cluster_size,
            min_samples=int(config.get("segmentation.hdbscan.min_samples", 20)),
            metric=config.get("segmentation.hdbscan.metric", "euclidean"),
            cluster_selection_method=config.get(
                "segmentation.hdbscan.cluster_selection_method", "eom"
            ),
        )
        label_sets["hdbscan"] = hdbscan["labels"]
        metrics_rows.append(
            {
                "method": "HDBSCAN (UMAP)",
                "silhouette": cluster.safe_silhouette(X_umap, hdbscan["labels"]),
                "clusters": hdbscan["n_clusters"],
                "noise_pct": round(hdbscan["noise_pct"], 2),
            }
        )
        hdbscan_sweep = cluster.hdbscan_sensitivity(
            X_umap, labels_dominant,
            config.get("segmentation.hdbscan.sweep_min_cluster_size", [100, 200, 500, 1000]),
            config.get("segmentation.hdbscan.sweep_min_samples", [5, 10, 20]),
        )

    # Scored per pair on the pixels both methods assigned, so that comparisons not
    # involving HDBSCAN stay independent of its (non-reproducible) noise set.
    agreement = cluster.agreement_metrics(label_sets)
    purity = cluster.segment_purity(labels_dominant, labels_kmeans)

    label_images = {
        name: cluster.labels_to_image(labels, fg_idx, shape)
        for name, labels in label_sets.items()
    }
    entropy_image = cluster.values_to_image(gmm["entropy"], fg_idx, shape)

    written = cluster.save_segments(
        config.processed_dir,
        label_sets,
        label_images,
        {
            "gmm_probabilities": gmm["probabilities"],
            "gmm_entropy": gmm["entropy"],
            "foreground_mask": fg_mask,
        },
        pd.DataFrame(metrics_rows),
        channel_labels,
    )

    agreement_path = config.processed_dir / "03_cross_method_agreement.csv"
    agreement.to_csv(agreement_path, index=False)
    written["agreement_csv"] = agreement_path

    dpi = config.get("figures.dpi", 250)
    titles = {
        "dominant": "Dominant NMF endmember",
        "kmeans_pca": f"K-Means k={n_clusters} (PCA)",
        "kmeans_pca_over": f"K-Means k={oversegment_k} (PCA)",
        "gmm": f"GMM k={n_clusters} (PCA)",
        "hdbscan": "HDBSCAN (UMAP)",
    }
    figures = StageResult()
    figures["comparison"] = segmentation_plots.plot_segmentation_comparison(
        {titles[name]: image for name, image in label_images.items()},
        config.figure_path("03_segmentation_comparison.png"),
        entropy_image=entropy_image,
        cmap=config.get("figures.segment_cmap", "Set1"),
        dpi=dpi,
    )
    figures["agreement_fig"] = segmentation_plots.plot_agreement_table(
        agreement, config.figure_path("03_cross_method_agreement.png"), dpi=dpi
    )
    figures["purity"] = segmentation_plots.plot_segment_purity(
        purity, config.figure_path("03_segment_purity.png"), dpi=dpi
    )
    if hdbscan_sweep is not None and not hdbscan_sweep.empty:
        sweep_path = config.processed_dir / "03_hdbscan_sensitivity.csv"
        hdbscan_sweep.to_csv(sweep_path, index=False)
        written["hdbscan_sweep_csv"] = sweep_path
        figures["hdbscan_sweep_fig"] = segmentation_plots.plot_hdbscan_sensitivity(
            hdbscan_sweep, config.figure_path("03_hdbscan_sensitivity.png"), dpi=dpi
        )

    # Chemistry profiles need the original counts, not the decomposition.
    images, metadata, selection = _load_selection(config)
    cube = build_cube(images, selection)
    cube_fg = cube.reshape(-1, cube.shape[-1])[fg_idx]
    profiles = validation.segment_chemistry_profiles(
        cube_fg, labels_dominant, selection.labels, segment_names(n_clusters, config)
    )
    figures["chemistry"] = segmentation_plots.plot_segment_chemistry(
        profiles, config.figure_path("03_segment_chemistry_profiles.png"), dpi=dpi
    )

    result = _merge(written, figures)
    _log(f"[segment] wrote {len(result)} artefact(s)")
    return result


# ----------------------------------------------------------------- stage: stats
def stage_stats(config: Config) -> StageResult:
    """Test whether the segments genuinely differ, channel by channel."""
    segments = cluster.load_segments(config.processed_dir)
    artefacts = decomp.load_decomposition(config.processed_dir)

    fg_mask = segments["foreground_mask"].astype(bool)
    fg_idx = np.where(fg_mask)[0]
    labels = segments["labels_dominant"]

    images, metadata, selection = _load_selection(config)
    cube = build_cube(images, selection)
    cube_fg = cube.reshape(-1, cube.shape[-1])[fg_idx]

    n_segments = len(set(labels))
    names = segment_names(n_segments, config)
    _log(f"[stats] testing {selection.n_channels} channels across {n_segments} segments")

    results, posthoc = validation.kruskal_dunn_by_channel(
        cube_fg, labels, selection.labels,
        p_adjust=config.get("stats.p_adjust", "bonferroni"),
        large_effect=float(config.get("stats.effect_size.large", 0.14)),
        medium_effect=float(config.get("stats.effect_size.medium", 0.06)),
        p_floor=float(config.get("stats.p_display_floor", 1e-50)),
    )
    _log(
        f"[stats] largest effect: {validation.strongest_effect_channel(results)} "
        f"(eps^2 = {results['eta_squared'].max():.3f})"
    )

    dpi = config.get("figures.dpi", 250)
    figures = StageResult()

    best_channel = validation.strongest_effect_channel(results)
    figures["dunn"] = stats_plots.plot_dunn_heatmap(
        posthoc[best_channel], best_channel,
        config.figure_path("04_dunn_posthoc_heatmap.png"),
        p_floor=float(config.get("stats.p_display_floor", 1e-50)), dpi=dpi,
    )

    profiles = validation.segment_chemistry_profiles(cube_fg, labels, selection.labels, names)
    figures["chemistry"] = stats_plots.plot_segment_chemistry_profiles(
        profiles, results, config.figure_path("04_segment_chemistry_profiles.png"), dpi=dpi
    )

    coloc = pd.DataFrame()
    if selection.secondary_keys:
        secondary = resample_to(images[selection.secondary_keys[0]], selection.shape)
        secondary_flat = secondary.reshape(-1)[fg_idx]

        coloc = validation.colocalization(
            secondary_flat, labels,
            strong=float(config.get("stats.colocalization.strong", 0.5)),
            moderate=float(config.get("stats.colocalization.moderate", 0.3)),
            segment_names=names,
        )
        # Which endmember the secondary ion actually tracks, rather than assuming one.
        abundances_fg = artefacts["nmf"]["abundance_maps"].reshape(
            -1, artefacts["nmf"]["abundance_maps"].shape[-1]
        )[fg_idx]
        endmember_r = validation.endmember_correlations(secondary_flat, abundances_fg)
        best_endmember = int(endmember_r["pearson_r"].abs().idxmax())
        _log(
            f"[stats] secondary ion tracks EM{best_endmember + 1} most closely "
            f"(r = {endmember_r.loc[best_endmember, 'pearson_r']:.3f})"
        )

        figures["overlay"] = stats_plots.plot_secondary_overlay(
            secondary,
            artefacts["nmf"]["abundance_maps"][:, :, best_endmember],
            config.figure_path("04_secondary_mode_overlay.png"),
            endmember_index=best_endmember,
            clip_percentile=float(config.get("figures.abundance_clip_percentile", 75)),
            dpi=dpi,
        )
        figures["coloc"] = stats_plots.plot_colocalization(
            coloc, config.figure_path("04_cross_polarity_coloc.png"), dpi=dpi
        )
    else:
        _log("[stats] no secondary-polarity images; skipping colocalization")

    written = validation.save_statistics(config.processed_dir, results, posthoc, coloc)
    result = _merge(written, figures)
    _log(f"[stats] wrote {len(result)} artefact(s)")
    return result


# --------------------------------------------------------------- stage: figures
def stage_figures(config: Config) -> StageResult:
    """Assemble the master report figures from every prior stage."""
    images, metadata, selection = _load_selection(config)
    artefacts = decomp.load_decomposition(config.processed_dir)
    segments = cluster.load_segments(config.processed_dir)

    fg_mask = segments["foreground_mask"].astype(bool)
    fg_idx = np.where(fg_mask)[0]
    labels = segments["labels_dominant"]
    shape = selection.shape
    dpi = config.get("figures.report_dpi", 300)

    result = StageResult()
    result["fig1"] = report.figure_dataset_overview(
        images, selection, metadata, config.figure_path("05_fig1_dataset_overview.png"),
        config, dpi=dpi,
    )
    result["fig2"] = report.figure_nmf_decomposition(
        artefacts["nmf"], selection.labels,
        config.figure_path("05_fig2_nmf_decomposition.png"), config, dpi=dpi,
    )

    titles = {
        "dominant": "Dominant NMF endmember",
        "kmeans_pca": "K-Means (PCA)",
        "gmm": "GMM (PCA)",
        "hdbscan": "HDBSCAN (UMAP)",
    }
    label_images = {}
    for name, title in titles.items():
        key = f"labels_{name}"
        if key in segments:
            label_images[title] = cluster.labels_to_image(segments[key], fg_idx, shape)

    agreement_path = config.processed_dir / "03_cross_method_agreement.csv"
    agreement = pd.read_csv(agreement_path) if agreement_path.exists() else pd.DataFrame()
    result["fig3"] = report.figure_segmentation_comparison(
        label_images, agreement, config.figure_path("05_fig3_segmentation_comparison.png"),
        cmap=config.get("figures.segment_cmap", "Set1"), dpi=dpi,
    )

    from src.utils.constants import STATISTICAL_RESULTS_FILE

    stats_path = config.processed_dir / STATISTICAL_RESULTS_FILE
    if stats_path.exists():
        cube = build_cube(images, selection)
        cube_fg = cube.reshape(-1, cube.shape[-1])[fg_idx]
        names = segment_names(len(set(labels)), config)
        profiles = validation.segment_chemistry_profiles(
            cube_fg, labels, selection.labels, names
        )
        result["fig4"] = report.figure_statistical_validation(
            profiles, pd.read_csv(stats_path),
            config.figure_path("05_fig4_statistical_validation.png"), dpi=dpi,
        )

    from src.utils.constants import COLOCALIZATION_FILE

    coloc_path = config.processed_dir / COLOCALIZATION_FILE
    if selection.secondary_keys and coloc_path.exists():
        coloc = pd.read_csv(coloc_path)
        secondary = resample_to(images[selection.secondary_keys[0]], shape)
        abundance = artefacts["nmf"]["abundance_maps"]
        abundances_fg = abundance.reshape(-1, abundance.shape[-1])[fg_idx]
        endmember_r = validation.endmember_correlations(
            secondary.reshape(-1)[fg_idx], abundances_fg
        )
        best_endmember = int(endmember_r["pearson_r"].abs().idxmax())
        result["fig5"] = report.figure_cross_polarity(
            secondary, abundance[:, :, best_endmember], best_endmember, coloc,
            config.figure_path("05_fig5_cross_polarity.png"), config,
            clip_percentile=float(config.get("figures.abundance_clip_percentile", 75)),
            dpi=dpi,
        )

    result["manifest"] = report.write_manifest(
        dict(result), config.figure_path("05_figure_manifest.md")
    )
    _log(f"[figures] wrote {len(result)} artefact(s)")
    return result


STAGE_FUNCTIONS: Dict[str, Callable[[Config], StageResult]] = {
    "load": stage_load,
    "overview": stage_overview,
    "decompose": stage_decompose,
    "segment": stage_segment,
    "stats": stage_stats,
    "figures": stage_figures,
}


def run_stages(config: Config, stages: List[str]) -> Dict[str, StageResult]:
    """Run the named stages in canonical order."""
    config.prepare_output_dirs()
    ordered = [stage for stage in STAGES if stage in set(stages)]

    results: Dict[str, StageResult] = {}
    for stage in ordered:
        _log(f"\n=== {stage} ===")
        results[stage] = STAGE_FUNCTIONS[stage](config)
    return results
