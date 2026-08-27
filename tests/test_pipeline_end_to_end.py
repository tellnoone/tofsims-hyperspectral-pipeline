"""
End-to-end pipeline tests.

Two things need to hold for this repository to be useful to anyone else:

  1. Re-running the pipeline on the bundled data reproduces the published numbers
     (`TestBundledDatasetRegression`).
  2. Pointing it at a completely different dataset produces the same *kinds* of
     results without editing any code (`TestUnseenDataset`).

The second is the one that matters for a newcomer: the synthetic dataset has a
different pixel grid, a different channel count and a different naming scheme, so
anything that still assumed the Fossil Fly geometry would fail here.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.pipeline import run_stages
from src.utils import constants

# Values captured from the published Fossil Fly analysis.
EXPECTED_FOREGROUND_PIXELS = 48294
EXPECTED_NMF_RANK = 3
EXPECTED_NMF_ERROR = 145738.2
EXPECTED_PCA_TOP3_VARIANCE = 0.8042
EXPECTED_MAX_EFFECT_SIZE = 0.7311


@pytest.fixture(scope="class")
def bundled_run(tmp_path_factory, real_data_dir):
    """Run every stage on the bundled data into a throwaway output directory."""
    output = tmp_path_factory.mktemp("bundled_output")
    config = load_config(
        overrides={
            "dataset": {
                "raw_dir": str(real_data_dir),
                "processed_dir": str(output),
                "figures_dir": str(output),
            }
        }
    )
    run_stages(config, ["load", "decompose", "segment", "stats"])
    return output


@pytest.fixture(scope="class")
def synthetic_run(tmp_path_factory, synthetic_dataset):
    """
    Run the pipeline on a dataset it has never seen, using the stock config.

    UMAP is disabled only because it is slow and stochastic on a tiny image; every
    other stage runs exactly as it would for a real new dataset.
    """
    output = tmp_path_factory.mktemp("synthetic_output")
    config = load_config(
        overrides={
            "dataset": {
                "raw_dir": str(synthetic_dataset),
                "processed_dir": str(output),
                "figures_dir": str(output),
            },
            "decomposition": {"umap": {"enabled": False}},
            "segmentation": {"hdbscan": {"enabled": False}},
        }
    )
    run_stages(config, ["load", "overview", "decompose", "segment", "stats", "figures"])
    return output


class TestBundledDatasetRegression:
    """Guards the published numbers against accidental drift."""

    def test_selects_the_five_negative_mode_channels(self, bundled_run):
        from src.features.selection import select_analysis_channels
        from src.preprocessing.stack_io import load_clean_stack

        config = load_config(overrides={"dataset": {"processed_dir": str(bundled_run)}})
        images, metadata = load_clean_stack(bundled_run)
        selection = select_analysis_channels(images, metadata, config)

        assert len(images) == 7
        assert selection.n_channels == 5
        assert selection.shape == (640, 640)
        assert selection.polarity == "Neg"
        assert len(selection.secondary_keys) == 2
        # Channels are ordered by mass so the cube is reproducible run to run.
        assert selection.masses == sorted(selection.masses)

    def test_foreground_pixel_count_is_unchanged(self, bundled_run):
        mask = np.load(bundled_run / constants.FOREGROUND_MASK_FILE)
        assert int(mask.sum()) == EXPECTED_FOREGROUND_PIXELS

    def test_nmf_rank_and_error_match_the_published_run(self, bundled_run):
        with np.load(bundled_run / constants.NMF_FILE) as archive:
            assert int(archive["n_components"]) == EXPECTED_NMF_RANK
            assert float(archive["reconstruction_error"]) == pytest.approx(
                EXPECTED_NMF_ERROR, rel=1e-3
            )

    def test_nmf_saves_the_endmember_matrices_not_the_image_dimensions(self, bundled_run):
        """
        Regression test for a bug in the original notebook.

        `H, W, C = cube.shape` shadowed the NMF matrices, so the saved archive held
        the integers 640 and 640 instead of the factorisation.
        """
        with np.load(bundled_run / constants.NMF_FILE) as archive:
            assert archive["W"].shape == (640 * 640, EXPECTED_NMF_RANK)
            assert archive["H"].shape == (EXPECTED_NMF_RANK, 5)

    def test_pca_variance_is_unchanged(self, bundled_run):
        with np.load(bundled_run / constants.PCA_FILE) as archive:
            top3 = archive["explained_variance"][:3].sum()
        assert top3 == pytest.approx(EXPECTED_PCA_TOP3_VARIANCE, abs=5e-4)

    def test_every_channel_separates_the_segments_significantly(self, bundled_run):
        results = pd.read_csv(bundled_run / constants.STATISTICAL_RESULTS_FILE)

        assert len(results) == 5
        assert (results["p_value"] < 0.001).all()
        assert results["eta_squared"].max() == pytest.approx(
            EXPECTED_MAX_EFFECT_SIZE, abs=5e-3
        )

    def test_cross_polarity_signal_does_not_colocalise(self, bundled_run):
        """The positive-mode ion tracks none of the segments — as originally found."""
        coloc = pd.read_csv(bundled_run / constants.COLOCALIZATION_FILE)
        assert (coloc["pearson_r"].abs() < 0.3).all()


class TestUnseenDataset:
    """The requirement that matters for anyone cloning this repository."""

    def test_pipeline_completes_on_an_unfamiliar_dataset(self, synthetic_run):
        expected = [
            constants.CLEAN_STACK_FILE,
            constants.METADATA_FILE,
            constants.PCA_FILE,
            constants.NMF_FILE,
            constants.SEGMENTS_FILE,
            constants.STATISTICAL_RESULTS_FILE,
        ]
        missing = [name for name in expected if not (synthetic_run / name).exists()]
        assert not missing, f"Stages did not produce: {missing}"

    def test_adapts_to_the_new_pixel_grid_and_channel_count(self, synthetic_run):
        """Nothing may fall back to the Fossil Fly geometry of 640x640x5."""
        with np.load(synthetic_run / constants.PCA_FILE) as archive:
            score_maps = archive["score_maps"]

        assert score_maps.shape[:2] == (96, 128)
        assert score_maps.shape[2] == 3  # three synthetic channels, not five

    def test_finds_the_planted_chemistry(self, synthetic_run):
        """
        The synthetic specimen has two spatially separated chemistries, so the
        segmentation should recover more than one populated segment and the tests
        should register a real effect.
        """
        with np.load(synthetic_run / constants.SEGMENTS_FILE) as archive:
            labels = archive["labels_dominant"]

        assert len(set(labels)) >= 2

        results = pd.read_csv(synthetic_run / constants.STATISTICAL_RESULTS_FILE)
        assert len(results) == 3
        assert results["eta_squared"].max() > 0.1

    def test_generates_the_report_figures(self, synthetic_run):
        figures = sorted(path.name for path in synthetic_run.glob("05_fig*.png"))
        assert len(figures) >= 3, f"Only produced: {figures}"

    def test_handles_the_mismatched_secondary_acquisition(self, synthetic_run):
        """The positive-mode image is on a different grid and must be resampled."""
        coloc_path = synthetic_run / constants.COLOCALIZATION_FILE
        assert coloc_path.exists()
        assert len(pd.read_csv(coloc_path)) >= 2
