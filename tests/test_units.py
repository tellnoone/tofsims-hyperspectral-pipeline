"""
Unit tests for the parts that decide how a dataset is interpreted.

These are the functions that replaced hardcoded assumptions, so they are the ones
that determine whether a new dataset is handled correctly.
"""
import numpy as np
import pytest

from src.features.decomposition import select_nmf_rank
from src.features.selection import modal_shape, resample_to
from src.preprocessing.load_images import parse_tof_bmp
from src.preprocessing.metadata import extract_metadata
from src.utils.formatting import effect_size_label, extract_mass, fmt_mass, fmt_p
from tests.conftest import write_tof_bmp


class TestMetadataExtraction:
    """Filename parsing has to survive inconsistent naming without losing fields."""

    def test_parses_full_instrument_filename(self):
        record = extract_metadata(
            "Fly_Neg_C60_30pA_10x10x64pix_400sh-35V_103.91326±0.02297u"
        )
        assert record["polarity"] == "Neg"
        assert record["shots"] == 400
        assert record["current"] == "30pA"
        assert record["dimensions"] == "10x10x64pix"
        assert record["mass"] == "103.91326±0.02297u"
        assert record["matched"] is True

    def test_recovers_fields_from_irregular_filename(self):
        # Space-separated, no beam current or raster geometry in the name.
        record = extract_metadata("FossilFly Pos 8mm x 11.2mm 53.93733±0.015u")
        assert record["polarity"] == "Pos"
        assert record["mass"] == "53.93733±0.015u"
        # Absent fields are reported as missing rather than guessed.
        assert record["shots"] is None
        assert record["matched"] is False

    def test_normalises_polarity_spelling(self):
        assert extract_metadata("sample_negative_10sh_1.0±0.1u")["polarity"] == "Neg"
        assert extract_metadata("sample_POSITIVE_10sh_1.0±0.1u")["polarity"] == "Pos"

    def test_overrides_fill_missing_fields(self):
        record = extract_metadata(
            "FossilFly Pos 8mm x 11.2mm 53.93733±0.015u",
            overrides={"fossilfly": {"shots": 100}},
        )
        assert record["shots"] == 100


class TestBmpParsing:
    """The binary reader must honour the header rather than assume one geometry."""

    def test_roundtrips_a_written_image(self, tmp_path):
        original = np.arange(64 * 48, dtype=np.uint16).reshape(64, 48) % 4096
        path = write_tof_bmp(tmp_path / "probe.bmp", original)

        parsed = parse_tof_bmp(path)

        assert parsed.shape == (64, 48)
        np.testing.assert_array_equal(parsed, original.astype(np.float64))

    def test_reads_dimensions_from_header_not_a_constant(self, tmp_path):
        path = write_tof_bmp(tmp_path / "odd.bmp", np.ones((37, 53), dtype=np.uint16))
        assert parse_tof_bmp(path).shape == (37, 53)

    def test_selects_the_configured_channel(self, tmp_path):
        image = np.full((16, 16), 7, dtype=np.uint16)
        path = write_tof_bmp(tmp_path / "ch.bmp", image, channel_index=2)

        assert parse_tof_bmp(path, channel_index=2).max() == 7
        assert parse_tof_bmp(path, channel_index=1).max() == 0

    def test_rejects_a_non_bmp_file(self, tmp_path):
        path = tmp_path / "bogus.bmp"
        path.write_bytes(b"XX" + bytes(200))
        with pytest.raises(ValueError, match="not a BMP"):
            parse_tof_bmp(path)


class TestSelectionHeuristics:
    def test_modal_shape_prefers_the_most_common_grid(self):
        images = {
            "a": np.zeros((10, 10)), "b": np.zeros((10, 10)),
            "c": np.zeros((20, 20)),
        }
        assert modal_shape(images) == (10, 10)

    def test_modal_shape_breaks_ties_toward_the_larger_grid(self):
        images = {"a": np.zeros((10, 10)), "b": np.zeros((20, 20))}
        assert modal_shape(images) == (20, 20)

    @pytest.mark.parametrize(
        "source,target",
        [((896, 640), (640, 640)), ((100, 100), (250, 80)), ((640, 640), (640, 640))],
    )
    def test_resample_lands_exactly_on_the_target_grid(self, source, target):
        assert resample_to(np.random.rand(*source), target).shape == target


class TestNmfRankSelection:
    def test_picks_the_elbow_of_the_error_curve(self):
        # The steep drop stops after k=3, which is where the elbow sits.
        sweep = {
            2: {"error": 248987.5},
            3: {"error": 145738.2},
            4: {"error": 144089.8},
            5: {"error": 117354.3},
        }
        assert select_nmf_rank(sweep) == 3

    def test_handles_a_sweep_too_short_to_have_an_elbow(self):
        assert select_nmf_rank({2: {"error": 10.0}, 3: {"error": 5.0}}) == 2


class TestFormatting:
    def test_extracts_mass_from_an_instrument_key(self):
        assert extract_mass("Fly_Neg_400sh_103.91326±0.02297u") == pytest.approx(103.91326)

    def test_formats_mass_for_display(self):
        assert fmt_mass("103.91326±0.02297u") == "m/z 103.91"

    def test_floors_unrepresentable_p_values(self):
        assert fmt_p(0.0) == "<1e-50"
        assert fmt_p(0.042) == "0.042"

    def test_buckets_effect_sizes(self):
        assert effect_size_label(0.73) == "Large"
        assert effect_size_label(0.10) == "Medium"
        assert effect_size_label(0.01) == "Small"
