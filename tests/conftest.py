"""
Shared test fixtures.

`synthetic_dataset` builds a complete, valid ToF-SIMS dataset from scratch with a
different pixel grid, channel count and naming scheme than the bundled Fossil Fly
data. Running the pipeline over it is the real test of whether the code generalises,
because nothing about it matches the dataset the analysis was originally written for.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def write_tof_bmp(
    path: Path, image: np.ndarray, num_channels: int = 4, channel_index: int = 1
) -> Path:
    """
    Write an image in the instrument's export format.

    A standard 54-byte BMP header followed by interleaved uint16 planes, with the
    payload stored bottom-up as BMP requires.
    """
    height, width = image.shape
    payload = np.zeros((height, width, num_channels), dtype=np.uint16)
    payload[:, :, channel_index] = np.flipud(image).astype(np.uint16)

    header = bytearray(54)
    header[0:2] = b"BM"
    struct.pack_into("<I", header, 2, 54 + payload.nbytes)  # file size
    struct.pack_into("<I", header, 10, 54)                  # payload offset
    struct.pack_into("<I", header, 14, 40)                  # info header size
    struct.pack_into("<I", header, 18, width)
    struct.pack_into("<I", header, 22, height)
    struct.pack_into("<H", header, 26, 1)                   # planes
    struct.pack_into("<H", header, 28, 16 * num_channels)   # bits per pixel

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(bytes(header))
        handle.write(payload.tobytes())
    return path


def _blob(shape, centre, radius, peak, rng):
    """A soft circular feature — stands in for an anatomical structure."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    distance = np.sqrt((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2)
    field = peak * np.exp(-(distance ** 2) / (2 * radius ** 2))
    return np.clip(field + rng.normal(0, peak * 0.02, shape), 0, None)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> Path:
    """
    A three-channel, 96x128 negative-mode dataset plus one positive-mode image on a
    different grid — deliberately unlike the bundled data in every dimension.
    """
    rng = np.random.default_rng(7)
    raw_dir = tmp_path_factory.mktemp("synthetic") / "raw"
    shape = (96, 128)

    # Two spatially distinct chemistries so segmentation has something real to find.
    region_a = _blob(shape, (30, 40), 14, 900, rng)
    region_b = _blob(shape, (65, 90), 18, 700, rng)

    channels = {
        "Bug_Neg_C60_20pA_8x8x32pix_250sh_44.12345±0.01000u": region_a * 1.0 + region_b * 0.1,
        "Bug_Neg_C60_20pA_8x8x32pix_250sh_71.98765±0.01500u": region_a * 0.15 + region_b * 1.0,
        "Bug_Neg_C60_20pA_8x8x32pix_250sh_95.55555±0.02000u": region_a * 0.6 + region_b * 0.55,
    }
    for name, image in channels.items():
        write_tof_bmp(raw_dir / f"{name}.bmp", image)

    # Secondary polarity on a different field of view, exercising the resampling path.
    write_tof_bmp(
        raw_dir / "Bug_Pos_C60_20pA_8x11x32pix_120sh_39.11111±0.01000u.bmp",
        _blob((132, 128), (40, 50), 16, 500, rng),
    )
    return raw_dir


@pytest.fixture(scope="session")
def real_data_dir() -> Path:
    """The bundled Fossil Fly raw data, skipping the test if it is absent."""
    from src.utils.paths import resolve_dir

    raw_dir = resolve_dir("data/raw")
    if not raw_dir.exists() or not list(raw_dir.glob("*.bmp")):
        pytest.skip("Bundled raw dataset not available")
    return raw_dir
