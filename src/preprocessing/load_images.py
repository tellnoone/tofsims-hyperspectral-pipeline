"""
load_images.py
Reader for the instrument's non-standard BMP export.

The ToF-SIMS exporter writes a normal 54-byte BMP header that claims 64 bits per
pixel, then dumps interleaved uint16 planes. PIL and OpenCV both refuse or misread
these files, so the payload is parsed directly from the bytes. Which plane carries
the ion counts, how many planes there are, and the pixel dtype are all configurable
via the `io` block of the pipeline config.
"""
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

# Byte offsets of the fields we need from a standard BITMAPFILEHEADER/BITMAPINFOHEADER.
_MAGIC = slice(0, 2)
_DATA_OFFSET = slice(10, 14)
_WIDTH = slice(18, 22)
_HEIGHT = slice(22, 26)


def read_bmp_header(file_path: PathLike, header_bytes: int = 54) -> Dict[str, int]:
    """Return the width, height and payload offset declared by the BMP header."""
    with open(file_path, "rb") as handle:
        header = handle.read(header_bytes)

    if len(header) < header_bytes:
        raise ValueError(f"{Path(file_path).name}: file shorter than its {header_bytes}-byte header")

    height = struct.unpack("<i", header[_HEIGHT])[0]
    return {
        "magic": header[_MAGIC],
        "data_offset": struct.unpack("<I", header[_DATA_OFFSET])[0],
        "width": struct.unpack("<I", header[_WIDTH])[0],
        # A negative height means the rows are already stored top-down.
        "height": abs(height),
        "top_down": height < 0,
    }


def parse_tof_bmp(
    file_path: PathLike,
    header_bytes: int = 54,
    num_channels: int = 4,
    channel_index: int = 1,
    dtype: str = "uint16",
    flip_vertical: bool = True,
    dimensions_from_header: bool = True,
    fallback_dimensions: Tuple[int, int] = (640, 640),
    require_bmp_magic: bool = True,
) -> np.ndarray:
    """
    Parse one exported BMP into a 2-D ion-count map.

    Returns a float64 array of shape (height, width) taken from `channel_index` of
    the interleaved planes.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    header = read_bmp_header(file_path, header_bytes=header_bytes)
    if require_bmp_magic and header["magic"] != b"BM":
        raise ValueError(f"{file_path.name} is not a BMP file (magic={header['magic']!r})")

    if dimensions_from_header and header["width"] and header["height"]:
        height, width = header["height"], header["width"]
        payload_offset = header["data_offset"] or header_bytes
    else:
        height, width = fallback_dimensions
        payload_offset = header_bytes

    expected_elements = height * width * num_channels

    with open(file_path, "rb") as handle:
        handle.seek(payload_offset)
        raw_bytes = handle.read()

    data = np.frombuffer(raw_bytes, dtype=np.dtype(dtype))
    if data.size < expected_elements:
        raise ValueError(
            f"{file_path.name}: payload holds {data.size} {dtype} elements, "
            f"expected at least {expected_elements} for a "
            f"{height}x{width}x{num_channels} frame."
        )

    # Some exports pad the head of the payload; the frame sits at the tail.
    frame = data[-expected_elements:].reshape((height, width, num_channels))

    if flip_vertical and not header["top_down"]:
        frame = np.flipud(frame)

    if not 0 <= channel_index < num_channels:
        raise ValueError(
            f"channel_index {channel_index} is out of range for {num_channels} channels"
        )

    return frame[:, :, channel_index].astype(np.float64)


def load_raw_images(
    folder_path: PathLike,
    file_pattern: str = "*.bmp",
    **parse_kwargs,
) -> Dict[str, np.ndarray]:
    """
    Parse every matching file in a folder into a {filename stem: 2-D array} mapping.

    Images are keyed by filename stem rather than stacked, because a dataset may mix
    acquisition geometries (different pixel grids) that cannot share one array.
    """
    folder_path = Path(folder_path)
    files = sorted(folder_path.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in {folder_path}")

    images: Dict[str, np.ndarray] = {}
    for path in files:
        images[path.stem] = parse_tof_bmp(path, **parse_kwargs)
    return images


def load_images_from_config(config) -> Dict[str, np.ndarray]:
    """Convenience wrapper that pulls every parser setting from the pipeline config."""
    io_cfg = config.section("io")
    return load_raw_images(
        config.raw_dir,
        file_pattern=config.get("dataset.file_pattern", "*.bmp"),
        header_bytes=io_cfg.get("header_bytes", 54),
        num_channels=io_cfg.get("num_channels", 4),
        channel_index=io_cfg.get("channel_index", 1),
        dtype=io_cfg.get("dtype", "uint16"),
        flip_vertical=io_cfg.get("flip_vertical", True),
        dimensions_from_header=io_cfg.get("dimensions_from_header", True),
        fallback_dimensions=tuple(io_cfg.get("fallback_dimensions", (640, 640))),
        require_bmp_magic=io_cfg.get("require_bmp_magic", True),
    )


def stack_images(images: Dict[str, np.ndarray], keys: List[str]) -> np.ndarray:
    """Stack the named images into an (H, W, C) cube, validating shape agreement."""
    if not keys:
        raise ValueError("No image keys supplied for stacking.")
    shapes = {images[key].shape for key in keys}
    if len(shapes) > 1:
        raise ValueError(f"Cannot stack images with differing shapes: {sorted(shapes)}")
    return np.stack([images[key] for key in keys], axis=-1).astype(np.float64)
