"""
Stage 2: resample the HiRISE orthoimage (0.25 m) onto the exact DEM grid
(1 m), after validating that the two rasters are genuinely co-registered —
same CRS, same bounds, and the image's pixel grid aligns exactly with the
DEM's (no fractional-pixel offset). The DEM and confidence raster are
already on matching grids and used as-is. Windowed processing throughout;
the orthoimage (~4.7 GB) is never loaded into memory whole.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.windows import Window

# Fraction of source (orthoimage) pixels within a ratio x ratio block that
# must be valid for the corresponding output pixel to be considered valid.
# This is a provisional, documented judgment call, not a derived constant —
# it directly trades off output coverage against averaging-in too much
# fabricated signal near data gaps. Output coverage is NOT guaranteed to
# stay close to the orthoimage's native 75.2%: partially-valid boundary
# blocks are discarded by this threshold, so coverage can drop below that
# figure, and run_stage_2() reports the actual resulting figure rather than
# assuming it. Revisit this value after inspecting real output coverage and,
# ideally, the spatial distribution of the input NoData gaps.
MIN_VALID_FRACTION = 0.5

DEM_NODATA = -32767.0
IMAGE_NODATA = 0
CONFIDENCE_NODATA = 0

# Tolerance for bounds/origin comparison, in CRS units (meters here). Not
# zero, because floating-point affine transforms from independently-written
# GeoTIFFs can differ at the sub-millimeter level even when "identical" —
# this tolerance is far tighter than a real misregistration would ever be,
# so it won't mask an actual problem.
COORD_TOLERANCE_M = 1e-3


@dataclass
class RasterInfo:
    path: Path
    crs: rasterio.crs.CRS
    transform: rasterio.Affine
    bounds: BoundingBox
    width: int
    height: int
    dtype: str
    nodata: float | None
    res: tuple[float, float]


def inspect_raster(path: Path) -> RasterInfo:
    with rasterio.open(path) as src:
        return RasterInfo(
            path=path, crs=src.crs, transform=src.transform, bounds=src.bounds,
            width=src.width, height=src.height, dtype=src.dtypes[0],
            nodata=src.nodata, res=(src.res[0], src.res[1]),
        )


def validate_matching_grid(dem: RasterInfo, confidence: RasterInfo) -> None:
    """DEM and confidence are expected to already share grid, bounds, CRS."""
    if dem.crs != confidence.crs:
        raise ValueError(f"DEM and confidence CRS differ: {dem.crs} vs {confidence.crs}")
    if (dem.width, dem.height) != (confidence.width, confidence.height):
        raise ValueError(
            f"DEM and confidence dimensions differ: "
            f"{dem.width}x{dem.height} vs {confidence.width}x{confidence.height}"
        )
    if dem.transform != confidence.transform:
        raise ValueError("DEM and confidence transforms differ — expected exact match")


def validate_resolution_ratio(dem: RasterInfo, image: RasterInfo) -> int:
    """
    Confirm the image resolution is an exact integer multiple of the DEM
    resolution, and return that integer ratio (e.g. 4). Raises if it isn't
    exact, since the block-averaging logic below assumes a clean subdivision.
    """
    ratio_x = dem.res[0] / image.res[0]
    ratio_y = dem.res[1] / image.res[1]
    if abs(ratio_x - round(ratio_x)) > 1e-6 or abs(ratio_y - round(ratio_y)) > 1e-6:
        raise ValueError(
            f"Resolution ratio is not a clean integer: x={ratio_x}, y={ratio_y}. "
            f"The block-averaging approach below assumes exact subdivision."
        )
    if round(ratio_x) != round(ratio_y):
        raise ValueError(f"Non-square resolution ratio: x={ratio_x}, y={ratio_y}")
    return round(ratio_x)


def validate_spatial_alignment(dem: RasterInfo, image: RasterInfo) -> None:
    """
    Confirm the orthoimage is genuinely co-registered with the DEM, not
    merely matching in resolution ratio and pixel count. Checks:

    1. Same CRS (exact equality — these are the same product family, so
       there's no reason to tolerate a CRS mismatch here).
    2. Same geographic bounds, within COORD_TOLERANCE_M.
    3. The image's pixel origin falls exactly on a DEM pixel boundary —
       i.e. the offset between the two origins, in DEM-pixel units, is a
       whole number (within tolerance). A fractional offset here would mean
       every 4x4 source block straddles two different DEM output pixels,
       which silently breaks the block-averaging assumption below even
       though dimensions and resolution ratio would both look correct.
    """
    if dem.crs != image.crs:
        raise ValueError(f"DEM and orthoimage CRS differ: {dem.crs} vs {image.crs}")

    for name, dem_edge, img_edge in [
        ("left", dem.bounds.left, image.bounds.left),
        ("bottom", dem.bounds.bottom, image.bounds.bottom),
        ("right", dem.bounds.right, image.bounds.right),
        ("top", dem.bounds.top, image.bounds.top),
    ]:
        if abs(dem_edge - img_edge) > COORD_TOLERANCE_M:
            raise ValueError(
                f"DEM and orthoimage bounds differ at '{name}': "
                f"{dem_edge} vs {img_edge} (tolerance {COORD_TOLERANCE_M} m). "
                f"These are not co-registered — do not proceed with resampling."
            )

    origin_offset_x = (image.bounds.left - dem.bounds.left) / dem.res[0]
    origin_offset_y = (dem.bounds.top - image.bounds.top) / dem.res[1]
    for axis, offset in [("x", origin_offset_x), ("y", origin_offset_y)]:
        if abs(offset - round(offset)) > (COORD_TOLERANCE_M / min(dem.res)):
            raise ValueError(
                f"Orthoimage pixel origin is offset from the DEM grid by a "
                f"fractional amount on the {axis} axis ({offset:.6f} DEM "
                f"pixels). Block-averaging assumes each source block maps "
                f"cleanly onto one DEM pixel — a fractional offset means it "
                f"doesn't, and must be corrected before resampling."
            )


def resample_image_to_dem_grid(
    image_path: Path,
    dem: RasterInfo,
    ratio: int,
    out_path: Path,
    row_chunk: int = 64,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> RasterInfo:
    """
    Average-resample the orthoimage down onto the DEM's exact grid, reading
    the source in horizontal strips of `row_chunk` DEM-grid rows at a time
    (each strip is `row_chunk * ratio` source rows), so memory use stays
    bounded regardless of file size. At row_chunk=64 and ratio=4, each
    strip reads 256 source rows x ~76,576 columns of uint8 — under 20 MB
    per strip, comfortably inside a 16 GB machine's budget even with a few
    intermediate float32 copies of the same shape held at once.

    NoData handling: within each ratio x ratio source block, pixels equal to
    IMAGE_NODATA are excluded from the average. If the valid fraction in a
    block falls below min_valid_fraction, the output pixel is set to
    IMAGE_NODATA rather than an average computed from too little real data.
    """
    with rasterio.open(image_path) as src:
        if src.width != dem.width * ratio or src.height != dem.height * ratio:
            raise ValueError(
                f"Orthoimage dimensions {src.width}x{src.height} are not "
                f"exactly {ratio}x the DEM's {dem.width}x{dem.height}"
            )

        out_meta = {
            "driver": "GTiff",
            "height": dem.height,
            "width": dem.width,
            "count": 1,
            "dtype": src.dtypes[0],
            "crs": dem.crs,
            "transform": dem.transform,
            "nodata": IMAGE_NODATA,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        with rasterio.open(out_path, "w", **out_meta) as dst:
            for out_row_start in range(0, dem.height, row_chunk):
                out_rows = min(row_chunk, dem.height - out_row_start)
                src_row_start = out_row_start * ratio
                src_rows = out_rows * ratio
                out_width = dem.width

                window = Window(0, src_row_start, src.width, src_rows)
                block = src.read(1, window=window)  # (src_rows, src.width), uint8

                valid = src.read_masks(1, window=window) != 0
                block_f = block.astype(np.float32)
                block_f[~valid] = 0.0

                block_f = block_f.reshape(out_rows, ratio, out_width, ratio)
                valid_f = valid.reshape(out_rows, ratio, out_width, ratio).astype(np.float32)

                sums = block_f.sum(axis=(1, 3))
                counts = valid_f.sum(axis=(1, 3))
                valid_fraction = counts / (ratio * ratio)

                with np.errstate(invalid="ignore", divide="ignore"):
                    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

                result = np.where(
                    valid_fraction >= min_valid_fraction,
                    np.round(means).astype(src.dtypes[0]),
                    IMAGE_NODATA,
                )

                dst.write(result, 1, window=Window(0, out_row_start, out_width, out_rows))

                del block, valid, block_f, valid_f, sums, counts, valid_fraction, means, result

    return inspect_raster(out_path)


def count_valid_fraction_windowed(path: Path, nodata, row_chunk: int = 512) -> float:
    """
    Compute the fraction of valid (non-NoData) pixels in a raster using
    windowed block reads, so inspecting a large output never requires
    loading it into memory whole — the same discipline used for input
    inspection, applied symmetrically to output validation.
    """
    with rasterio.open(path) as src:
        valid_count = 0
        total = 0
        for row_start in range(0, src.height, row_chunk):
            rows = min(row_chunk, src.height - row_start)
            window = Window(0, row_start, src.width, rows)
            block = src.read(1, window=window)
            valid_count += int(np.sum(block != nodata))
            total += block.size
    return valid_count / total


def run_stage_2(
    raw_dir: Path,
    out_dir: Path,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> dict:
    """
    Entry point. Validates the three input rasters — including genuine
    spatial co-registration, not just resolution/dimension agreement —
    resamples the orthoimage onto the DEM grid, and returns a summary dict
    with the ACTUAL resulting coverage (not assumed to match the native
    75.2% figure).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    dem = inspect_raster(raw_dir / "dem.tif")
    image = inspect_raster(raw_dir / "orthoimage.tif")
    confidence = inspect_raster(raw_dir / "confidence.tif")

    if dem.nodata != DEM_NODATA:
        raise ValueError(f"Expected DEM NoData {DEM_NODATA}, found {dem.nodata}")
    if confidence.nodata != CONFIDENCE_NODATA:
        raise ValueError(f"Expected confidence NoData {CONFIDENCE_NODATA}, found {confidence.nodata}")

    validate_matching_grid(dem, confidence)
    ratio = validate_resolution_ratio(dem, image)
    validate_spatial_alignment(dem, image)

    out_image_path = out_dir / "orthoimage_resampled_1m.tif"
    resampled_image = resample_image_to_dem_grid(
        image_path=raw_dir / "orthoimage.tif",
        dem=dem,
        ratio=ratio,
        out_path=out_image_path,
        min_valid_fraction=min_valid_fraction,
    )

    actual_valid_fraction = count_valid_fraction_windowed(resampled_image.path, IMAGE_NODATA)

    return {
        "dem": dem,
        "confidence": confidence,
        "resampled_image": resampled_image,
        "min_valid_fraction_used": min_valid_fraction,
        "resampled_valid_fraction": actual_valid_fraction,
    }


if __name__ == "__main__":
    summary = run_stage_2(Path("data-pipeline/raw"), Path("data-pipeline/output"))
    print(f"DEM grid:              {summary['dem'].width}x{summary['dem'].height}")
    print(f"Resampled image grid:  {summary['resampled_image'].width}x{summary['resampled_image'].height}")
    print(f"MIN_VALID_FRACTION:    {summary['min_valid_fraction_used']}")
    print(f"Resampled valid frac:  {summary['resampled_valid_fraction']:.3%}  (native orthoimage: 75.203%)")
