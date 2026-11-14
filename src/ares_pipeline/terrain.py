"""
Stage 3: derive slope, detrended roughness, and traversal cost from the DEM.

Processing is performed in row strips with halo padding, avoiding loading
the full ~513-million-cell DEM into memory.

Roughness is calculated by fitting a local plane to each 5x5 neighbourhood.
Convolution computes the required local sums, and NumPy solves the resulting
small linear systems in batches.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import convolve


DEM_NODATA = -32767.0

# Modelling assumptions, not NASA-published EVA safety limits.
SLOPE_WEIGHT = 0.7
ROUGHNESS_WEIGHT = 0.3
IMPASSABLE_SLOPE_DEGREES = 30.0

NORMALIZATION_LOW_PERCENTILE = 2.0
NORMALIZATION_HIGH_PERCENTILE = 98.0

ROUGHNESS_WINDOW = 5
MIN_VALID_FRACTION_FOR_ROUGHNESS = 0.6

PERCENTILE_SAMPLE_SIZE = 200_000
PERCENTILE_RANDOM_SEED = 0


@dataclass
class TerrainSummary:
    rows: int
    cols: int
    slope_path: Path
    roughness_path: Path
    cost_path: Path
    passable_fraction: float
    slope_min: float
    slope_max: float
    slope_mean: float


def _halo(window: int) -> int:
    return window // 2


def _validate_roughness_window(window: int) -> None:
    if not isinstance(window, int) or window < 3 or window % 2 == 0:
        raise ValueError("roughness_window must be an odd integer >= 3")


def load_dem_metadata(dem_path: Path):
    with rasterio.open(dem_path) as src:
        return (
            src.height,
            src.width,
            src.res[0],
            src.res[1],
            src.crs,
            src.transform,
        )


def _read_elevation_and_validity(
    dem_src: rasterio.DatasetReader,
    window: Window,
):
    """
    Read elevation and use Rasterio's dataset mask to determine validity.

    Cross-check the mask against the expected source NoData value.
    """
    block = dem_src.read(1, window=window).astype(np.float64)
    mask = dem_src.read_masks(1, window=window)

    valid = mask != 0
    expected_valid = block != DEM_NODATA

    if not np.array_equal(valid, expected_valid):
        mismatch_count = int(np.sum(valid != expected_valid))
        raise ValueError(
            "Rasterio's dataset mask disagrees with the expected "
            f"NoData value ({DEM_NODATA}) at {mismatch_count} pixels."
        )

    elevation = np.where(valid, block, np.nan)

    return elevation, valid


def _compute_slope_strip(
    elevation: np.ndarray,
    valid: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    local_start: int,
    local_end: int,
) -> np.ndarray:
    """
    Calculate slope in degrees using centered differences.

    A cell is invalid if it or any of its four direct neighbours is invalid.
    Grid boundaries are handled explicitly without wrapping.
    """
    rows, cols = elevation.shape

    dz_dx = np.full((rows, cols), np.nan, dtype=np.float64)
    dz_dy = np.full((rows, cols), np.nan, dtype=np.float64)

    has_left = np.zeros((rows, cols), dtype=bool)
    has_right = np.zeros((rows, cols), dtype=bool)

    has_left[:, 1:] = valid[:, :-1]
    has_right[:, :-1] = valid[:, 1:]

    center_x = has_left & has_right & valid

    dz_dx[:, 1:-1] = np.where(
        center_x[:, 1:-1],
        (elevation[:, 2:] - elevation[:, :-2]) / (2 * pixel_size_x),
        np.nan,
    )

    has_up = np.zeros((rows, cols), dtype=bool)
    has_down = np.zeros((rows, cols), dtype=bool)

    has_up[1:, :] = valid[:-1, :]
    has_down[:-1, :] = valid[1:, :]

    center_y = has_up & has_down & valid

    dz_dy[1:-1, :] = np.where(
        center_y[1:-1, :],
        (elevation[2:, :] - elevation[:-2, :]) / (2 * pixel_size_y),
        np.nan,
    )

    slope = np.degrees(
        np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    )

    slope[~valid] = np.nan

    return slope[local_start:local_end, :]


def _compute_roughness_strip_vectorized(
    elevation: np.ndarray,
    valid: np.ndarray,
    window: int,
    local_start: int,
    local_end: int,
    min_valid_fraction: float = MIN_VALID_FRACTION_FOR_ROUGHNESS,
    touches_top_boundary: bool = True,
    touches_bottom_boundary: bool = True,
) -> np.ndarray:
    """
    Calculate detrended roughness using vectorized local plane fitting.

    Each neighbourhood gets a best-fit plane. Roughness is the standard
    deviation of the elevation residuals after removing that plane.

    Actual DEM boundaries are treated differently from artificial strip
    boundaries. Artificial boundaries are allowed because the caller
    supplies halo rows around the output strip.
    """
    _validate_roughness_window(window)

    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in (0, 1]")

    half = _halo(window)
    rows, cols = elevation.shape

    if valid.shape != elevation.shape:
        raise ValueError("elevation and valid must have the same shape")

    if not 0 <= local_start <= local_end <= rows:
        raise ValueError("Invalid local_start/local_end range")

    # Coordinates relative to the centre of each neighbourhood.
    yy, xx = np.mgrid[
        -half:half + 1,
        -half:half + 1,
    ].astype(np.float64)

    # Shift elevations by a reference value before calculating sums.
    # Plane fitting and residual standard deviation are invariant under
    # a constant vertical offset. This reduces cancellation in RSS.
    if np.any(valid):
        first_valid_index = int(np.argmax(valid))
        reference_elevation = elevation.flat[first_valid_index]
    else:
        reference_elevation = 0.0

    shifted = np.zeros((rows, cols), dtype=np.float64)
    np.subtract(
        elevation,
        reference_elevation,
        out=shifted,
        where=valid,
    )

    v = valid.astype(np.float64)

    def conv(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return convolve(
            array,
            kernel,
            mode="constant",
            cval=0.0,
        )

    ones = np.ones((window, window), dtype=np.float64)

    # Local position sums.
    n = conv(v, ones)
    sum_x = conv(v, xx)
    sum_y = conv(v, yy)
    sum_xx = conv(v, xx * xx)
    sum_yy = conv(v, yy * yy)
    sum_xy = conv(v, xx * yy)

    # Local elevation sums, using shifted elevations.
    ez = shifted * v

    sum_z = conv(ez, ones)
    sum_zx = conv(ez, xx)
    sum_zy = conv(ez, yy)
    sum_zz = conv(shifted**2 * v, ones)

    min_valid = max(
        4,
        int(window * window * min_valid_fraction),
    )

    enough = n >= min_valid

    # Build the normal-equation matrix for the plane:
    # z = a*x + b*y + c
    A = np.zeros((rows, cols, 3, 3), dtype=np.float64)

    A[..., 0, 0] = sum_xx
    A[..., 0, 1] = sum_xy
    A[..., 0, 2] = sum_x

    A[..., 1, 0] = sum_xy
    A[..., 1, 1] = sum_yy
    A[..., 1, 2] = sum_y

    A[..., 2, 0] = sum_x
    A[..., 2, 1] = sum_y
    A[..., 2, 2] = n

    b = np.stack(
        [sum_zx, sum_zy, sum_z],
        axis=-1,
    )

    # Insufficient neighbourhoods do not need a plane fit.
    # Replace their systems with identity systems so the batched solver
    # never encounters singular matrices from empty/incomplete windows.
    identity = np.eye(3, dtype=np.float64)

    A[~enough] = identity
    b[~enough] = 0.0

    coeffs = np.linalg.solve(A, b[..., None])[..., 0]

    a_coef = coeffs[..., 0]
    b_coef = coeffs[..., 1]
    c_coef = coeffs[..., 2]

    # Residual sum of squares from the least-squares identity.
    rss = sum_zz - (
        a_coef * sum_zx
        + b_coef * sum_zy
        + c_coef * sum_z
    )

    # Permit only tiny negative values caused by floating-point rounding.
    # A materially negative RSS indicates a numerical or implementation
    # problem and should not silently be converted into zero.
    scale = np.maximum(np.abs(sum_zz), 1.0)
    tolerance = 100.0 * np.finfo(np.float64).eps * scale

    materially_negative = rss < -tolerance

    if np.any(materially_negative & enough):
        raise FloatingPointError(
            "Roughness calculation produced materially negative "
            "residual sums of squares."
        )

    rss = np.maximum(rss, 0.0)

    variance = np.full((rows, cols), np.nan, dtype=np.float64)

    np.divide(
        rss,
        n,
        out=variance,
        where=enough,
    )

    roughness = np.sqrt(variance)

    # Reject incomplete neighbourhoods at the actual raster boundaries.
    # Artificial strip boundaries remain usable because their halo
    # provides the missing neighbouring rows.
    roughness[:, :half] = np.nan
    roughness[:, -half:] = np.nan

    if touches_top_boundary:
        roughness[:half, :] = np.nan

    if touches_bottom_boundary:
        roughness[-half:, :] = np.nan

    return roughness[local_start:local_end, :]


def run_stage_3(
    dem_path: Path,
    out_dir: Path,
    row_chunk: int = 256,
    roughness_window: int = ROUGHNESS_WINDOW,
) -> TerrainSummary:
    """
    Generate slope, roughness, and cost GeoTIFFs from the DEM.
    """
    _validate_roughness_window(roughness_window)

    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)

    (
        rows,
        cols,
        pixel_size_x,
        pixel_size_y,
        crs,
        transform,
    ) = load_dem_metadata(dem_path)

    slope_path = out_dir / "slope_degrees.tif"
    roughness_path = out_dir / "roughness.tif"
    cost_path = out_dir / "cost.tif"

    meta_f32 = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    halo = max(1, _halo(roughness_window))

    with (
        rasterio.open(dem_path) as dem_src,
        rasterio.open(slope_path, "w", **meta_f32) as slope_dst,
        rasterio.open(roughness_path, "w", **meta_f32) as rough_dst,
    ):
        for row_start in range(0, rows, row_chunk):
            strip_rows = min(row_chunk, rows - row_start)

            read_start = max(0, row_start - halo)
            read_end = min(rows, row_start + strip_rows + halo)

            read_window = Window(
                0,
                read_start,
                cols,
                read_end - read_start,
            )

            elevation, valid = _read_elevation_and_validity(
                dem_src,
                read_window,
            )

            local_start = row_start - read_start
            local_end = local_start + strip_rows

            slope_strip = _compute_slope_strip(
                elevation,
                valid,
                pixel_size_x,
                pixel_size_y,
                local_start,
                local_end,
            )

            roughness_strip = _compute_roughness_strip_vectorized(
                elevation,
                valid,
                roughness_window,
                local_start,
                local_end,
                touches_top_boundary=(read_start == 0),
                touches_bottom_boundary=(read_end == rows),
            )

            write_window = Window(
                0,
                row_start,
                cols,
                strip_rows,
            )

            slope_dst.write(
                slope_strip.astype(np.float32),
                1,
                window=write_window,
            )

            rough_dst.write(
                roughness_strip.astype(np.float32),
                1,
                window=write_window,
            )

            del elevation, valid, slope_strip, roughness_strip

    # Calculate separate normalization bounds for slope and roughness.
    slope_low, slope_high = _compute_percentile_bounds(
        [slope_path],
        NORMALIZATION_LOW_PERCENTILE,
        NORMALIZATION_HIGH_PERCENTILE,
    )

    roughness_low, roughness_high = _compute_percentile_bounds(
        [roughness_path],
        NORMALIZATION_LOW_PERCENTILE,
        NORMALIZATION_HIGH_PERCENTILE,
    )

    (
        passable_count,
        total_count,
        slope_min,
        slope_max,
        slope_sum,
        slope_n,
    ) = _write_cost_and_summarize(
        slope_path,
        roughness_path,
        cost_path,
        meta_f32,
        slope_low,
        slope_high,
        roughness_low,
        roughness_high,
        row_chunk,
    )

    return TerrainSummary(
        rows=rows,
        cols=cols,
        slope_path=slope_path,
        roughness_path=roughness_path,
        cost_path=cost_path,
        passable_fraction=passable_count / total_count,
        slope_min=slope_min,
        slope_max=slope_max,
        slope_mean=(
            slope_sum / slope_n
            if slope_n > 0
            else float("nan")
        ),
    )


def _compute_percentile_bounds(
    paths: list[Path],
    low_pct: float,
    high_pct: float,
    row_chunk: int = 512,
) -> tuple[float, float]:
    """
    Estimate percentile bounds from a deterministic random sample.

    Call separately for each physical quantity. Do not pool slope and
    roughness values.
    """
    if not 0 <= low_pct < high_pct <= 100:
        raise ValueError("Invalid percentile range")

    rng = np.random.default_rng(PERCENTILE_RANDOM_SEED)
    samples = []

    for path in paths:
        with rasterio.open(path) as src:
            for row_start in range(0, src.height, row_chunk):
                chunk_rows = min(row_chunk, src.height - row_start)

                block = src.read(
                    1,
                    window=Window(
                        0,
                        row_start,
                        src.width,
                        chunk_rows,
                    ),
                )

                valid = block[~np.isnan(block)]

                if valid.size == 0:
                    continue

                if valid.size > PERCENTILE_SAMPLE_SIZE:
                    indices = rng.choice(
                        valid.size,
                        PERCENTILE_SAMPLE_SIZE,
                        replace=False,
                    )
                    valid = valid[indices]

                samples.append(valid)

    if not samples:
        raise ValueError(
            f"No valid pixels found in raster(s): {paths}"
        )

    all_samples = np.concatenate(samples)

    low = float(np.percentile(all_samples, low_pct))
    high = float(np.percentile(all_samples, high_pct))

    if high <= low:
        high = low + 1.0

    return low, high


def _write_cost_and_summarize(
    slope_path: Path,
    roughness_path: Path,
    cost_path: Path,
    meta_f32: dict,
    slope_low: float,
    slope_high: float,
    roughness_low: float,
    roughness_high: float,
    row_chunk: int,
):
    """
    Independently normalize slope and roughness, then combine them.

    NaN means invalid or impassable. Stage 4 must treat NaN as
    non-traversable.
    """
    passable_count = 0
    total_count = 0

    slope_min = float("inf")
    slope_max = float("-inf")
    slope_sum = 0.0
    slope_n = 0

    with (
        rasterio.open(slope_path) as slope_src,
        rasterio.open(roughness_path) as rough_src,
        rasterio.open(cost_path, "w", **meta_f32) as cost_dst,
    ):
        for row_start in range(0, slope_src.height, row_chunk):
            chunk_rows = min(row_chunk, slope_src.height - row_start)

            window = Window(
                0,
                row_start,
                slope_src.width,
                chunk_rows,
            )

            slope = slope_src.read(1, window=window)
            roughness = rough_src.read(1, window=window)

            invalid = np.isnan(slope) | np.isnan(roughness)

            slope_clipped = np.clip(
                slope,
                slope_low,
                slope_high,
            )

            slope_norm = (
                (slope_clipped - slope_low)
                / (slope_high - slope_low)
            )

            roughness_clipped = np.clip(
                roughness,
                roughness_low,
                roughness_high,
            )

            roughness_norm = (
                (roughness_clipped - roughness_low)
                / (roughness_high - roughness_low)
            )

            cost = (
                SLOPE_WEIGHT * slope_norm
                + ROUGHNESS_WEIGHT * roughness_norm
            )

            cost = np.clip(cost, 0.0, 1.0)

            cost[invalid] = np.nan
            cost[slope > IMPASSABLE_SLOPE_DEGREES] = np.nan

            cost_dst.write(
                cost.astype(np.float32),
                1,
                window=window,
            )

            passable_count += int(np.sum(~np.isnan(cost)))
            total_count += cost.size

            valid_slope = slope[~np.isnan(slope)]

            if valid_slope.size:
                slope_min = min(
                    slope_min,
                    float(valid_slope.min()),
                )

                slope_max = max(
                    slope_max,
                    float(valid_slope.max()),
                )

                slope_sum += float(valid_slope.sum())
                slope_n += valid_slope.size

            del (
                slope,
                roughness,
                invalid,
                slope_clipped,
                roughness_clipped,
                slope_norm,
                roughness_norm,
                cost,
            )

    return (
        passable_count,
        total_count,
        slope_min,
        slope_max,
        slope_sum,
        slope_n,
    )


if __name__ == "__main__":
    summary = run_stage_3(
        Path("data-pipeline/raw/dem.tif"),
        Path("data-pipeline/output"),
    )

    print(f"Grid: {summary.rows} x {summary.cols}")

    print(
        f"Slope range: {summary.slope_min:.1f}° to "
        f"{summary.slope_max:.1f}°, "
        f"mean {summary.slope_mean:.1f}°"
    )

    print(f"Passable fraction: {summary.passable_fraction:.1%}")
