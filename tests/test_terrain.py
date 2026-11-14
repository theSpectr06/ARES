"""
Stage 3 tests: slope, vectorized detrended roughness, boundary handling,
numerical stability, percentile normalization, and modelling assumptions.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ares_pipeline.terrain import (
    IMPASSABLE_SLOPE_DEGREES,
    ROUGHNESS_WEIGHT,
    SLOPE_WEIGHT,
    _compute_percentile_bounds,
    _compute_roughness_strip_vectorized,
    _compute_slope_strip,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def slow_roughness_reference(
    elevation: np.ndarray,
    valid: np.ndarray,
    window: int = 5,
    min_valid_fraction: float = 0.6,
) -> np.ndarray:
    """
    Reference implementation using np.linalg.lstsq.

    This deliberately uses a simple per-cell implementation so that
    the vectorized implementation can be checked against it.
    """
    rows, cols = elevation.shape
    half = window // 2
    min_valid = max(4, int(window * window * min_valid_fraction))

    result = np.full((rows, cols), np.nan, dtype=np.float64)

    for r in range(half, rows - half):
        for c in range(half, cols - half):
            block_valid = valid[
                r - half:r + half + 1,
                c - half:c + half + 1,
            ]

            block_elevation = elevation[
                r - half:r + half + 1,
                c - half:c + half + 1,
            ]

            if np.count_nonzero(block_valid) < min_valid:
                continue

            yy, xx = np.mgrid[0:window, 0:window]

            x = xx[block_valid].astype(np.float64)
            y = yy[block_valid].astype(np.float64)
            z = block_elevation[block_valid].astype(np.float64)

            # Centre elevations before fitting to reduce numerical
            # problems in this independent reference implementation too.
            z = z - z.mean()

            design = np.column_stack([
                x,
                y,
                np.ones_like(x),
            ])

            coeffs, _, _, _ = np.linalg.lstsq(
                design,
                z,
                rcond=None,
            )

            residuals = z - design @ coeffs
            result[r, c] = np.std(residuals)

    return result


def make_test_raster(path, data):
    """Write a small float32 GeoTIFF for percentile tests."""
    data = np.asarray(data, dtype=np.float32)
    rows, cols = data.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, rows, 1.0, 1.0),
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)


# ---------------------------------------------------------------------
# Slope tests
# ---------------------------------------------------------------------

def test_flat_valid_grid_has_zero_slope():
    elevation = np.full((10, 10), 50.0)
    valid = np.ones((10, 10), dtype=bool)

    slope = _compute_slope_strip(
        elevation,
        valid,
        1.0,
        1.0,
        local_start=0,
        local_end=10,
    )

    np.testing.assert_allclose(
        slope[2:-2, 2:-2],
        0.0,
        atol=1e-6,
    )


def test_constant_tilt_produces_expected_slope_angle():
    rows, cols = 12, 12

    elevation = np.tile(
        np.arange(cols, dtype=np.float64),
        (rows, 1),
    )

    valid = np.ones((rows, cols), dtype=bool)

    slope = _compute_slope_strip(
        elevation,
        valid,
        1.0,
        1.0,
        local_start=0,
        local_end=rows,
    )

    np.testing.assert_allclose(
        slope[2:-2, 2:-2],
        45.0,
        atol=1e-6,
    )


def test_slope_does_not_wrap_across_grid_edges():
    rows, cols = 10, 10

    elevation = np.zeros((rows, cols))
    elevation[:, -1] = 1000.0

    valid = np.ones((rows, cols), dtype=bool)

    slope = _compute_slope_strip(
        elevation,
        valid,
        1.0,
        1.0,
        local_start=0,
        local_end=rows,
    )

    assert np.all(np.isnan(slope[:, 0]))
    assert np.all(np.isnan(slope[:, -1]))


def test_slope_is_nan_adjacent_to_internal_gap():
    elevation = np.full((9, 9), 10.0)
    valid = np.ones((9, 9), dtype=bool)

    valid[4, 4] = False
    elevation[4, 4] = np.nan

    slope = _compute_slope_strip(
        elevation,
        valid,
        1.0,
        1.0,
        local_start=0,
        local_end=9,
    )

    # The invalid cell and cells requiring its elevation are invalid.
    assert np.isnan(slope[4, 4])
    assert np.isnan(slope[3, 4])
    assert np.isnan(slope[5, 4])
    assert np.isnan(slope[4, 3])
    assert np.isnan(slope[4, 5])

    # A sufficiently distant interior cell remains valid.
    assert np.isfinite(slope[2, 2])



# ---------------------------------------------------------------------
# Basic roughness tests
# ---------------------------------------------------------------------

def test_flat_grid_has_near_zero_vectorized_roughness():
    elevation = np.full((9, 9), 25.0)
    valid = np.ones((9, 9), dtype=bool)

    roughness = _compute_roughness_strip_vectorized(
        elevation,
        valid,
        window=5,
        local_start=0,
        local_end=9,
    )

    np.testing.assert_allclose(
        roughness[2:-2, 2:-2],
        0.0,
        atol=1e-6,
    )


def test_smooth_ramp_has_near_zero_vectorized_roughness():
    rows, cols = 11, 11
    yy, xx = np.mgrid[0:rows, 0:cols]

    elevation = xx * 2.0 + yy * 0.5
    valid = np.ones((rows, cols), dtype=bool)

    roughness = _compute_roughness_strip_vectorized(
        elevation,
        valid,
        window=5,
        local_start=0,
        local_end=rows,
    )

    np.testing.assert_allclose(
        roughness[3:-3, 3:-3],
        0.0,
        atol=1e-3,
    )


def test_bumpy_surface_on_a_tilt_still_registers_as_rough():
    rows, cols = 11, 11
    yy, xx = np.mgrid[0:rows, 0:cols]

    tilt = xx * 2.0 + yy * 0.5

    rng = np.random.default_rng(1)
    bumps = rng.uniform(-3.0, 3.0, size=(rows, cols))

    elevation = tilt + bumps
    valid = np.ones((rows, cols), dtype=bool)

    roughness = _compute_roughness_strip_vectorized(
        elevation,
        valid,
        window=5,
        local_start=0,
        local_end=rows,
    )

    assert np.nanmean(roughness[3:-3, 3:-3]) > 0.5


def test_roughness_is_nan_when_neighborhood_mostly_invalid():
    rows, cols = 7, 7

    elevation = np.zeros((rows, cols))
    valid = np.zeros((rows, cols), dtype=bool)

    valid[3, 3] = True

    roughness = _compute_roughness_strip_vectorized(
        elevation,
        valid,
        window=5,
        local_start=0,
        local_end=rows,
    )

    assert np.isnan(roughness[3, 3])


# ---------------------------------------------------------------------
# Actual raster boundary tests
# ---------------------------------------------------------------------

def test_actual_raster_boundaries_have_nan_roughness():
    """
    The outer two rows and columns cannot have complete 5x5 windows.
    Interior cells should remain valid.
    """
    elevation = np.full((12, 12), 10.0)
    valid = np.ones((12, 12), dtype=bool)

    roughness = _compute_roughness_strip_vectorized(
        elevation,
        valid,
        window=5,
        local_start=0,
        local_end=12,
        touches_top_boundary=True,
        touches_bottom_boundary=True,
    )

    assert np.all(np.isnan(roughness[:2, :]))
    assert np.all(np.isnan(roughness[-2:, :]))
    assert np.all(np.isnan(roughness[:, :2]))
    assert np.all(np.isnan(roughness[:, -2:]))

    assert np.all(np.isfinite(roughness[2:-2, 2:-2]))


def test_internal_nodata_gap_uses_sixty_percent_threshold():
    """
    Exactly 15 valid cells in a 5x5 neighbourhood must pass.
    Fourteen valid cells must fail.
    """
    rows, cols = 11, 11

    elevation = np.full((rows, cols), 10.0)
    valid = np.ones((rows, cols), dtype=bool)

    # Keep exactly 15 valid cells in the centre's 5x5 neighbourhood.
    valid[3:8, 3:8] = False

    positions = [
        (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
        (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
        (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
    ]

    for r, c in positions:
        valid[r, c] = True

    elevation[~valid] = np.nan

    roughness = _compute_roughness_strip_vectorized(
        elevation,
        valid,
        window=5,
        local_start=0,
        local_end=rows,
    )

    assert np.isfinite(roughness[5, 5])

    # Remove one more valid cell: 14/25 is below 60%.
    valid[5, 7] = False
    elevation[5, 7] = np.nan

    roughness_after = _compute_roughness_strip_vectorized(
        elevation,
        valid,
        window=5,
        local_start=0,
        local_end=rows,
    )

    assert np.isnan(roughness_after[5, 5])


# ---------------------------------------------------------------------
# Artificial strip / halo tests
# ---------------------------------------------------------------------

def test_interior_strip_matches_whole_grid_with_halo():
    rows, cols = 20, 14

    rng = np.random.default_rng(3)
    elevation = rng.uniform(0, 30, size=(rows, cols))
    valid = np.ones((rows, cols), dtype=bool)

    whole = _compute_roughness_strip_vectorized(
        elevation,
        valid,
        window=5,
        local_start=0,
        local_end=rows,
    )

    # Output rows [6:12], with two halo rows on either side.
    strip_start = 6
    strip_end = 12
    halo = 2

    padded_elevation = elevation[
        strip_start - halo:strip_end + halo,
        :,
    ]

    padded_valid = valid[
        strip_start - halo:strip_end + halo,
        :,
    ]

    strip = _compute_roughness_strip_vectorized(
        padded_elevation,
        padded_valid,
        window=5,
        local_start=halo,
        local_end=halo + (strip_end - strip_start),
        touches_top_boundary=False,
        touches_bottom_boundary=False,
    )

    np.testing.assert_allclose(
        strip,
        whole[strip_start:strip_end, :],
        equal_nan=True,
        atol=1e-6,
    )


def test_top_boundary_strip_invalidates_actual_top_rows_only():
    rows, cols = 16, 12

    rng = np.random.default_rng(4)
    elevation = rng.uniform(0, 30, size=(rows, cols))
    valid = np.ones((rows, cols), dtype=bool)

    # First output strip: actual rows [0:6], with two rows of halo below.
    strip_end = 6
    halo = 2

    padded_elevation = elevation[:strip_end + halo, :]
    padded_valid = valid[:strip_end + halo, :]

    strip = _compute_roughness_strip_vectorized(
        padded_elevation,
        padded_valid,
        window=5,
        local_start=0,
        local_end=strip_end,
        touches_top_boundary=True,
        touches_bottom_boundary=False,
    )

    assert np.all(np.isnan(strip[:2, :]))
    assert np.all(np.isfinite(strip[2:, 2:-2]))


# ---------------------------------------------------------------------
# Vectorized implementation versus least-squares reference
# ---------------------------------------------------------------------

def test_vectorized_roughness_matches_slow_reference_on_random_grid():
    rng = np.random.default_rng(0)

    rows, cols = 15, 15
    elevation = rng.uniform(0, 10, size=(rows, cols))
    valid = np.ones((rows, cols), dtype=bool)

    valid[7, 7] = False
    elevation[7, 7] = np.nan

    fast = _compute_roughness_strip_vectorized(
        elevation,
        valid,
        window=5,
        local_start=0,
        local_end=rows,
    )

    slow = slow_roughness_reference(
        elevation,
        valid,
        window=5,
    )

    np.testing.assert_allclose(
        fast,
        slow,
        equal_nan=True,
        atol=1e-6,
        rtol=1e-6,
    )


def test_large_elevation_offset_matches_least_squares_reference():
    """
    Small bumps on a plane with a large absolute elevation offset.

    The comparison checks that roughness remains accurate when the
    absolute elevation is much larger than the residual variation.
    """
    rows, cols = 15, 15
    yy, xx = np.mgrid[0:rows, 0:cols]

    rng = np.random.default_rng(12)

    bumps = rng.uniform(-0.05, 0.05, size=(rows, cols))

    elevation = (
        3_000_000.0
        + 2.0 * xx
        + 0.5 * yy
        + bumps
    )

    valid = np.ones((rows, cols), dtype=bool)

    fast = _compute_roughness_strip_vectorized(
        elevation,
        valid,
        window=5,
        local_start=0,
        local_end=rows,
    )

    slow = slow_roughness_reference(
        elevation,
        valid,
        window=5,
    )

    np.testing.assert_allclose(
        fast[2:-2, 2:-2],
        slow[2:-2, 2:-2],
        atol=1e-8,
        rtol=1e-5,
    )


def test_roughness_is_invariant_to_constant_elevation_offset():
    """
    Adding a constant vertical offset must not change detrended roughness.
    """
    rows, cols = 15, 15
    yy, xx = np.mgrid[0:rows, 0:cols]

    rng = np.random.default_rng(19)
    bumps = rng.uniform(-0.05, 0.05, size=(rows, cols))

    base = 2.0 * xx + 0.5 * yy + bumps
    shifted = base + 3_000_000.0

    valid = np.ones((rows, cols), dtype=bool)

    base_roughness = _compute_roughness_strip_vectorized(
        base,
        valid,
        window=5,
        local_start=0,
        local_end=rows,
    )

    shifted_roughness = _compute_roughness_strip_vectorized(
        shifted,
        valid,
        window=5,
        local_start=0,
        local_end=rows,
    )

    np.testing.assert_allclose(
        shifted_roughness[2:-2, 2:-2],
        base_roughness[2:-2, 2:-2],
        atol=1e-8,
        rtol=1e-5,
    )


# ---------------------------------------------------------------------
# Percentile normalization tests
# ---------------------------------------------------------------------

def test_percentile_bounds_are_computed_separately(tmp_path):
    slope_path = tmp_path / "slope.tif"
    roughness_path = tmp_path / "roughness.tif"

    make_test_raster(
        slope_path,
        np.full((20, 20), 20.0),
    )

    make_test_raster(
        roughness_path,
        np.full((20, 20), 0.5),
    )

    slope_low, slope_high = _compute_percentile_bounds(
        [slope_path],
        2.0,
        98.0,
    )

    rough_low, rough_high = _compute_percentile_bounds(
        [roughness_path],
        2.0,
        98.0,
    )

    assert slope_low == pytest.approx(20.0)
    assert rough_low == pytest.approx(0.5)

    assert slope_high != pytest.approx(rough_high)


def test_percentile_bounds_resist_single_extreme_outlier(tmp_path):
    data = np.full((20, 20), 5.0, dtype=np.float32)
    data[0, 0] = 10_000.0

    path = tmp_path / "values.tif"
    make_test_raster(path, data)

    low, high = _compute_percentile_bounds(
        [path],
        low_pct=2.0,
        high_pct=98.0,
    )

    assert low == pytest.approx(5.0)
    assert high < 100.0


# ---------------------------------------------------------------------
# Configuration assumptions
# ---------------------------------------------------------------------

def test_cost_weights_and_cutoff_are_named_assumptions():
    assert SLOPE_WEIGHT + ROUGHNESS_WEIGHT == pytest.approx(1.0)
    assert IMPASSABLE_SLOPE_DEGREES > 0
