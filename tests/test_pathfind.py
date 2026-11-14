"""
Synthetic-grid tests for A* pathfinding, the LRU block cache,
corner-cutting prevention, heuristic admissibility, distance/cost
separation, and input validation.

No real Mars data required.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ares_pipeline.pathfind import (
    BlockCache,
    diagonal_move_blocked_by_corner,
    find_route,
    octile_heuristic,
)


def make_tiled_cost_raster(path, cost_array, block_size=16):
    """
    GeoTIFF tile dimensions must be multiples of 16 -- rasterio/GDAL
    rejects anything else outright. Always use this for cache-behavior
    tests, with a synthetic raster large enough to be meaningfully tiled.
    """
    rows, cols = cost_array.shape
    transform = from_origin(0, rows, 1.0, 1.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=rows, width=cols, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
        tiled=True, blockxsize=block_size, blockysize=block_size,
    ) as dst:
        dst.write(cost_array.astype(np.float32), 1)


def make_untiled_cost_raster(path, cost_array):
    """
    For tiny grids used only in algorithm/logic tests (not cache-behavior
    tests), skip tiling entirely -- there is no tile-size constraint to
    satisfy, and BlockCache still works against an untiled file as long as
    its own block_size is configured to match what BlockCache expects to
    read in one window (untiled GeoTIFFs still serve arbitrary windows
    correctly; they just aren't physically stored in same-sized chunks).
    """
    rows, cols = cost_array.shape
    transform = from_origin(0, rows, 1.0, 1.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=rows, width=cols, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(cost_array.astype(np.float32), 1)


# --- heuristic correctness ---

def test_octile_distance_matches_known_values():
    assert octile_heuristic((0, 0), (0, 3)) == pytest.approx(3.0)
    assert octile_heuristic((0, 0), (3, 3)) == pytest.approx(3 * math.sqrt(2))
    assert octile_heuristic((0, 0), (2, 5)) == pytest.approx(2 * math.sqrt(2) + 3.0)


# --- flat terrain: cost should equal distance exactly ---

def test_flat_terrain_cost_equals_geometric_distance(tmp_path):
    cost = np.zeros((10, 10))
    path = tmp_path / "cost.tif"
    make_untiled_cost_raster(path, cost)
    # block_size must still match something BlockCache can validate
    # against; for an untiled raster, rasterio reports the whole-row as
    # the "block" shape by default for small rasters, so configure
    # BlockCache with block_size large enough to cover the grid in one
    # cached block, sidestepping the tiling question entirely for this
    # pure-algorithm test.
    with BlockCache(path, block_size=10) as cache:
        result = find_route(cache, (0, 0), (9, 9))
    assert result.path is not None
    assert result.accumulated_cost == pytest.approx(result.geometric_distance_m, rel=1e-6)


# --- terrain-aware routing: corrected with verified, not assumed, numbers ---

def test_route_prefers_lower_cost_over_shorter_distance(tmp_path):
    """
    CORRECTED from an earlier, mathematically wrong version of this test.

    The direct route along row 4 crosses one high-cost cell (0.95) at
    column 4. Under the real cost formula with terrain_weight=1.0, this
    direct route's accumulated cost is approximately 8.95 (6 ordinary
    cardinal steps at cost 1.0 each, plus 2 steps adjacent to the 0.95
    cell at cost ~1.475 each) -- cheaper than any detour (~11.31 for a
    pure-diagonal detour around the wall), so at terrain_weight=1.0 A*
    correctly crosses the high-cost cell. An earlier version of this test
    wrongly asserted the opposite at terrain_weight=1.0 and would have
    failed.

    This corrected version uses an explicitly exaggerated terrain_weight
    (5.0, well beyond the default of 1.0, and used only in this test) at
    which the detour genuinely becomes cheaper -- verified by hand
    calculation and by running both the direct and detour costs through
    the real formula before writing this assertion, not assumed.
    """
    rows, cols = 9, 9
    cost = np.zeros((rows, cols))
    cost[1:8, 4] = 0.95  # a column of high (but not impassable) cost
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)

    EXAGGERATED_TERRAIN_WEIGHT = 5.0  # local to this test only; verified
                                        # to make the detour cheaper than
                                        # the direct crossing for this
                                        # specific grid -- see build guide

    with BlockCache(path, block_size=16) as cache:
        detour_result = find_route(cache, (4, 0), (4, 8), terrain_weight=EXAGGERATED_TERRAIN_WEIGHT)

        # Force-compute the direct route's stats for comparison by
        # manually walking straight across row 4, using the SAME cost
        # formula the real search uses, rather than re-deriving a
        # separate "expected" number by hand only in the test.
        direct_distance = 0.0
        direct_cost = 0.0
        for col in range(cols - 1):
            c0 = cache.get(4, col)
            c1 = cache.get(4, col + 1)
            avg = (c0 + c1) / 2.0
            step_distance = 1.0
            direct_distance += step_distance
            direct_cost += step_distance * (1.0 + EXAGGERATED_TERRAIN_WEIGHT * avg)

    assert detour_result.path is not None
    # The actual proof: longer geometric distance, but lower accumulated
    # cost than the direct route -- this is what demonstrates the
    # algorithm is trading distance for lower modeled difficulty, not
    # merely "avoiding a region" by coincidence.
    assert detour_result.geometric_distance_m > direct_distance
    assert detour_result.accumulated_cost < direct_cost

    crossed_high_cost = any(cost[r, c] > 0.9 for r, c in detour_result.path)
    assert not crossed_high_cost


def test_default_terrain_weight_crosses_moderate_high_cost_region(tmp_path):
    """
    The complementary case to the test above, using the DEFAULT
    terrain_weight (1.0): confirms that for this same grid, the direct
    route genuinely is cheaper and IS taken, which is the correct,
    verified behavior at the default weight -- this is what the earlier,
    incorrect test got backwards.
    """
    rows, cols = 9, 9
    cost = np.zeros((rows, cols))
    cost[1:8, 4] = 0.95
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)

    with BlockCache(path, block_size=16) as cache:
        result = find_route(cache, (4, 0), (4, 8))  # default terrain_weight=1.0

    assert result.path is not None
    assert result.accumulated_cost == pytest.approx(8.95, abs=0.01)
    crossed_high_cost = any(cost[r, c] > 0.9 for r, c in result.path)
    assert crossed_high_cost  # at the default weight, crossing is genuinely cheaper


def test_heuristic_never_overestimates_actual_cost(tmp_path):
    rows, cols = 16, 16
    rng = np.random.default_rng(0)
    cost = rng.uniform(0, 0.8, size=(rows, cols))
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)

    with BlockCache(path, block_size=16) as cache:
        start, goal = (0, 0), (15, 15)
        result = find_route(cache, start, goal)

    assert result.path is not None
    h = octile_heuristic(start, goal)
    assert h <= result.accumulated_cost + 1e-9


# --- impassable cells and unreachable goals ---

def test_nan_cells_are_excluded_from_the_graph(tmp_path):
    cost = np.zeros((16, 16))
    cost[:, 8] = np.nan  # a solid impassable wall down one column, full height
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)
    with BlockCache(path, block_size=16) as cache:
        result = find_route(cache, (0, 0), (0, 15))
    assert result.path is None


def test_start_on_impassable_cell_returns_no_route(tmp_path):
    cost = np.zeros((16, 16))
    cost[0, 0] = np.nan
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)
    with BlockCache(path, block_size=16) as cache:
        result = find_route(cache, (0, 0), (15, 15))
    assert result.path is None


def test_goal_on_impassable_cell_returns_no_route(tmp_path):
    cost = np.zeros((16, 16))
    cost[15, 15] = np.nan
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)
    with BlockCache(path, block_size=16) as cache:
        result = find_route(cache, (0, 0), (15, 15))
    assert result.path is None


def test_out_of_bounds_coordinates_treated_as_impassable(tmp_path):
    cost = np.zeros((16, 16))
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)
    with BlockCache(path, block_size=16) as cache:
        assert math.isnan(cache.get(-1, 0))
        assert math.isnan(cache.get(0, 100))


# --- corner-cutting prevention: all three cases, plus a full-search proof ---

def test_diagonal_move_allowed_when_both_flanking_cells_passable(tmp_path):
    cost = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    path = tmp_path / "cost.tif"
    make_untiled_cost_raster(path, cost)
    with BlockCache(path, block_size=2) as cache:
        # Moving from (0,0) to (1,1): flanking cells (1,0) and (0,1), both passable.
        assert diagonal_move_blocked_by_corner(cache, 0, 0, 1, 1) is False


def test_diagonal_move_blocked_when_one_flanking_cell_impassable(tmp_path):
    """
    Block exactly one flank, (0,0), leaving the other flank (1,1) and the
    destination (0,1) passable -- confirms a single impassable flank is
    enough to block the move.
    """
    cost = np.array([
        [np.nan, 0.0],
        [0.0,    0.0],
    ])
    path = tmp_path / "cost.tif"
    make_untiled_cost_raster(path, cost)
    with BlockCache(path, block_size=2) as cache:
        # Moving from (1,0) to (0,1): dr=-1, dc=1. Flanks: (r+dr,c)=(0,0)
        # [impassable] and (r,c+dc)=(1,1) [passable]. One impassable flank
        # is enough to block.
        assert diagonal_move_blocked_by_corner(cache, 1, 0, -1, 1) is True


def test_diagonal_move_blocked_when_both_flanking_cells_impassable(tmp_path):
    cost = np.array([
        [np.nan, 0.0],
        [0.0,    np.nan],
    ])
    path = tmp_path / "cost.tif"
    make_untiled_cost_raster(path, cost)
    with BlockCache(path, block_size=2) as cache:
        # Moving from (0,1) to (1,0): dr=1, dc=-1. Flanks: (r+dr,c)=(1,1)
        # [impassable] and (r,c+dc)=(0,0) [impassable]. Both blocked.
        assert diagonal_move_blocked_by_corner(cache, 0, 1, 1, -1) is True


def test_full_search_returns_none_when_only_connection_is_a_blocked_corner(tmp_path):
    """
    End-to-end proof that find_route() itself -- not just the helper
    function in isolation -- correctly refuses to route through a
    diagonal whose corner is blocked. Grid (checkerboard of passable and
    impassable cells):
        .  X  .
        X  .  X
        .  X  .
    The only geometric path from (0,0) to (2,2) is through the center
    cell via diagonal moves, but every such diagonal's flanking cells are
    both impassable -- so no route should exist at all.
    """
    cost = np.array([
        [0.0,    np.nan, 0.0],
        [np.nan, 0.0,    np.nan],
        [0.0,    np.nan, 0.0],
    ])
    path = tmp_path / "cost.tif"
    make_untiled_cost_raster(path, cost)
    with BlockCache(path, block_size=3) as cache:
        result = find_route(cache, (0, 0), (2, 2))
    assert result.path is None


# --- block cache behavior: valid tile sizes, read counting, eviction ---

def test_block_cache_reads_each_block_from_disk_only_once(tmp_path):
    cost = np.zeros((32, 32))
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)

    with BlockCache(path, block_size=16, max_blocks=64) as cache:
        read_count = 0
        original_load = cache._load_block

        def counting_load(key):
            nonlocal read_count
            read_count += 1
            return original_load(key)

        cache._load_block = counting_load

        for _ in range(3):
            for r in range(16):
                for c in range(16):
                    cache.get(r, c)

        assert read_count == 1


def test_block_cache_evicts_least_recently_used_block(tmp_path):
    cost = np.zeros((64, 64))
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)

    with BlockCache(path, block_size=16, max_blocks=2) as cache:
        cache.get(0, 0)    # loads block (0,0)
        cache.get(0, 16)   # loads block (0,1)
        cache.get(0, 0)    # re-touches block (0,0) -- now most recently used
        cache.get(0, 32)   # loads block (0,2) -- cache full, must evict LRU

        assert (0, 1) not in cache._cache
        assert (0, 0) in cache._cache
        assert (0, 2) in cache._cache


def test_block_cache_never_exceeds_max_blocks(tmp_path):
    cost = np.zeros((64, 64))
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)

    with BlockCache(path, block_size=16, max_blocks=3) as cache:
        for r in range(0, 64, 16):
            for c in range(0, 64, 16):
                cache.get(r, c)
                assert len(cache._cache) <= 3


# --- input validation ---

def test_block_cache_rejects_non_positive_block_size(tmp_path):
    cost = np.zeros((16, 16))
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)
    with pytest.raises(ValueError, match="block_size must be positive"):
        BlockCache(path, block_size=0)


def test_block_cache_rejects_non_positive_max_blocks(tmp_path):
    cost = np.zeros((16, 16))
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)
    with pytest.raises(ValueError, match="max_blocks must be positive"):
        BlockCache(path, block_size=16, max_blocks=0)


def test_block_cache_rejects_block_size_not_matching_actual_raster_tiling(tmp_path):
    cost = np.zeros((32, 32))
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)  # actually tiled at 16x16
    with pytest.raises(ValueError, match="does not match"):
        BlockCache(path, block_size=32)  # deliberately wrong


def test_find_route_rejects_negative_terrain_weight(tmp_path):
    cost = np.zeros((16, 16))
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)
    with BlockCache(path, block_size=16) as cache:
        with pytest.raises(ValueError, match="terrain_weight must be finite and >= 0"):
            find_route(cache, (0, 0), (15, 15), terrain_weight=-1.0)


def test_find_route_rejects_non_finite_terrain_weight(tmp_path):
    cost = np.zeros((16, 16))
    path = tmp_path / "cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)
    with BlockCache(path, block_size=16) as cache:
        with pytest.raises(ValueError, match="terrain_weight must be finite and >= 0"):
            find_route(cache, (0, 0), (15, 15), terrain_weight=float("inf"))

def test_rejects_multiband_cost_raster(tmp_path):
    path = tmp_path / "multiband.tif"

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=2,
        dtype="float32",
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        dst.write(np.zeros((16, 16), dtype=np.float32), 1)
        dst.write(np.zeros((16, 16), dtype=np.float32), 2)

    with pytest.raises(ValueError, match="single-band"):
        BlockCache(path, block_size=16)


@pytest.mark.parametrize(
    "invalid_value",
    [float("inf"), float("-inf"), -0.1, 1.1],
)
def test_rejects_invalid_cost_values(tmp_path, invalid_value):
    cost = np.zeros((16, 16), dtype=np.float32)
    cost[5, 5] = invalid_value

    path = tmp_path / "invalid_cost.tif"
    make_tiled_cost_raster(path, cost, block_size=16)

    with BlockCache(path, block_size=16) as cache:
        with pytest.raises(ValueError, match="Cost raster"):
            cache.get(5, 5)
