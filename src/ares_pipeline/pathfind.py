"""
Stage 4: A* pathfinding over the terrain cost raster from Stage 3.

Design summary (see build-guide-stage4-onwards.md for full reasoning):
- 8-directional movement, cardinal=1m, diagonal=sqrt(2)m (real distances on
  the confirmed 1m/pixel grid).
- Step cost = distance * (1 + terrain_weight * average_cell_cost).
- NaN cost cells are excluded from the graph entirely (impassable, not
  high-cost).
- Diagonal moves are blocked if either flanking orthogonal cell is
  impassable (no corner-cutting through blocked terrain).
- Octile distance heuristic -- admissible under this specific cost model
  because every real step costs at least its raw physical distance.
- Route distance (real meters) and accumulated traversal cost (modeled
  difficulty, in metre-equivalent units, NOT actual distance) are reported
  SEPARATELY and never conflated.
- Cost raster access goes through a bounded LRU block cache matching the
  file's native 256x256 tiling -- never a full in-memory load, never
  per-pixel reads.
"""
import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import rasterio

# --- Named, configurable modeling assumptions ------------------------------
TERRAIN_WEIGHT = 1.0  # how strongly terrain cost amplifies step distance;
                        # 0.0 would reduce to pure shortest-distance routing
BLOCK_SIZE = 256       # must match the cost raster's own blockxsize/blockysize
MAX_CACHED_BLOCKS = 64  # ~16.8 MB at float32; see reasoning in the build guide
# ---------------------------------------------------------------------------

CARDINAL_DISTANCE = 1.0
DIAGONAL_DISTANCE = math.sqrt(2.0)

# The 8 movement directions, as (row_delta, col_delta, distance).
NEIGHBOR_OFFSETS = [
    (-1, 0, CARDINAL_DISTANCE), (1, 0, CARDINAL_DISTANCE),
    (0, -1, CARDINAL_DISTANCE), (0, 1, CARDINAL_DISTANCE),
    (-1, -1, DIAGONAL_DISTANCE), (-1, 1, DIAGONAL_DISTANCE),
    (1, -1, DIAGONAL_DISTANCE), (1, 1, DIAGONAL_DISTANCE),
]


class BlockCache:
    """
    Bounded LRU cache of 256x256 NumPy blocks read from a single-band
    raster. Matches the file's native block size exactly, so one cached
    block corresponds to exactly one real on-disk tile read.
    """

    def __init__(self, raster_path: Path, block_size: int = BLOCK_SIZE,
                 max_blocks: int = MAX_CACHED_BLOCKS):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if max_blocks <= 0:
            raise ValueError(f"max_blocks must be positive, got {max_blocks}")

        self._dataset = rasterio.open(raster_path)

        if self._dataset.count != 1:
            self._dataset.close()
            raise ValueError(
                f"Expected a single-band cost raster, "
                f"found {self._dataset.count} bands."
            )

        # Verify the configured block_size actually matches the raster's
        # real on-disk tiling, rather than assuming it silently -- a
        # mismatch here means the cache would not correspond to real tile
        # boundaries, defeating the whole point of the cache.
        actual_block_height, actual_block_width = self._dataset.block_shapes[0]
        if (actual_block_height, actual_block_width) != (block_size, block_size):
            actual_shape = (actual_block_height, actual_block_width)
            self._dataset.close()
            raise ValueError(
                f"Configured block_size={block_size} does not match "
                f"{raster_path.name}'s actual on-disk block shape "
                f"{actual_shape}. Fix block_size to match the real tiling."
            )

        self.block_size = block_size
        self.max_blocks = max_blocks
        self.rows = self._dataset.height
        self.cols = self._dataset.width
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def close(self):
        self._dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _block_key(self, row: int, col: int) -> tuple[int, int]:
        return (row // self.block_size, col // self.block_size)

    def _load_block(self, block_key: tuple[int, int]) -> np.ndarray:
        block_row, block_col = block_key
        row_start = block_row * self.block_size
        col_start = block_col * self.block_size
        height = min(self.block_size, self.rows - row_start)
        width = min(self.block_size, self.cols - col_start)

        window = rasterio.windows.Window(
            col_start, row_start, width, height
        )

        block = self._dataset.read(1, window=window)

        # NaN represents impassable terrain.
        # Every other value must be a valid normalized terrain cost.
        valid_values = block[~np.isnan(block)]
        
        if not np.all(np.isfinite(valid_values)):
            raise ValueError(
                f"Cost raster contains infinite values in block {block_key}."
            )
        
        if np.any((valid_values < 0.0) | (valid_values > 1.0)):
            raise ValueError(
                f"Cost raster contains values outside [0, 1] "
                f"in block {block_key}."
            )

        return block

    def get(self, row: int, col: int) -> float:
        """Return the cost value at (row, col), using cached blocks where possible."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return float("nan")  # outside the grid is treated the same as impassable

        key = self._block_key(row, col)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            block = self._load_block(key)
            self._cache[key] = block
            self._cache.move_to_end(key)
            if len(self._cache) > self.max_blocks:
                self._cache.popitem(last=False)  # evict least-recently-used

        block = self._cache[key]
        block_row, block_col = key
        local_row = row - block_row * self.block_size
        local_col = col - block_col * self.block_size
        return float(block[local_row, local_col])


def is_passable(cache: BlockCache, row: int, col: int) -> bool:
    value = cache.get(row, col)
    return not math.isnan(value)


def diagonal_move_blocked_by_corner(
    cache: BlockCache, row: int, col: int, dr: int, dc: int
) -> bool:
    """
    For a diagonal move from (row, col) to (row+dr, col+dc), check the two
    orthogonal 'flanking' cells. If either is impassable, the move would
    cut a corner through blocked terrain and is disallowed -- verified on
    hand-built grids before being used here (see build guide).
    """
    corner1_passable = is_passable(cache, row + dr, col)
    corner2_passable = is_passable(cache, row, col + dc)
    return not (corner1_passable and corner2_passable)


def octile_heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """
    Admissible heuristic for this movement/cost model: the cheapest
    possible distance between two grid points using 8-directional moves
    with cardinal=1, diagonal=sqrt(2) costs, ignoring terrain. Since real
    step costs are always >= their raw distance under the cost formula
    used here, this never overestimates the true remaining cost.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return CARDINAL_DISTANCE * (dr + dc) + (DIAGONAL_DISTANCE - 2 * CARDINAL_DISTANCE) * min(dr, dc)


@dataclass
class RouteResult:
    path: list[tuple[int, int]] | None  # None if unreachable
    geometric_distance_m: float         # real physical distance, meters
    accumulated_cost: float             # modeled terrain-weighted cost, in
                                         # metre-equivalent units — NOT real
                                         # walking distance, and not unitless
                                         # either (see build guide Stage 4
                                         # "Why the accumulated cost is not
                                         # unitless" for the dimensional
                                         # reasoning)
    cells_expanded: int                 # for diagnostics / performance tracking


def find_route(
    cache: BlockCache,
    start: tuple[int, int],
    goal: tuple[int, int],
    terrain_weight: float = TERRAIN_WEIGHT,
) -> RouteResult:
    """
    A* search from start to goal over the terrain cost raster. Returns a
    RouteResult with path=None if the goal is unreachable (including the
    case where start or goal themselves are impassable or out of bounds).

    terrain_weight must be finite and non-negative: a negative weight
    would let terrain REDUCE a step's cost below its raw physical
    distance, which breaks the admissibility proof for the octile
    heuristic -- that proof specifically depends on every real step
    costing at least its raw distance.
    """
    if not math.isfinite(terrain_weight) or terrain_weight < 0:
        raise ValueError(
            f"terrain_weight must be finite and >= 0, got {terrain_weight}. "
            f"A negative value would break the heuristic's admissibility "
            f"guarantee (see build guide Stage 4)."
        )

    if not is_passable(cache, *start):
        return RouteResult(path=None, geometric_distance_m=0.0, accumulated_cost=0.0, cells_expanded=0)
    if not is_passable(cache, *goal):
        return RouteResult(path=None, geometric_distance_m=0.0, accumulated_cost=0.0, cells_expanded=0)

    open_set: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_cost: dict[tuple[int, int], float] = {start: 0.0}      # accumulated modeled cost
    g_distance: dict[tuple[int, int], float] = {start: 0.0}  # accumulated real distance
    visited: set[tuple[int, int]] = set()
    cells_expanded = 0
    started_at = time.perf_counter()
    progress_interval = 100_000

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        cells_expanded += 1

        if cells_expanded % progress_interval == 0:
            elapsed = time.perf_counter() - started_at
            print(
                f"A*: expanded={cells_expanded:,}, "
                f"open_set={len(open_set):,}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return RouteResult(
                path=path,
                geometric_distance_m=g_distance[goal],
                accumulated_cost=g_cost[goal],
                cells_expanded=cells_expanded,
            )

        r, c = current
        current_terrain_cost = cache.get(r, c)

        # Read each neighbour exactly once. Reuse these values for
        # passability, diagonal corner checks, and traversal costs.
        neighbor_costs = [
            cache.get(r + dr, c + dc)
            for dr, dc, _ in NEIGHBOR_OFFSETS
        ]

        for index, (dr, dc, step_distance) in enumerate(NEIGHBOR_OFFSETS):
            neighbor = (r + dr, c + dc)
            neighbor_terrain_cost = neighbor_costs[index]

            # NaN means impassable or outside the raster.
            if math.isnan(neighbor_terrain_cost):
                continue

            # Diagonal movement requires both flanking cardinal cells
            # to be passable.
            if dr != 0 and dc != 0:
                row_flank_index = 0 if dr < 0 else 1
                col_flank_index = 2 if dc < 0 else 3

                if (
                    math.isnan(neighbor_costs[row_flank_index])
                    or math.isnan(neighbor_costs[col_flank_index])
                ):
                    continue

            average_cell_cost = (
                current_terrain_cost + neighbor_terrain_cost
            ) / 2.0

            step_cost = step_distance * (
                1.0 + terrain_weight * average_cell_cost
            )

            tentative_g_cost = g_cost[current] + step_cost

            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                g_distance[neighbor] = (
                    g_distance[current] + step_distance
                )
                came_from[neighbor] = current

                f_score = (
                    tentative_g_cost + octile_heuristic(neighbor, goal)
                )
                heapq.heappush(open_set, (f_score, neighbor))

    # Open set exhausted without reaching goal -- genuinely unreachable
    # (e.g. goal is on an isolated island of passable cells).
    return RouteResult(path=None, geometric_distance_m=0.0, accumulated_cost=0.0, cells_expanded=cells_expanded)


if __name__ == "__main__":
    cost_path = Path("data-pipeline/output/cost.tif")
    with BlockCache(cost_path) as cache:
        # Placeholder start/end -- replace with real coordinates once a
        # specific route is chosen for the demo.
        start = (1000, 1000)
        goal = (1200, 1300)
        result = find_route(cache, start, goal)

        if result.path is None:
            print("No route found (unreachable, or start/goal impassable).")
        else:
            print(f"Route found: {len(result.path)} cells")
            print(f"Geometric distance: {result.geometric_distance_m:.1f} m")
            print(f"Accumulated modeled cost: {result.accumulated_cost:.2f} m-equivalent (NOT real walking distance)")
            print(f"Cells expanded during search: {result.cells_expanded}")
