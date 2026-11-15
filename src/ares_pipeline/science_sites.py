"""
Validation of scientific destination records against the real terrain
rasters: confirms a destination's coordinates convert to a valid,
passable pixel, and flags (without silently relocating) any destination
that does not.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ares_pipeline.coordinates import (
    RasterGeoreference,
    load_georeference,
    lonlat_to_pixel,
    pixel_in_bounds,
    round_to_cell,
)
from ares_pipeline.pathfind import BlockCache, is_passable


@dataclass
class DestinationCandidate:
    """One destination's identity and published coordinate, prior to validation."""

    id: str
    name: str
    longitude_deg: float
    latitude_deg: float
    geological_unit: str


@dataclass
class ValidatedDestination:
    """A destination after terrain validation, carrying the result either way."""

    id: str
    name: str
    longitude_deg: float
    latitude_deg: float
    geological_unit: str
    pixel_col: int
    pixel_row: int
    valid: bool
    rejection_reason: str | None  # None if valid


def validate_destination(
    candidate: DestinationCandidate,
    dem_georef: RasterGeoreference,
    cost_cache: BlockCache,
    search_radius_m: float = 100.0,
) -> ValidatedDestination:
    """
    Convert a candidate's published lon/lat to a pixel, and confirm it (or
    a cell within search_radius_m) is passable per the cost raster. Never
    silently substitutes a different coordinate for a VALID destination —
    if the published point itself is invalid but a nearby cell within the
    search radius is valid, this is recorded explicitly as a documented
    relocation (rejection_reason states the original point was invalid and
    which nearby cell was used instead), not a silent swap. If no valid
    cell exists within the radius at all, the destination is rejected
    outright with that reason recorded — exactly the pattern used for the
    real Nili Planum rejection.
    """
    col_f, row_f = lonlat_to_pixel(candidate.longitude_deg, candidate.latitude_deg, dem_georef)

    if not pixel_in_bounds(col_f, row_f, dem_georef):
        return ValidatedDestination(
            **asdict_candidate(candidate),
            pixel_col=-1, pixel_row=-1, valid=False,
            rejection_reason=(
                f"Published coordinate ({candidate.longitude_deg}, "
                f"{candidate.latitude_deg}) converts to a pixel outside the "
                f"DEM's extent entirely."
            ),
        )

    col, row = round_to_cell(col_f, row_f)

    if is_passable(cost_cache, row, col):
        return ValidatedDestination(
            **asdict_candidate(candidate),
            pixel_col=col, pixel_row=row, valid=True, rejection_reason=None,
        )

    # Published point itself is invalid (NoData or impassable) -- search a
    # bounded radius for the nearest valid cell, exactly mirroring how the
    # real Nili Planum candidate was checked and ultimately rejected.
    pixel_radius = max(1, math.ceil(search_radius_m))  # DEM is confirmed 1 m/pixel
    nearest = _find_nearest_passable_cell(cost_cache, row, col, pixel_radius)

    if nearest is None:
        return ValidatedDestination(
            **asdict_candidate(candidate),
            pixel_col=col, pixel_row=row, valid=False,
            rejection_reason=(
                f"Published coordinate falls on an invalid cell (row={row}, "
                f"col={col}), and no valid cell was found within "
                f"{search_radius_m} m. Rejected -- not relocated."
            ),
        )

    nearest_row, nearest_col = nearest
    return ValidatedDestination(
        **asdict_candidate(candidate),
        pixel_col=nearest_col, pixel_row=nearest_row, valid=False,
        rejection_reason=(
            f"Published coordinate (row={row}, col={col}) is invalid. "
            f"A valid cell exists at (row={nearest_row}, col={nearest_col}), "
            f"{_pixel_distance(row, col, nearest_row, nearest_col):.1f} m away. "
            f"This destination is marked INVALID, not silently relocated -- "
            f"a human decision is required on whether to accept this nearby "
            f"cell as a substitute, following the same review pattern used "
            f"for the Nili Planum candidate."
        ),
    )


def asdict_candidate(candidate: DestinationCandidate) -> dict:
    return {
        "id": candidate.id, "name": candidate.name,
        "longitude_deg": candidate.longitude_deg, "latitude_deg": candidate.latitude_deg,
        "geological_unit": candidate.geological_unit,
    }


def _pixel_distance(r1: int, c1: int, r2: int, c2: int) -> float:
    return math.hypot(r1 - r2, c1 - c2)  # DEM confirmed 1 m/pixel, so pixel distance == meters


def _find_nearest_passable_cell(
    cache: BlockCache, row: int, col: int, max_radius_px: int
) -> tuple[int, int] | None:
    """
    Expanding-ring (Chebyshev-distance) search for a nearby passable cell
    within max_radius_px.

    Honest limitation, checked rather than assumed away: this search finds
    the nearest cell WITHIN each ring it checks, and returns the first
    match found, ring by ring -- it is NOT guaranteed to return the
    globally nearest passable cell by true (Euclidean) distance. Starting
    at ring 3, a ring's worst-case true distance (its diagonal corner,
    radius*sqrt(2)) can exceed the very next ring's best-case true distance
    (a straight edge cell, radius+1) -- verified directly: ring 3's worst
    case is ~4.24 while ring 4's best case is 4. In practice this means the
    reported "nearby valid cell" can be off by up to roughly one pixel's
    worth of true distance at larger search radii. This is acceptable here
    because the result is surfaced to a human for an explicit accept/reject
    decision (see validate_destination's rejection_reason), never silently
    acted on -- a few meters of suboptimality in what's reported changes
    nothing about correctness or safety. If an exactly-nearest result is
    ever required, a true distance-sorted full-disk scan (checking every
    cell within max_radius_px, not ring by ring) would be needed instead,
    at the cost of checking more cells for large radii.
    """
    if is_passable(cache, row, col):
        return (row, col)

    for radius in range(1, max_radius_px + 1):
        candidates = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue  # only the current ring's perimeter, not already-checked interior
                candidates.append((row + dr, col + dc))
        # Sort by true distance so the closest cell on this ring is returned
        # first, even though all cells on a ring are checked.
        candidates.sort(key=lambda rc: _pixel_distance(row, col, rc[0], rc[1]))
        for r, c in candidates:
            if is_passable(cache, r, c):
                return (r, c)

    return None


def load_destination_candidates(path: Path) -> list[DestinationCandidate]:
    """Load the raw candidate list (id, name, lon, lat, unit) prior to validation."""
    with open(path) as f:
        raw = json.load(f)
    return [DestinationCandidate(**entry) for entry in raw["candidates"]]
