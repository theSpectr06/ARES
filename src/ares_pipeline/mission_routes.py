"""
Stage 6 mission orchestration: Basecamp connectivity, per-destination
routing, detail-crop sizing, and web-asset generation.

This module reuses the existing Stage 4 pathfinding and export
implementations. It does not implement its own routing or cost model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ares_pipeline.coordinates import (
    load_georeference,
    lonlat_to_pixel,
    round_to_cell,
)
from ares_pipeline.export import export_route
from ares_pipeline.pathfind import BlockCache, RouteResult, find_route
from ares_pipeline.science_sites import (
    DestinationCandidate,
    ValidatedDestination,
    validate_destination,
)


BASECAMP = DestinationCandidate(
    id="candidate-4-basecamp",
    name="Mission Basecamp (Candidate 4)",
    longitude_deg=77.293427,
    latitude_deg=18.464211,
    geological_unit="N/A - staging point, not a science destination",
)


@dataclass
class DestinationRoute:
    """The outcome of routing from the Basecamp to one destination."""

    destination_id: str
    destination_name: str
    status: Literal["reachable", "unreachable", "terrain_invalid"]
    route: RouteResult | None
    gpkg_path: Path | None
    reason: str | None = None


def compute_basecamp_to_destination_routes(
    destinations: list[ValidatedDestination],
    basecamp: ValidatedDestination,
    cost_raster_path: Path,
    output_dir: Path,
) -> list[DestinationRoute]:
    """
    Route from the Basecamp to each destination using the existing
    Stage 4 A* implementation.

    Terrain-invalid destinations are recorded but not routed.
    Valid destinations with no available path are marked unreachable.
    Reachable routes are exported to individual GeoPackages.
    """
    if not basecamp.valid:
        raise ValueError(
            "Basecamp itself is not valid per the cost raster "
            f"(reason: {basecamp.rejection_reason}). "
            "Resolve this before routing."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[DestinationRoute] = []

    with BlockCache(cost_raster_path) as cache:
        for destination in destinations:

            if not destination.valid:
                results.append(
                    DestinationRoute(
                        destination_id=destination.id,
                        destination_name=destination.name,
                        status="terrain_invalid",
                        route=None,
                        gpkg_path=None,
                        reason=destination.rejection_reason,
                    )
                )
                continue

            start = (basecamp.pixel_row, basecamp.pixel_col)
            goal = (destination.pixel_row, destination.pixel_col)

            print(
                f"\nComputing route: {destination.id} "
                f"from {start} to {goal}",
                flush=True,
            )
            route = find_route(cache, start, goal)

            print(
                f"Completed {destination.id}: "
                f"{route.cells_expanded:,} cells expanded",
                flush=True,
            )
            if route.path is None:
                results.append(
                    DestinationRoute(
                        destination_id=destination.id,
                        destination_name=destination.name,
                        status="unreachable",
                        route=route,
                        gpkg_path=None,
                        reason="No traversable path found by A*.",
                    )
                )
                continue

            gpkg_path = output_dir / f"route-{destination.id}.gpkg"
            export_route(route, cost_raster_path, gpkg_path)

            results.append(
                DestinationRoute(
                    destination_id=destination.id,
                    destination_name=destination.name,
                    status="reachable",
                    route=route,
                    gpkg_path=gpkg_path,
                )
            )

    return results


def validate_all_sites(
    candidates_path: Path,
    dem_path: Path,
    cost_path: Path,
) -> tuple[ValidatedDestination, list[ValidatedDestination]]:
    """
    Load and validate the Basecamp and all destination candidates.

    One BlockCache is reused for all coordinate/passability checks.
    """
    georef = load_georeference(dem_path)
    destinations = load_destination_candidates_from_science_sites(
        candidates_path
    )

    with BlockCache(cost_path) as cache:
        basecamp_result = validate_destination(BASECAMP, georef, cache)

        destination_results = [
            validate_destination(candidate, georef, cache)
            for candidate in destinations
        ]

    return basecamp_result, destination_results


def load_destination_candidates_from_science_sites(
    path: Path,
) -> list[DestinationCandidate]:
    """Load destination candidates from data/science-sites.json."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    return [
        DestinationCandidate(
            id=entry["id"],
            name=entry["name"],
            longitude_deg=entry["coordinates"]["longitude_deg"],
            latitude_deg=entry["coordinates"]["latitude_deg"],
            geological_unit=entry["geological_unit"],
        )
        for entry in data["destinations"]
    ]
