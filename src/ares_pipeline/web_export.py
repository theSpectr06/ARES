"""Prepare browser-friendly map assets from ARES geospatial outputs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import fiona
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import shape

DEFAULT_DETAIL_SIZE_M = 2000
DEFAULT_OVERVIEW_MAX_DIMENSION = 2048
COST_OVERLAY_ALPHA = 165


def _validate_matching_grids(
    surface: rasterio.io.DatasetReader,
    cost: rasterio.io.DatasetReader,
    dem: rasterio.io.DatasetReader,
) -> None:
    """Ensure all three rasters use the same grid."""

    datasets = {
        "surface": surface,
        "cost": cost,
        "DEM": dem,
    }
    reference = datasets["DEM"]

    if reference.crs is None:
        raise ValueError("DEM has no CRS.")

    for name, dataset in datasets.items():
        if dataset.crs != reference.crs:
            raise ValueError(f"{name} raster CRS does not match the DEM.")

        if (dataset.width, dataset.height) != (
            reference.width,
            reference.height,
        ):
            raise ValueError(f"{name} raster dimensions do not match the DEM.")

        if not dataset.transform.almost_equals(reference.transform):
            raise ValueError(f"{name} raster transform does not match the DEM.")

        if dataset.count != 1:
            raise ValueError(f"{name} raster must contain exactly one band.")

    transform = reference.transform

    if not math.isclose(transform.b, 0.0, abs_tol=1e-12):
        raise ValueError("Rotated rasters are not supported by this export.")

    if not math.isclose(transform.d, 0.0, abs_tol=1e-12):
        raise ValueError("Rotated rasters are not supported by this export.")

    if transform.a <= 0 or transform.e >= 0:
        raise ValueError(
            "Expected a north-up raster with positive X and negative Y resolution."
        )


def _read_route(route_path: Path) -> tuple[Any, dict[str, Any], CRS]:
    """Read the exported route geometry and attributes."""

    with fiona.open(route_path, layer="route") as source:
        if len(source) != 1:
            raise ValueError(
                "Expected exactly one route feature in the GeoPackage."
            )

        feature = next(iter(source))

        if feature["geometry"] is None:
            raise ValueError("Route feature has no geometry.")

        geometry = shape(feature["geometry"])

        if geometry.geom_type != "LineString":
            raise ValueError("Route geometry must be a LineString.")

        if len(geometry.coords) < 2:
            raise ValueError("Route must contain at least two coordinates.")

        if source.crs_wkt:
            route_crs = CRS.from_wkt(source.crs_wkt)
        elif source.crs:
            route_crs = CRS.from_user_input(source.crs)
        else:
            raise ValueError("Route GeoPackage has no CRS.")

        properties = dict(feature["properties"])

    required = {"distance_m", "modeled_cost", "cells", "expanded"}
    missing = required.difference(properties)

    if missing:
        raise ValueError(
            f"Route is missing required attributes: {sorted(missing)}"
        )

    return geometry, properties, route_crs


def _route_to_pixel_coordinates(
    geometry: Any,
    transform: rasterio.Affine,
) -> list[tuple[float, float]]:
    """Convert route map coordinates to (column, row) pixel coordinates."""

    inverse = ~transform
    coordinates = []

    for x, y in geometry.coords:
        col = inverse.a * x + inverse.b * y + inverse.c
        row = inverse.d * x + inverse.e * y + inverse.f
        coordinates.append((float(col), float(row)))

    return coordinates


def _make_detail_window(
    route_pixels: list[tuple[float, float]],
    width: int,
    height: int,
    transform: rasterio.Affine,
    detail_size_m: float,
) -> Window:
    """Create a square-ish crop centred on the route."""

    if detail_size_m <= 0:
        raise ValueError("detail_size_m must be positive.")

    pixel_width_m = abs(transform.a)
    pixel_height_m = abs(transform.e)

    crop_width = min(
        width,
        max(1, math.ceil(detail_size_m / pixel_width_m)),
    )
    crop_height = min(
        height,
        max(1, math.ceil(detail_size_m / pixel_height_m)),
    )

    min_col = min(point[0] for point in route_pixels)
    max_col = max(point[0] for point in route_pixels)
    min_row = min(point[1] for point in route_pixels)
    max_row = max(point[1] for point in route_pixels)

    center_col = (min_col + max_col) / 2
    center_row = (min_row + max_row) / 2

    col_off = math.floor(center_col - crop_width / 2)
    row_off = math.floor(center_row - crop_height / 2)

    col_off = max(0, min(col_off, width - crop_width))
    row_off = max(0, min(row_off, height - crop_height))

    window = Window(
        col_off=col_off,
        row_off=row_off,
        width=crop_width,
        height=crop_height,
    )

    for col, row in route_pixels:
        if not (
            col_off <= col < col_off + crop_width
            and row_off <= row < row_off + crop_height
        ):
            raise ValueError(
                "The selected detail crop does not contain the entire route."
            )

    return window


def _surface_rgba(values: np.ma.MaskedArray) -> np.ndarray:
    """Convert a masked grayscale raster into transparent RGBA imagery."""

    data = np.asarray(values.data, dtype=np.float32)
    invalid = np.ma.getmaskarray(values) | ~np.isfinite(data)
    valid_values = data[~invalid]

    if valid_values.size == 0:
        raise ValueError("Surface image contains no valid pixels.")

    low, high = np.percentile(valid_values, [2, 98])

    if high <= low:
        high = low + 1.0

    scaled = np.clip(
        np.nan_to_num(
            (data - low) / (high - low),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )

    grayscale = np.round(scaled * 255).astype(np.uint8)
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

    rgba[..., 0] = grayscale
    rgba[..., 1] = grayscale
    rgba[..., 2] = grayscale
    rgba[..., 3] = np.where(invalid, 0, 255).astype(np.uint8)

    return rgba


def _cost_rgba(values: np.ma.MaskedArray) -> np.ndarray:
    """Colourize normalized terrain cost into a transparent RGBA overlay."""

    data = np.asarray(values.data, dtype=np.float32)
    invalid = np.ma.getmaskarray(values) | ~np.isfinite(data)
    valid_values = data[~invalid]

    if valid_values.size and (
        np.any(valid_values < 0.0) or np.any(valid_values > 1.0)
    ):
        raise ValueError("Terrain cost values must be within [0, 1].")

    normalized = np.clip(
        np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )

    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

    # Low cost: green. High cost: red.
    rgba[..., 0] = np.round(normalized * 255).astype(np.uint8)
    rgba[..., 1] = np.round((1.0 - normalized) * 220).astype(np.uint8)
    rgba[..., 2] = np.round((1.0 - normalized) * 80).astype(np.uint8)
    rgba[..., 3] = np.where(
        invalid,
        0,
        COST_OVERLAY_ALPHA,
    ).astype(np.uint8)

    return rgba


def _write_rgba_png(path: Path, rgba: np.ndarray) -> None:
    """Write an RGBA array as a PNG without geospatial metadata."""

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected an RGBA array with four channels.")

    height, width, _ = rgba.shape

    profile = {
        "driver": "PNG",
        "width": width,
        "height": height,
        "count": 4,
        "dtype": "uint8",
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path, "w", **profile) as destination:
        destination.write(np.moveaxis(rgba, -1, 0))


def _sample_elevation_profile(
    dem: rasterio.io.DatasetReader,
    coordinates: list[tuple[float, float]],
) -> list[dict[str, float | None]]:
    """Sample DEM elevation at each route vertex."""

    samples = list(dem.sample(coordinates, masked=True))
    profile = []
    cumulative_distance = 0.0
    previous = None

    for coordinate, sample in zip(coordinates, samples, strict=True):
        if previous is not None:
            cumulative_distance += math.dist(previous, coordinate)

        previous = coordinate
        value = sample[0]

        elevation = (
            None
            if np.ma.is_masked(value) or not np.isfinite(value)
            else float(value)
        )

        profile.append(
            {
                "distance_m": cumulative_distance,
                "elevation_m": elevation,
            }
        )

    return profile


def _write_json(path: Path, content: dict[str, Any]) -> None:
    """Write JSON with stable, readable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_overview(
    surface: rasterio.io.DatasetReader,
    dem: rasterio.io.DatasetReader,
    output_dir: Path,
    overview_max_dimension: int,
) -> dict[str, Any]:
    """Generate the shared regional overview."""

    if overview_max_dimension < 1:
        raise ValueError("overview_max_dimension must be positive.")

    scale = min(
        1.0,
        overview_max_dimension / max(dem.width, dem.height),
    )

    overview_width = max(1, round(dem.width * scale))
    overview_height = max(1, round(dem.height * scale))

    overview = surface.read(
        1,
        out_shape=(overview_height, overview_width),
        masked=True,
        resampling=Resampling.average,
    )

    overview_rgba = _surface_rgba(overview)
    overview_path = output_dir / "surface-overview.png"
    _write_rgba_png(overview_path, overview_rgba)

    overview_transform = (
        dem.transform
        * rasterio.Affine.scale(
            dem.width / overview_width,
            dem.height / overview_height,
        )
    )

    return {
        "path": overview_path,
        "file": "surface-overview.png",
        "width": overview_width,
        "height": overview_height,
        "transform": list(overview_transform)[:6],
    }


def _prepare_route_bundle(
    surface: rasterio.io.DatasetReader,
    cost: rasterio.io.DatasetReader,
    dem: rasterio.io.DatasetReader,
    route_path: Path,
    destination_dir: Path,
    *,
    overview: dict[str, Any],
    asset_prefix: str,
    detail_size_m: float,
    destination: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate one route's detail images and JSON metadata."""

    geometry, route_properties, route_crs = _read_route(route_path)

    if route_crs != dem.crs:
        raise ValueError(f"Route CRS does not match the raster CRS: {route_path}")

    transform = dem.transform
    route_pixels = _route_to_pixel_coordinates(geometry, transform)

    for col, row in route_pixels:
        if not (0 <= col < dem.width and 0 <= row < dem.height):
            raise ValueError(
                f"Route contains coordinates outside the DEM extent: {route_path}"
            )

    window = _make_detail_window(
        route_pixels,
        dem.width,
        dem.height,
        transform,
        detail_size_m,
    )

    detail_surface = surface.read(1, window=window, masked=True)
    detail_cost = cost.read(1, window=window, masked=True)

    surface_rgba = _surface_rgba(detail_surface)
    cost_rgba = _cost_rgba(detail_cost)

    destination_dir.mkdir(parents=True, exist_ok=True)

    detail_surface_path = destination_dir / "surface-detail.png"
    cost_overlay_path = destination_dir / "cost-overlay.png"

    _write_rgba_png(detail_surface_path, surface_rgba)
    _write_rgba_png(cost_overlay_path, cost_rgba)

    col_off = int(window.col_off)
    row_off = int(window.row_off)

    route_local_pixels = [
        {
            "x": col - col_off,
            "y": row - row_off,
        }
        for col, row in route_pixels
    ]

    route_map_coordinates = list(geometry.coords)
    elevation_profile = _sample_elevation_profile(
        dem,
        route_map_coordinates,
    )

    detail_transform = rasterio.windows.transform(window, transform)

    route_json = {
        "coordinate_system": "image-pixels",
        "origin": "top-left",
        "coordinates": route_local_pixels,
        "elevation_profile": elevation_profile,
    }

    route_output_path = destination_dir / "route.json"
    mission_output_path = destination_dir / "mission.json"

    _write_json(route_output_path, route_json)

    mission_json: dict[str, Any] = {
        "mission": "Jezero Crater EVA",
        "source": {
            "surface": surface.name,
            "cost": cost.name,
            "dem": dem.name,
            "route": route_path.name,
        },
        "crs": dem.crs.to_wkt(),
        "raster": {
            "width": dem.width,
            "height": dem.height,
            "transform": list(transform)[:6],
            "pixel_size_m": [
                abs(transform.a),
                abs(transform.e),
            ],
        },
        "overview": {
            "file": overview["file"],
            "width": overview["width"],
            "height": overview["height"],
            "transform": overview["transform"],
            "coordinate_system": "image-pixels",
            "origin": "top-left",
        },
        "detail": {
            "surface_file": f"{asset_prefix}surface-detail.png",
            "cost_file": f"{asset_prefix}cost-overlay.png",
            "width": int(window.width),
            "height": int(window.height),
            "row_offset": row_off,
            "column_offset": col_off,
            "transform": list(detail_transform)[:6],
            "coordinate_system": "image-pixels",
            "origin": "top-left",
        },
        "cost_model": {
            "normalization": "0 to 1",
            "low_cost_color": "#00dc50",
            "high_cost_color": "#ff0000",
            "meaning": (
                "Relative modelled traversal cost from slope and roughness"
            ),
            "warning": (
                "Not a complete hazard-detection or astronaut-safety model"
            ),
        },
        "route": {
            "distance_m": float(route_properties["distance_m"]),
            "modeled_cost_m_equivalent": float(
                route_properties["modeled_cost"]
            ),
            "cells": int(route_properties["cells"]),
            "expanded": int(route_properties["expanded"]),
            "elevation_profile": elevation_profile,
        },
    }

    if destination is not None:
        mission_json["destination"] = {
            "id": destination["id"],
            "name": destination["name"],
        }

    _write_json(mission_output_path, mission_json)

    return {
        "surface": detail_surface_path,
        "cost": cost_overlay_path,
        "route": route_output_path,
        "mission": mission_output_path,
    }


def prepare_web_assets(
    surface_path: Path,
    cost_path: Path,
    dem_path: Path,
    route_path: Path,
    output_dir: Path,
    *,
    detail_size_m: float = DEFAULT_DETAIL_SIZE_M,
    overview_max_dimension: int = DEFAULT_OVERVIEW_MAX_DIMENSION,
) -> dict[str, Path]:
    """Generate regional and detailed browser assets for one route.

    Retained for compatibility with the original single-route workflow.
    """

    paths = [
        Path(surface_path),
        Path(cost_path),
        Path(dem_path),
        Path(route_path),
    ]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        rasterio.open(surface_path) as surface,
        rasterio.open(cost_path) as cost,
        rasterio.open(dem_path) as dem,
    ):
        _validate_matching_grids(surface, cost, dem)

        overview = _prepare_overview(
            surface,
            dem,
            output_dir,
            overview_max_dimension,
        )

        bundle = _prepare_route_bundle(
            surface,
            cost,
            dem,
            Path(route_path),
            output_dir,
            overview=overview,
            asset_prefix="",
            detail_size_m=detail_size_m,
        )

    return {
        "overview": overview["path"],
        **bundle,
    }


def prepare_multi_destination_web_assets(
    surface_path: Path,
    cost_path: Path,
    dem_path: Path,
    destinations: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    detail_size_m: float = DEFAULT_DETAIL_SIZE_M,
    overview_max_dimension: int = DEFAULT_OVERVIEW_MAX_DIMENSION,
) -> dict[str, Any]:
    """Generate one shared overview and one detail bundle per destination.

    Each destination mapping must contain:
        id: stable destination ID
        name: display name
        route_path: path to its exported route GeoPackage

    Output:
        output_dir/
            surface-overview.png
            destinations.json
            destinations/<destination-id>/
                mission.json
                route.json
                surface-detail.png
                cost-overlay.png
    """

    raster_paths = [
        Path(surface_path),
        Path(cost_path),
        Path(dem_path),
    ]

    for path in raster_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    if not destinations:
        raise ValueError("At least one destination must be provided.")

    if detail_size_m <= 0:
        raise ValueError("detail_size_m must be positive.")

    if overview_max_dimension < 1:
        raise ValueError("overview_max_dimension must be positive.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validated_destinations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for item in destinations:
        destination_id = str(item.get("id", "")).strip()
        destination_name = str(item.get("name", "")).strip()
        route_path = Path(item.get("route_path", ""))

        if not destination_id or not destination_name:
            raise ValueError("Each destination requires a non-empty id and name.")

        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", destination_id):
            raise ValueError(
                f"Invalid destination ID {destination_id!r}; "
                "use lowercase letters, numbers, and hyphens."
            )

        if destination_id in seen_ids:
            raise ValueError(f"Duplicate destination ID: {destination_id}")

        if not route_path.is_file():
            raise FileNotFoundError(
                f"Route file for {destination_id} does not exist: {route_path}"
            )

        seen_ids.add(destination_id)

        validated_destinations.append(
            {
                "id": destination_id,
                "name": destination_name,
                "route_path": route_path,
            }
        )

    index_entries: list[dict[str, Any]] = []
    bundles: dict[str, dict[str, Path]] = {}

    with (
        rasterio.open(surface_path) as surface,
        rasterio.open(cost_path) as cost,
        rasterio.open(dem_path) as dem,
    ):
        _validate_matching_grids(surface, cost, dem)

        # Generate the expensive regional overview exactly once.
        overview = _prepare_overview(
            surface,
            dem,
            output_dir,
            overview_max_dimension,
        )

        for destination in validated_destinations:
            destination_id = destination["id"]
            destination_dir = (
                output_dir / "destinations" / destination_id
            )
            asset_prefix = f"destinations/{destination_id}/"

            bundle = _prepare_route_bundle(
                surface,
                cost,
                dem,
                destination["route_path"],
                destination_dir,
                overview=overview,
                asset_prefix=asset_prefix,
                detail_size_m=detail_size_m,
                destination=destination,
            )

            bundles[destination_id] = bundle

            # Read the route metrics from the generated mission metadata.
            mission_data = json.loads(
                bundle["mission"].read_text(encoding="utf-8")
            )
            route_metrics = mission_data["route"]

            index_entries.append(
                {
                    "id": destination_id,
                    "name": destination["name"],
                    "mission_file": (
                        f"{asset_prefix}mission.json"
                    ),
                    "route_file": (
                        f"{asset_prefix}route.json"
                    ),
                    "distance_m": route_metrics["distance_m"],
                    "modeled_cost_m_equivalent": (
                        route_metrics["modeled_cost_m_equivalent"]
                    ),
                }
            )

    index_path = output_dir / "destinations.json"

    _write_json(
        index_path,
        {
            "schema_version": "1.0",
            "overview_file": "surface-overview.png",
            "destinations": index_entries,
        },
    )

    return {
        "overview": overview["path"],
        "index": index_path,
        "destinations": bundles,
    }
