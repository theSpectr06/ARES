"""Export A* routes to GeoPackage."""

from pathlib import Path

import fiona
import rasterio
from rasterio.transform import xy
from shapely.geometry import LineString, mapping

from ares_pipeline.pathfind import RouteResult


def export_route(
    route: RouteResult,
    cost_raster_path: Path,
    output_path: Path,
) -> Path:
    """Export a RouteResult as a LineString in the cost raster's CRS."""

    if route.path is None:
        raise ValueError("Cannot export a route that was not found.")

    if len(route.path) < 2:
        raise ValueError("A route must contain at least two cells.")

    cost_raster_path = Path(cost_raster_path)
    output_path = Path(output_path)

    if output_path.resolve() == cost_raster_path.resolve():
        raise ValueError("Output path must not overwrite the cost raster.")

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Remove it first if you want to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(cost_raster_path) as raster:
        if raster.crs is None:
            raise ValueError("Cost raster has no coordinate reference system.")

        # Convert (row, column) cells to map coordinates at pixel centres.
        coordinates = [
            xy(raster.transform, row, col, offset="center")
            for row, col in route.path
        ]

        line = LineString(coordinates)

        schema = {
            "geometry": "LineString",
            "properties": {
                "distance_m": "float",
                "modeled_cost": "float",
                "cells": "int",
                "expanded": "int",
            },
        }

        properties = {
            "distance_m": route.geometric_distance_m,
            "modeled_cost": route.accumulated_cost,
            "cells": len(route.path),
            "expanded": route.cells_expanded,
        }

        with fiona.open(
            output_path,
            mode="w",
            driver="GPKG",
            layer="route",
            schema=schema,
            crs_wkt=raster.crs.to_wkt(),
        ) as destination:
            destination.write(
                {
                    "geometry": mapping(line),
                    "properties": properties,
                }
            )

    return output_path
