"""Tests for browser asset preparation."""

import json

import fiona
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, mapping

from ares_pipeline.web_export import prepare_web_assets


def create_raster(
    path,
    data,
    *,
    dtype,
    nodata,
    transform,
):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:3857",
        transform=transform,
        nodata=nodata,
    ) as destination:
        destination.write(data.astype(dtype), 1)


def create_route(path, transform):
    # Pixel-centre coordinates for row/column pairs (10, 10) and (12, 12).
    start = transform @ (10.5, 10.5)
    end = transform @ (12.5, 12.5)

    geometry = LineString([start, end])

    schema = {
        "geometry": "LineString",
        "properties": {
            "distance_m": "float",
            "modeled_cost": "float",
            "cells": "int",
            "expanded": "int",
        },
    }

    with fiona.open(
        path,
        "w",
        driver="GPKG",
        layer="route",
        schema=schema,
        crs="EPSG:3857",
    ) as destination:
        destination.write(
            {
                "geometry": mapping(geometry),
                "properties": {
                    "distance_m": 2.828,
                    "modeled_cost": 3.5,
                    "cells": 3,
                    "expanded": 20,
                },
            }
        )


def create_test_inputs(tmp_path):
    transform = from_origin(100, 200, 1, 1)

    surface = np.full((20, 20), 120, dtype=np.uint8)
    cost = np.full((20, 20), 0.5, dtype=np.float32)
    dem = np.full((20, 20), 100.0, dtype=np.float32)

    # Simulate an irregular NoData footprint.
    surface[0:4, 0:4] = 0
    cost[0:4, 0:4] = np.nan
    dem[0:4, 0:4] = -32767.0

    surface_path = tmp_path / "surface.tif"
    cost_path = tmp_path / "cost.tif"
    dem_path = tmp_path / "dem.tif"
    route_path = tmp_path / "route.gpkg"

    create_raster(
        surface_path,
        surface,
        dtype="uint8",
        nodata=0,
        transform=transform,
    )

    create_raster(
        cost_path,
        cost,
        dtype="float32",
        nodata=np.nan,
        transform=transform,
    )

    create_raster(
        dem_path,
        dem,
        dtype="float32",
        nodata=-32767.0,
        transform=transform,
    )

    create_route(route_path, transform)

    return surface_path, cost_path, dem_path, route_path


def test_prepare_web_assets_creates_all_outputs(tmp_path):
    inputs = create_test_inputs(tmp_path)
    output_dir = tmp_path / "web-data"

    outputs = prepare_web_assets(
        *inputs,
        output_dir,
        detail_size_m=10,
    )

    assert set(outputs) == {
        "overview",
        "surface",
        "cost",
        "route",
        "mission",
    }

    for path in outputs.values():
        assert path.is_file()


def test_surface_and_cost_images_have_matching_dimensions(tmp_path):
    inputs = create_test_inputs(tmp_path)
    output_dir = tmp_path / "web-data"

    outputs = prepare_web_assets(
        *inputs,
        output_dir,
        detail_size_m=10,
    )

    with (
        rasterio.open(outputs["surface"]) as surface,
        rasterio.open(outputs["cost"]) as cost,
    ):
        assert (surface.width, surface.height) == (
            cost.width,
            cost.height,
        )

        assert surface.count == 4
        assert cost.count == 4


def test_nodata_becomes_transparent_in_overview(tmp_path):
    inputs = create_test_inputs(tmp_path)
    output_dir = tmp_path / "web-data"

    outputs = prepare_web_assets(
        *inputs,
        output_dir,
    )

    with rasterio.open(outputs["overview"]) as overview:
        alpha = overview.read(4)

        assert alpha[0, 0] == 0
        assert alpha[-1, -1] == 255


def test_route_coordinates_are_relative_to_detail_crop(tmp_path):
    inputs = create_test_inputs(tmp_path)
    output_dir = tmp_path / "web-data"

    outputs = prepare_web_assets(
        *inputs,
        output_dir,
        detail_size_m=10,
    )

    route = json.loads(outputs["route"].read_text())
    mission = json.loads(outputs["mission"].read_text())

    assert route["origin"] == "top-left"
    assert route["coordinate_system"] == "image-pixels"

    assert len(route["coordinates"]) == 2

    for point in route["coordinates"]:
        assert 0 <= point["x"] < mission["detail"]["width"]
        assert 0 <= point["y"] < mission["detail"]["height"]

    assert mission["route"]["distance_m"] == 2.828
    assert mission["route"]["modeled_cost_m_equivalent"] == 3.5


def test_elevation_profile_uses_dem_values(tmp_path):
    inputs = create_test_inputs(tmp_path)
    output_dir = tmp_path / "web-data"

    outputs = prepare_web_assets(
        *inputs,
        output_dir,
        detail_size_m=10,
    )

    route = json.loads(outputs["route"].read_text())

    profile = route["elevation_profile"]

    assert len(profile) == 2
    assert profile[0]["distance_m"] == 0.0
    assert profile[0]["elevation_m"] == 100.0
    assert profile[1]["elevation_m"] == 100.0
    assert profile[1]["distance_m"] > 0


def test_misaligned_raster_is_rejected(tmp_path):
    surface_path, cost_path, dem_path, route_path = create_test_inputs(
        tmp_path
    )

    misaligned_path = tmp_path / "misaligned-cost.tif"

    with rasterio.open(cost_path) as source:
        data = source.read(1)

    create_raster(
        misaligned_path,
        data,
        dtype="float32",
        nodata=np.nan,
        transform=from_origin(101, 200, 1, 1),
    )

    import pytest

    with pytest.raises(ValueError, match="transform"):
        prepare_web_assets(
            surface_path,
            misaligned_path,
            dem_path,
            route_path,
            tmp_path / "web-data",
            detail_size_m=10,
        )
