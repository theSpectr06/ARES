"""
Tests for Basecamp-to-destination routing orchestration.

Uses small synthetic rasters and the existing Stage 4 pathfinding
and GeoPackage export implementations.
"""
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from ares_pipeline.mission_routes import (
    compute_basecamp_to_destination_routes,
)
from ares_pipeline.science_sites import ValidatedDestination


MARS_CRS_WKT = (
    'PROJCS["Equirectangular_MARS",GEOGCS["GCS_D_MARS",'
    'DATUM["D_MARS",SPHEROID["MARS_localRadius",3396190,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
    'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]'
)


def make_cost_raster(path, cost_array, block_size=256):
    rows, cols = cost_array.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=CRS.from_wkt(MARS_CRS_WKT),
        transform=from_origin(0, rows, 1.0, 1.0),
        nodata=np.nan,
        tiled=True,
        blockxsize=block_size,
        blockysize=block_size,
    ) as dst:
        dst.write(cost_array.astype(np.float32), 1)


def make_validated(id_, name, row, col, valid=True):
    return ValidatedDestination(
        id=id_,
        name=name,
        longitude_deg=0.0,
        latitude_deg=0.0,
        geological_unit="Test",
        pixel_col=col,
        pixel_row=row,
        valid=valid,
        rejection_reason=None if valid else "test rejection",
    )


def test_reachable_destination_produces_exported_route(tmp_path):
    cost_path = tmp_path / "cost.tif"
    make_cost_raster(cost_path, np.zeros((32, 32)))

    basecamp = make_validated("basecamp", "Basecamp", 0, 0)
    destination = make_validated("dest-a", "Destination A", 20, 20)

    results = compute_basecamp_to_destination_routes(
        [destination], basecamp, cost_path, tmp_path / "routes"
    )

    assert len(results) == 1

    result = results[0]
    assert result.status == "reachable"
    assert result.route is not None
    assert result.route.path is not None
    assert result.gpkg_path is not None
    assert result.gpkg_path.exists()

    assert (
        result.route.accumulated_cost
        >= result.route.geometric_distance_m
    )


def test_unreachable_destination_is_reported(tmp_path):
    cost = np.zeros((32, 32))
    cost[:, 16] = np.nan

    cost_path = tmp_path / "cost.tif"
    make_cost_raster(cost_path, cost)

    basecamp = make_validated("basecamp", "Basecamp", 0, 0)
    destination = make_validated("dest-a", "Destination A", 0, 31)

    results = compute_basecamp_to_destination_routes(
        [destination], basecamp, cost_path, tmp_path / "routes"
    )

    assert len(results) == 1
    assert results[0].status == "unreachable"
    assert results[0].route is not None
    assert results[0].route.path is None
    assert results[0].gpkg_path is None
    assert results[0].reason is not None


def test_terrain_invalid_destination_is_not_routed(tmp_path):
    cost_path = tmp_path / "cost.tif"
    make_cost_raster(cost_path, np.zeros((32, 32)))

    basecamp = make_validated("basecamp", "Basecamp", 0, 0)
    destination = make_validated(
        "dest-bad", "Bad Destination", 20, 20, valid=False
    )

    results = compute_basecamp_to_destination_routes(
        [destination], basecamp, cost_path, tmp_path / "routes"
    )

    assert len(results) == 1
    assert results[0].status == "terrain_invalid"
    assert results[0].route is None
    assert results[0].gpkg_path is None
    assert results[0].reason == "test rejection"


def test_invalid_basecamp_raises_value_error(tmp_path):
    cost_path = tmp_path / "cost.tif"
    make_cost_raster(cost_path, np.zeros((32, 32)))

    basecamp = make_validated(
        "basecamp", "Basecamp", 0, 0, valid=False
    )
    destination = make_validated("dest-a", "Destination A", 20, 20)

    with pytest.raises(ValueError, match="Basecamp itself is not valid"):
        compute_basecamp_to_destination_routes(
            [destination], basecamp, cost_path, tmp_path / "routes"
        )


def test_multiple_destinations_get_separate_geopackages(tmp_path):
    cost_path = tmp_path / "cost.tif"
    make_cost_raster(cost_path, np.zeros((32, 32)))

    basecamp = make_validated("basecamp", "Basecamp", 0, 0)

    destinations = [
        make_validated("dest-a", "Destination A", 10, 10),
        make_validated("dest-b", "Destination B", 20, 5),
    ]

    results = compute_basecamp_to_destination_routes(
        destinations, basecamp, cost_path, tmp_path / "routes"
    )

    assert len(results) == 2
    assert all(result.status == "reachable" for result in results)

    gpkg_paths = [result.gpkg_path for result in results]

    assert all(path is not None and path.exists() for path in gpkg_paths)
    assert len(set(gpkg_paths)) == 2

    assert {path.name for path in gpkg_paths} == {
        "route-dest-a.gpkg",
        "route-dest-b.gpkg",
    }
