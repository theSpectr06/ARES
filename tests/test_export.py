import fiona
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import shape

from ares_pipeline.export import export_route
from ares_pipeline.pathfind import RouteResult


@pytest.fixture
def cost_raster(tmp_path):
    path = tmp_path / "cost.tif"

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=3,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(100, 200, 1, 1),
    ) as raster:
        raster.write(
            np.array(
                [
                    [0.2, 0.3, 0.4],
                    [0.3, 0.4, 0.5],
                    [0.4, 0.5, 0.6],
                ],
                dtype=np.float32,
            ),
            1,
        )

    return path


def test_export_route_geometry_and_attributes(cost_raster, tmp_path):
    route = RouteResult(
        path=[(0, 0), (1, 1), (2, 2)],
        geometric_distance_m=2.828427,
        accumulated_cost=3.2,
        cells_expanded=12,
    )

    output = tmp_path / "route.gpkg"

    export_route(route, cost_raster, output)

    assert output.exists()

    with fiona.open(output, layer="route") as source:
        assert len(source) == 1

        feature = next(iter(source))
        geometry = shape(feature["geometry"])

        assert geometry.geom_type == "LineString"
        assert list(geometry.coords) == [
            (100.5, 199.5),
            (101.5, 198.5),
            (102.5, 197.5),
        ]

        assert feature["properties"]["distance_m"] == pytest.approx(
            2.828427
        )
        assert feature["properties"]["modeled_cost"] == pytest.approx(3.2)
        assert feature["properties"]["cells"] == 3
        assert feature["properties"]["expanded"] == 12


def test_export_preserves_crs(cost_raster, tmp_path):
    route = RouteResult(
        path=[(0, 0), (1, 1)],
        geometric_distance_m=1.414214,
        accumulated_cost=1.8,
        cells_expanded=5,
    )

    output = tmp_path / "route.gpkg"
    export_route(route, cost_raster, output)

    with rasterio.open(cost_raster) as raster:
        expected_crs = raster.crs

    with fiona.open(output, layer="route") as source:
        actual_crs = rasterio.crs.CRS.from_wkt(source.crs_wkt)

    assert actual_crs == expected_crs


def test_export_rejects_missing_route(cost_raster, tmp_path):
    route = RouteResult(
        path=None,
        geometric_distance_m=0.0,
        accumulated_cost=0.0,
        cells_expanded=0,
    )

    with pytest.raises(ValueError, match="route that was not found"):
        export_route(route, cost_raster, tmp_path / "route.gpkg")


def test_export_rejects_existing_output(cost_raster, tmp_path):
    output = tmp_path / "route.gpkg"
    output.touch()

    route = RouteResult(
        path=[(0, 0), (1, 1)],
        geometric_distance_m=1.414214,
        accumulated_cost=1.8,
        cells_expanded=5,
    )

    with pytest.raises(FileExistsError):
        export_route(route, cost_raster, output)
