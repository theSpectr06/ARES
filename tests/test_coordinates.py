"""
Tests for Mars raster coordinate transformations and pixel indexing.
"""
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from ares_pipeline.coordinates import (
    load_georeference,
    lonlat_to_pixel,
    pixel_in_bounds,
    round_to_cell,
)

MARS_CRS_WKT = (
    'PROJCS["Equirectangular_MARS",GEOGCS["GCS_D_MARS",'
    'DATUM["D_MARS",SPHEROID["MARS_localRadius",3396190,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
    'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]'
)


def make_test_raster(path, origin_x, origin_y, pixel_size, width, height):
    """Create a small metadata-only GeoTIFF for georeferencing tests."""
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    crs = CRS.from_wkt(MARS_CRS_WKT)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-32767.0,
    ):
        pass


def test_candidate_4_converts_to_the_tasks_stated_pixel(tmp_path):
    """
    Candidate 4's published coordinates, converted using the real DEM's
    known geotransform (left=4567580.0, top=1101802.0, 1 m/pixel),
    must reproduce the stated pixel (col=13964, row=7341).

    The synthetic raster only stores the geotransform; its dimensions
    and pixel values are irrelevant to this coordinate conversion.
    """
    path = tmp_path / "dem.tif"

    make_test_raster(
        path,
        origin_x=4567580.0,
        origin_y=1101802.0,
        pixel_size=1.0,
        width=1,
        height=1,
    )

    georef = load_georeference(path)

    # Candidate 4: 77.293427 E, 18.464211 N
    col_f, row_f = lonlat_to_pixel(77.293427, 18.464211, georef)

    col, row = round_to_cell(col_f, row_f)

    assert (col, row) == (13964, 7341)


def test_pixel_to_lonlat_round_trips(tmp_path):
    path = tmp_path / "dem.tif"

    make_test_raster(
        path,
        origin_x=4567580.0,
        origin_y=1101802.0,
        pixel_size=1.0,
        width=1,
        height=1,
    )

    georef = load_georeference(path)

    original_col = 1000
    original_row = 2000

    from ares_pipeline.coordinates import pixel_to_lonlat

    lon, lat = pixel_to_lonlat(original_col, original_row, georef)

    col_f, row_f = lonlat_to_pixel(lon, lat, georef)

    assert abs(col_f - original_col) < 1e-6
    assert abs(row_f - original_row) < 1e-6


def test_pixel_in_bounds_rejects_outside_coordinates(tmp_path):
    path = tmp_path / "dem.tif"

    make_test_raster(
        path,
        origin_x=0.0,
        origin_y=100.0,
        pixel_size=1.0,
        width=50,
        height=50,
    )

    georef = load_georeference(path)

    assert pixel_in_bounds(10.0, 10.0, georef)
    assert not pixel_in_bounds(-1.0, 10.0, georef)
    assert not pixel_in_bounds(10.0, -1.0, georef)
    assert not pixel_in_bounds(50.0, 10.0, georef)
    assert not pixel_in_bounds(10.0, 50.0, georef)


def test_round_to_cell_uses_floor_not_nearest():
    assert round_to_cell(10.2, 20.8) == (10, 20)
    assert round_to_cell(10.9, 20.1) == (10, 20)
    assert round_to_cell(10.0, 20.0) == (10, 20)
