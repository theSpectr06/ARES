"""
Tests for destination terrain validation, including the exact rejection
pattern required for a destination like Nili Planum (invalid, with no
nearby valid cell), and the "nearby valid cell exists but is not silently
used" case.
"""
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from ares_pipeline.coordinates import load_georeference
from ares_pipeline.pathfind import BlockCache
from ares_pipeline.science_sites import DestinationCandidate, validate_destination

MARS_CRS_WKT = (
    'PROJCS["Equirectangular_MARS",GEOGCS["GCS_D_MARS",'
    'DATUM["D_MARS",SPHEROID["MARS_localRadius",3396190,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
    'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]'
)


def make_rasters(tmp_path, cost_array, origin=(0.0, 100.0), pixel_size=1.0):
    rows, cols = cost_array.shape
    transform = from_origin(origin[0], origin[1], pixel_size, pixel_size)
    crs = CRS.from_wkt(MARS_CRS_WKT)

    dem_path = tmp_path / "dem.tif"

    # Only georeferencing is needed from the DEM in these tests.
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-32767.0,
    ):
        pass

    cost_path = tmp_path / "cost.tif"

    # The cost raster must contain actual values because BlockCache reads it.
    with rasterio.open(
        cost_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        dst.write(cost_array.astype(np.float32), 1)

    return dem_path, cost_path


def lonlat_for_pixel(col, row, origin=(0.0, 100.0), pixel_size=1.0):
    """Compute lon/lat that converts back to a given pixel."""
    from pyproj import Transformer

    crs = CRS.from_wkt(MARS_CRS_WKT)

    x = origin[0] + col * pixel_size
    y = origin[1] - row * pixel_size

    geographic = CRS.from_user_input(
        "+proj=longlat +a=3396190 +b=3396190 +no_defs"
    )

    transformer = Transformer.from_crs(
        crs,
        geographic,
        always_xy=True,
    )

    lon, lat = transformer.transform(x, y)

    return lon, lat


def test_valid_destination_is_accepted_without_relocation(tmp_path):
    cost = np.zeros((32, 32))

    dem_path, cost_path = make_rasters(tmp_path, cost)

    georef = load_georeference(dem_path)

    lon, lat = lonlat_for_pixel(16, 16)

    candidate = DestinationCandidate(
        "site-a",
        "Site A",
        lon,
        lat,
        "Test unit",
    )

    with BlockCache(cost_path, block_size=16) as cache:
        result = validate_destination(candidate, georef, cache)

    assert result.valid is True
    assert result.rejection_reason is None
    assert (result.pixel_row, result.pixel_col) == (16, 16)


def test_destination_on_nodata_with_no_nearby_valid_cell_is_rejected_not_relocated(
    tmp_path,
):
    """
    Mirrors the real Nili Planum rejection: entirely NoData region, no
    valid cell anywhere nearby -- must be rejected outright, never
    silently moved.
    """
    cost = np.full((32, 32), np.nan)

    dem_path, cost_path = make_rasters(tmp_path, cost)

    georef = load_georeference(dem_path)

    lon, lat = lonlat_for_pixel(16, 16)

    candidate = DestinationCandidate(
        "nili-planum",
        "Nili Planum fractured unit",
        lon,
        lat,
        "Test unit",
    )

    with BlockCache(cost_path, block_size=16) as cache:
        result = validate_destination(
            candidate,
            georef,
            cache,
            search_radius_m=5.0,
        )

    assert result.valid is False
    assert "no valid cell was found" in result.rejection_reason


def test_destination_with_nearby_valid_cell_is_flagged_not_silently_relocated(
    tmp_path,
):
    cost = np.full((32, 32), np.nan)
    cost[16, 20] = 0.1

    dem_path, cost_path = make_rasters(tmp_path, cost)

    georef = load_georeference(dem_path)

    lon, lat = lonlat_for_pixel(16, 16)

    candidate = DestinationCandidate(
        "site-b",
        "Site B",
        lon,
        lat,
        "Test unit",
    )

    with BlockCache(cost_path, block_size=16) as cache:
        result = validate_destination(
            candidate,
            georef,
            cache,
            search_radius_m=10.0,
        )

    # Must be marked INVALID; a human must decide whether to accept
    # the nearby substitute.
    assert result.valid is False
    assert "marked INVALID, not silently relocated" in result.rejection_reason
    assert (result.pixel_row, result.pixel_col) == (16, 20)


def test_out_of_bounds_coordinate_is_rejected(tmp_path):
    cost = np.zeros((32, 32))

    dem_path, cost_path = make_rasters(tmp_path, cost)

    georef = load_georeference(dem_path)

    candidate = DestinationCandidate(
        "far-away",
        "Far Away",
        0.0,
        0.0,
        "Test unit",
    )

    with BlockCache(cost_path, block_size=16) as cache:
        result = validate_destination(candidate, georef, cache)

    assert result.valid is False
    assert "outside the DEM's extent" in result.rejection_reason
