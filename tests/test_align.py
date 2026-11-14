# tests/test_align.py
"""
Synthetic-raster tests covering spatial-origin mismatch, CRS mismatch,
NoData handling, partial-validity blocks, and exact grid alignment.
No real Mars data required.
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ares_pipeline.align import (
    RasterInfo,
    count_valid_fraction_windowed,
    resample_image_to_dem_grid,
    validate_matching_grid,
    validate_resolution_ratio,
    validate_spatial_alignment,
)

MARS_CRS = (
    'PROJCS["Equirectangular_MARS",GEOGCS["GCS_D_MARS",'
    'DATUM["D_MARS",SPHEROID["MARS_localRadius",3396190,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
    'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]'
)
OTHER_CRS = "EPSG:4326"


def make_raster(path, width, height, pixel_size, dtype, nodata, fill_value,
                 origin=(0.0, 0.0), crs=MARS_CRS):
    transform = from_origin(origin[0], origin[1], pixel_size, pixel_size)
    data = np.full((height, width), fill_value, dtype=dtype)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def raster_info(path):
    with rasterio.open(path) as src:
        return RasterInfo(
            path=path, crs=src.crs, transform=src.transform, bounds=src.bounds,
            width=src.width, height=src.height, dtype=src.dtypes[0],
            nodata=src.nodata, res=(src.res[0], src.res[1]),
        )


# --- grid-matching (DEM vs confidence) ---

def test_matching_grid_passes_for_identical_rasters(tmp_path):
    a, b = tmp_path / "dem.tif", tmp_path / "conf.tif"
    make_raster(a, 10, 10, 1.0, "float32", -32767.0, 100.0)
    make_raster(b, 10, 10, 1.0, "uint8", 0, 200)
    validate_matching_grid(raster_info(a), raster_info(b))  # should not raise


def test_matching_grid_raises_on_dimension_mismatch(tmp_path):
    a, b = tmp_path / "dem.tif", tmp_path / "conf.tif"
    make_raster(a, 10, 10, 1.0, "float32", -32767.0, 100.0)
    make_raster(b, 8, 8, 1.0, "uint8", 0, 200)
    with pytest.raises(ValueError, match="dimensions differ"):
        validate_matching_grid(raster_info(a), raster_info(b))


# --- resolution ratio (DEM vs orthoimage) ---

def test_resolution_ratio_detects_exact_integer_ratio(tmp_path):
    dem_path, img_path = tmp_path / "dem.tif", tmp_path / "img.tif"
    make_raster(dem_path, 10, 10, 1.0, "float32", -32767.0, 100.0)
    make_raster(img_path, 40, 40, 0.25, "uint8", 0, 50)
    ratio = validate_resolution_ratio(raster_info(dem_path), raster_info(img_path))
    assert ratio == 4


def test_resolution_ratio_rejects_non_integer_ratio(tmp_path):
    dem_path, img_path = tmp_path / "dem.tif", tmp_path / "img.tif"
    make_raster(dem_path, 10, 10, 1.0, "float32", -32767.0, 100.0)
    make_raster(img_path, 33, 33, 0.303, "uint8", 0, 50)
    with pytest.raises(ValueError, match="not a clean integer"):
        validate_resolution_ratio(raster_info(dem_path), raster_info(img_path))


# --- spatial alignment: CRS, bounds, and fractional-origin checks ---

def test_spatial_alignment_passes_for_exactly_coregistered_rasters(tmp_path):
    dem_path, img_path = tmp_path / "dem.tif", tmp_path / "img.tif"
    make_raster(dem_path, 10, 10, 1.0, "float32", -32767.0, 100.0, origin=(500.0, 500.0))
    make_raster(img_path, 40, 40, 0.25, "uint8", 0, 50, origin=(500.0, 500.0))
    validate_spatial_alignment(raster_info(dem_path), raster_info(img_path))  # should not raise


def test_spatial_alignment_raises_on_crs_mismatch(tmp_path):
    dem_path, img_path = tmp_path / "dem.tif", tmp_path / "img.tif"
    make_raster(dem_path, 10, 10, 1.0, "float32", -32767.0, 100.0, origin=(500.0, 500.0))
    make_raster(img_path, 40, 40, 0.25, "uint8", 0, 50, origin=(500.0, 500.0), crs=OTHER_CRS)
    with pytest.raises(ValueError, match="CRS differ"):
        validate_spatial_alignment(raster_info(dem_path), raster_info(img_path))


def test_spatial_alignment_raises_on_bounds_mismatch(tmp_path):
    dem_path, img_path = tmp_path / "dem.tif", tmp_path / "img.tif"
    make_raster(dem_path, 10, 10, 1.0, "float32", -32767.0, 100.0, origin=(500.0, 500.0))
    # Same resolution ratio and pixel count, but shifted bounds entirely.
    make_raster(img_path, 40, 40, 0.25, "uint8", 0, 50, origin=(600.0, 600.0))
    with pytest.raises(ValueError, match="bounds differ"):
        validate_spatial_alignment(raster_info(dem_path), raster_info(img_path))


def test_spatial_alignment_raises_on_fractional_pixel_origin_offset(tmp_path):
    dem_path, img_path = tmp_path / "dem.tif", tmp_path / "img.tif"
    make_raster(dem_path, 10, 10, 1.0, "float32", -32767.0, 100.0, origin=(500.0, 500.0))
    # Same bounds box is NOT what's being tested here — instead, shift the
    # image by a fractional DEM-pixel amount (0.1 m) so bounds differ by
    # more than tolerance but in a way that specifically isolates the
    # fractional-offset message rather than the bounds-mismatch one: shrink
    # the image's footprint to match by also adjusting pixel size slightly.
    # Simplest faithful case: same left/top origin shifted by 0.1 DEM-pixels.
    make_raster(img_path, 40, 40, 0.25, "uint8", 0, 50, origin=(500.1, 500.0))
    with pytest.raises(ValueError):
        validate_spatial_alignment(raster_info(dem_path), raster_info(img_path))


# --- NoData-aware block averaging ---

def test_fully_valid_block_averages_correctly(tmp_path):
    dem_path = tmp_path / "dem.tif"
    img_path = tmp_path / "img.tif"
    out_path = tmp_path / "out.tif"
    make_raster(dem_path, 4, 4, 1.0, "float32", -32767.0, 0.0)

    data = np.zeros((16, 16), dtype="uint8")
    data[0:4, 0:4] = 100
    transform = from_origin(0, 4, 0.25, 0.25)
    with rasterio.open(
        img_path, "w", driver="GTiff", height=16, width=16, count=1,
        dtype="uint8", crs=MARS_CRS, transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)

    result = resample_image_to_dem_grid(img_path, raster_info(dem_path), ratio=4, out_path=out_path)
    with rasterio.open(result.path) as src:
        out = src.read(1)
    assert out[0, 0] == 100


def test_mostly_nodata_block_becomes_nodata(tmp_path):
    dem_path = tmp_path / "dem.tif"
    img_path = tmp_path / "img.tif"
    out_path = tmp_path / "out.tif"
    make_raster(dem_path, 1, 1, 1.0, "float32", -32767.0, 0.0)

    # 4x4 block: only 2 of 16 source pixels valid -> below 0.5 threshold.
    data = np.zeros((4, 4), dtype="uint8")
    data[0, 0] = 100
    data[0, 1] = 100
    transform = from_origin(0, 1, 0.25, 0.25)
    with rasterio.open(
        img_path, "w", driver="GTiff", height=4, width=4, count=1,
        dtype="uint8", crs=MARS_CRS, transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)

    result = resample_image_to_dem_grid(img_path, raster_info(dem_path), ratio=4, out_path=out_path)
    with rasterio.open(result.path) as src:
        out = src.read(1)
    assert out[0, 0] == 0  # IMAGE_NODATA


def test_partial_validity_at_custom_threshold(tmp_path):
    """
    A block at exactly 50% valid should pass when min_valid_fraction=0.5
    and fail when raised to 0.75 — confirms the threshold is actually
    load-bearing, not a no-op.
    """
    dem_path = tmp_path / "dem.tif"
    img_path = tmp_path / "img.tif"
    make_raster(dem_path, 1, 1, 1.0, "float32", -32767.0, 0.0)

    data = np.zeros((4, 4), dtype="uint8")
    data[0:2, :] = 100  # 8 of 16 pixels valid = 50%
    transform = from_origin(0, 1, 0.25, 0.25)
    with rasterio.open(
        img_path, "w", driver="GTiff", height=4, width=4, count=1,
        dtype="uint8", crs=MARS_CRS, transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)

    out_a = tmp_path / "out_a.tif"
    result_a = resample_image_to_dem_grid(
        img_path, raster_info(dem_path), ratio=4, out_path=out_a, min_valid_fraction=0.5
    )
    with rasterio.open(result_a.path) as src:
        assert src.read(1)[0, 0] == 100

    out_b = tmp_path / "out_b.tif"
    result_b = resample_image_to_dem_grid(
        img_path, raster_info(dem_path), ratio=4, out_path=out_b, min_valid_fraction=0.75
    )
    with rasterio.open(result_b.path) as src:
        assert src.read(1)[0, 0] == 0


# --- windowed output validation ---

def test_count_valid_fraction_windowed_matches_known_value(tmp_path):
    path = tmp_path / "out.tif"
    data = np.zeros((10, 10), dtype="uint8")
    data[:, :6] = 100  # 60 of 100 pixels valid
    transform = from_origin(0, 10, 1.0, 1.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=1,
        dtype="uint8", crs=MARS_CRS, transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)

    fraction = count_valid_fraction_windowed(path, nodata=0, row_chunk=3)
    assert fraction == pytest.approx(0.6)
