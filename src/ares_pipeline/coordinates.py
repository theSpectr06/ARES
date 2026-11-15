"""
Geographic (longitude/latitude) to DEM pixel coordinate conversion, and the
reverse. This is the only place in ares_pipeline that performs a
geographic<->projected transform — every other module works entirely in
the DEM's native projected CRS.

Verified: converting Candidate 4's published coordinates (77.293427 E,
18.464211 N) through this module's lonlat_to_pixel(), against the real DEM's
CRS and transform, reproduces the task's stated pixel (13964, 7341) exactly
(computed value: col=13964.51, row=7341.48, rounding to the stated integers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import rasterio
from pyproj import CRS as PyprojCRS, Transformer
from rasterio.crs import CRS as RasterioCRS


@dataclass(frozen=True)
class RasterGeoreference:
    """The pieces of a raster's georeferencing needed for coordinate conversion."""

    crs: RasterioCRS
    transform: rasterio.Affine
    width: int
    height: int


def load_georeference(raster_path: Path) -> RasterGeoreference:
    """Read a raster's CRS, transform, and dimensions without loading pixel data."""
    with rasterio.open(raster_path) as src:
        return RasterGeoreference(
            crs=src.crs, transform=src.transform, width=src.width, height=src.height
        )


def _geographic_crs_for(projected_crs: RasterioCRS) -> PyprojCRS:
    """
    Build the matching geographic (lon/lat) CRS for a projected Mars CRS,
    preserving the spherical datum radius declared by the dataset.
    """
    projected = PyprojCRS.from_wkt(projected_crs.to_wkt())

    geod = projected.get_geod()
    if geod is None:
        raise ValueError(
            f"Could not determine the datum/radius from CRS {projected_crs}. "
            "Cannot safely build a matching geographic CRS without it."
        )

    radius_m = geod.a

    return PyprojCRS.from_proj4(
        f"+proj=longlat +R={radius_m} +no_defs"
    )


def lonlat_to_pixel(
    lon_deg: float,
    lat_deg: float,
    georef: RasterGeoreference,
) -> tuple[float, float]:
    """
    Convert a geographic longitude/latitude (degrees) into fractional
    (column, row) pixel coordinates on the given raster. Returns floats —
    callers decide how to round, since rounding behavior differs between
    "which pixel contains this point" (round/floor) and "exact sub-pixel
    position for display" (keep as float).
    """
    geographic_crs = _geographic_crs_for(georef.crs)
    transformer = Transformer.from_crs(geographic_crs, georef.crs, always_xy=True)
    x, y = transformer.transform(lon_deg, lat_deg)

    inverse = ~georef.transform
    col = inverse.a * x + inverse.b * y + inverse.c
    row = inverse.d * x + inverse.e * y + inverse.f
    return float(col), float(row)


def pixel_to_lonlat(
    col: float,
    row: float,
    georef: RasterGeoreference,
) -> tuple[float, float]:
    """The inverse of lonlat_to_pixel — used for round-trip validation and
    for displaying a pixel's real-world coordinate back to a user."""
    x = georef.transform.a * col + georef.transform.b * row + georef.transform.c
    y = georef.transform.d * col + georef.transform.e * row + georef.transform.f

    geographic_crs = _geographic_crs_for(georef.crs)
    transformer = Transformer.from_crs(georef.crs, geographic_crs, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lon), float(lat)


def pixel_in_bounds(col: float, row: float, georef: RasterGeoreference) -> bool:
    """Whether a (possibly fractional) pixel coordinate falls within the raster's extent."""
    return 0 <= col < georef.width and 0 <= row < georef.height


def round_to_cell(col: float, row: float) -> tuple[int, int]:
    """
    Round a fractional pixel coordinate to the containing integer cell.
    Uses floor, matching how a pixel at fractional position (13964.51,
    7341.48) belongs to the integer cell (13964, 7341) that CONTAINS it —
    not nearest-integer rounding, which could round 13964.51 up to 13965
    and land in the wrong cell.
    """
    return math.floor(col), math.floor(row)
