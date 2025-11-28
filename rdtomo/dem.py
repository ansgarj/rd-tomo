import laspy
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import rasterio 
from pyproj import Transformer

from .utils import leap_seconds, warn
from .transformers import geo_to_ecef, ecef_to_geo, change_rf
from .config import Settings, LOCAL
from .manager import build_vrt

# Get elevation from DEMs for a point
def elevation(lat: float, lon: float, dem_path: Path|str = None) -> float | None:
    """Returns the elevation for a specific point from the highest resolved DEM at that point,
    or from the user specified DEM."""
    def  get_dem() -> tuple[np.ndarray, rasterio.DatasetReader]:
        def _check_file_type(filename: Path) -> str:
            ext = filename.suffix.lower()
            if ext in [".tif", ".tiff"]:
                return "TIFF"
            elif ext == ".vrt":
                return "VRT"
            else:
                return "Unknown"
        file_type = _check_file_type(dem_path)
        def _point_in_bounds(src: rasterio.DatasetReader) -> bool:
            bounds = src.bounds
            try:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                return bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top
            except Exception:
                return False

        import xml.etree.ElementTree as ET
        import os

        def _normalize_path(path: Path) -> Path:
            """
            Normalize a path string:
            - Expand environment variables and user home (~)
            - Resolve symlinks if it's a real filesystem path
            - Leave GDAL virtual paths (/vsizip/, /vsicurl/, etc.) untouched
            """
            path_str = str(path)
            if path_str.startswith("/vsi"):  # GDAL virtual path
                return path_str
            # Expand environment variables and ~
            expanded = os.path.expandvars(os.path.expanduser(path_str))
            # Resolve if possible
            return Path(expanded).resolve()

        def _find_raster_in_vrt(vrt_path: Path, lat: float, lon: float) -> str | None:
            tree = ET.parse(vrt_path)
            root = tree.getroot()

            for source in root.iter("SourceFilename"):
                raster_name = source.text.strip()
                relative = source.attrib.get("relativeToVRT", "1") == "1"

                # Compute full path
                if relative:
                    raster_path = vrt_path.resolve().parent / raster_name
                else:
                    raster_path = Path(raster_name)

                # Normalize path (handles env vars, symlinks, GDAL paths)
                full_path = _normalize_path(raster_path)

                try:
                    with rasterio.open(full_path) as src:
                        if _point_in_bounds(src, lat, lon):
                            return full_path
                except Exception:
                    continue

            return None

        if file_type == "TIFF":
            with rasterio.open(dem_path) as src:
                if _point_in_bounds(src, lat, lon):
                    dem = src.read(1)
                    return dem, src
                else:
                    return np.array([]), None
        
        elif file_type == "VRT":
            raster_path = _find_raster_in_vrt(dem_path, lat, lon)
            if raster_path:
                with rasterio.open(raster_path) as src:
                    dem = src.read(1)
                    return dem, src
            else:
                return np.array([]), None
        
        else:
            return np.array([]), None
    
    dem_path = Path(dem_path)
    if dem_path.is_file():
        dem, src = get_dem()
    else:
        vrt_path = LOCAL / "DEM.vrt"
        dem_path = build_vrt(vrt_path, Settings().DEMS)
        
        dem, src = get_dem()    

    if src:
        x, y = src.index(lon, lat)
        return dem[x,y]
    else:
        warn(f"No DEM found for coordinates ({lat}, {lon})")
        return None
 

def las_acquisition_time(las_path: str, reference_date: datetime) -> datetime:
    """
    Extract median acquisition timestamp from LAS file using GPS time.
    Auto-detects GPS time format and converts to UTC.

    Returns:
    - epoch as fractional year
    """
    las = laspy.read(las_path)
    gps_times = las.gps_time
    if gps_times.size == 0:
        raise ValueError("No GPS time data found in LAS file.")
    
    gps_median = float(np.median(gps_times))
    gps_epoch = datetime(1980, 1, 6)
    expected_abs = (reference_date - gps_epoch).total_seconds()

    # Detect format
    if gps_median > 1e9:
        # Absolute GPS time
        naive_time = gps_epoch + timedelta(seconds=gps_median)
        fmt = "absolute"
    elif gps_median < 604800:
        # Seconds-of-week
        gps_week = int((reference_date - gps_epoch).days // 7)
        gps_week_start = gps_epoch + timedelta(weeks=gps_week)
        naive_time = gps_week_start + timedelta(seconds=gps_median)
        fmt = "seconds-of-week"
    else:
        # Check for vendor offset (e.g., minus 1e9)
        diff = abs(expected_abs - gps_median)
        if abs(diff - 1e9) < 5e7:  # tolerance ~50 million seconds (~1.5 years)
            naive_time = gps_epoch + timedelta(seconds=gps_median + 1e9)
            fmt = "offset(+1e9)"
        else:
            raise ValueError(f"Unknown GPS time format: median={gps_median}, diff={diff}")
    
    # Apply leap second correction
    utc_time = naive_time - timedelta(seconds=leap_seconds(naive_time))
    utc_time = utc_time.replace(tzinfo=timezone.utc)

    # Convert to fractional year
    year = utc_time.year
    start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    year_length = (end_of_year - start_of_year).total_seconds()
    seconds_into_year = (utc_time - start_of_year).total_seconds()

    epoch = year + seconds_into_year / year_length

    # Optional: print detected format for debugging
    # print(f"Detected GPS time format: {fmt}")
    
    return epoch

import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from scipy.interpolate import griddata

def dem_warp(dem_path, geoid_path, out_path, epoch, output_res=None) -> np.ndarray:
    """
    Apply Helmert transformation to DEM with orthometric heights and warp to new grid.
    Output is ellipsoidal heights in EPSG:4326.
    
    Parameters:
        dem_path: Path to DEM raster (orthometric heights).
        geoid_path: Path to geoid raster (global EPSG:4326).
        out_path: Output raster path.
        epoch: Epoch for Helmert transformation (float or datetime).
        output_res: Resolution in degrees (tuple: (lon_res, lat_res)). If None, match original approx.
    """
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype(np.float64)
        nodata = dem_src.nodata
        transform = dem_src.transform
        profile = dem_src.profile
        height, width = dem.shape

        # Resample geoid to DEM grid
        with rasterio.open(geoid_path) as geoid_src:
            geoid_resampled = np.empty_like(dem, dtype=np.float64)
            reproject(
                source=rasterio.band(geoid_src, 1),
                destination=geoid_resampled,
                src_transform=geoid_src.transform,
                src_crs=geoid_src.crs,
                dst_transform=transform,
                dst_crs=dem_src.crs,
                resampling=Resampling.bilinear
            )

        # Compute original lon/lat grid
        rows, cols = np.indices(dem.shape)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        lon = np.array(xs).flatten()
        lat = np.array(ys).flatten()
        ortho_h = dem.flatten()
        geoid_h = geoid_resampled.flatten()

        # Mask nodata and NaNs
        if nodata is None:
            mask = ~np.isnan(ortho_h) & ~np.isnan(geoid_h)
        else:
            mask = (ortho_h != nodata) & ~np.isnan(ortho_h) & ~np.isnan(geoid_h)

        # Convert orthometric -> ellipsoidal
        ellipsoid_h = ortho_h[mask] + geoid_h[mask]

        # Apply transformations
        X, Y, Z = geo_to_ecef(lon[mask], lat[mask], ellipsoid_h, rf="SWEREF99")
        Xh, Yh, Zh = change_rf("SWEREF99", "ITRF2020", X, Y, Z, epoch=epoch)
        lon_new, lat_new, ellipsoid_h_new = ecef_to_geo(Xh, Yh, Zh, rf="ITRF2020")

        # Remove NaNs before interpolation
        valid = ~np.isnan(lon_new) & ~np.isnan(lat_new) & ~np.isnan(ellipsoid_h_new)
        lon_new, lat_new, ellipsoid_h_new = lon_new[valid], lat_new[valid], ellipsoid_h_new[valid]


        # Determine output grid
        if output_res is None:
            # Approximate resolution from original
            lon_res = abs(transform.a)
            lat_res = abs(transform.e)
        else:
            lon_res, lat_res = output_res

        # Compute new bounds
        min_lon, max_lon = lon_new.min(), lon_new.max()
        min_lat, max_lat = lat_new.min(), lat_new.max()

        out_width = int((max_lon - min_lon) / lon_res)
        out_height = int((max_lat - min_lat) / lat_res)

        out_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, out_width, out_height)

        # Interpolate onto new grid
        grid_lon = np.linspace(min_lon, max_lon, out_width)
        grid_lat = np.linspace(min_lat, max_lat, out_height)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

        fill_val = np.nan if nodata is None else nodata

        out_data = griddata(
            points=(lon_new, lat_new),
            values=ellipsoid_h_new,
            xi=(grid_lon_mesh, grid_lat_mesh),
            method='linear',
            fill_value=fill_val
)

        # Write output raster
        profile.update({
            "crs": "EPSG:4326",
            "transform": out_transform,
            "width": out_width,
            "height": out_height,
            "dtype": "float64",
            "nodata": fill_val
        })

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_data, 1)

    print(f"Warped ellipsoidal DEM written to {out_path}")
    return out_data

