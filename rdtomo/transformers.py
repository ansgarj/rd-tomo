from pyproj import Transformer
from datetime import datetime, timezone
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from .utils import warn
from .manager import resource
from .config import Settings

# Realizations
def _itrf20_to_etrf14(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ITRF2020 realization to ETRF2014. This is the first step in the NKG2020 transformation
    from ITRF2020 to Nat. ETRS89 in the Nordic region (t_r = target epoch of final transformation):
    - SWEREF99 in Sweden (ETRF97): t_r = 1999.5
    - EUREF89 in Norway (ETRF93): t_r = 1995.0
    - LKS-94 in Lithuania (ETRF2000): t_r = 2003.75
    - LKS-92 in Latvia (ETRF89): t_r = 1992.75
    - EUREF-FIN in Finland (ETRF96): t_r = 1997.0
    - EUREF-EST97 in Estonia (ETRF96): t_r = 1997.56
    - EUREF-DK94 in Denmark (ETRF92): t_r = 2015.829
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)
    
    Coordinate operation 4D EPSG:10587"""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 3:
        # Get time coordinate from epoch parameter
        if isinstance(epoch, datetime):
            # Normalize to UTC
            if epoch.tzinfo is None:
                epoch = epoch.replace(tzinfo=timezone.utc)
            else:
                epoch = epoch.astimezone(timezone.utc)

            # Convert to fractional year
            year = epoch.year
            start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
            end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            year_length = (end_of_year - start_of_year).total_seconds()
            seconds_into_year = (epoch - start_of_year).total_seconds()

            epoch = year + seconds_into_year / year_length
        
        coordinates = (*coordinates, np.full_like(coordinates[0], epoch))

    proj_str = ("+proj=helmert "
        "+x=-0.0014 +y=-0.0009 +z=0.0014 "
        "+rx=0.00221 +ry=0.013806 +rz=-0.02002 +s=-0.00042 "
        "+dx=0 +dy=-0.0001 +dz=0.0002 "
        "+drx=8.5e-05 +dry=0.000531 +drz=-0.00077 +ds=0 "
        f"+t_epoch=2015 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)

def _etrf14_to_itrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF2014 realization to ITRF2020. This is the final step in the inverse NKG2020 transformation
    from Nat. ETRS89 in the Nordic region to ITRF2020. The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)
    
    Coordinate operation 4D EPSG:10587 (inverse)"""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 3:
        # Get time coordinate from epoch parameter
        if isinstance(epoch, datetime):
            # Normalize to UTC
            if epoch.tzinfo is None:
                epoch = epoch.replace(tzinfo=timezone.utc)
            else:
                epoch = epoch.astimezone(timezone.utc)

            # Convert to fractional year
            year = epoch.year
            start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
            end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            year_length = (end_of_year - start_of_year).total_seconds()
            seconds_into_year = (epoch - start_of_year).total_seconds()

            epoch = year + seconds_into_year / year_length
        
        coordinates = (*coordinates, np.full_like(coordinates[0], epoch))

    proj_str = ("+inv +proj=helmert "
        "+x=-0.0014 +y=-0.0009 +z=0.0014 "
        "+rx=0.00221 +ry=0.013806 +rz=-0.02002 +s=-0.00042 "
        "+dx=0 +dy=-0.0001 +dz=0.0002 "
        "+drx=8.5e-05 +dry=0.000531 +drz=-0.00077 +ds=0 "
        f"+t_epoch=2015 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)

def _etrf14_to_etrf97(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF2014 realization to ETRF97 at epoch 2000.0. This is the third step in the NKG2020 transformation
    from ITRF2020 to SWEREF99. Helmert parameters are taken from the NKG2020 paper."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")

    proj_str = (
        "+proj=helmert "
        "+x=0.03054 +y=0.04606 +z=-0.07944 "
        "+rx=0.00141958 +ry=0.00015132 +rz=0.00150337 "
        "+s=0.003002 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)
    
def _etrf97_to_etrf14(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF97 realization to ETRF2014 at epoch 2000.0. This is the second step in the 
    inverse NKG2020 transformation from SWEREF99 to ITRF2020. Helmert parameters are taken from the NKG2020 paper."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")

    proj_str = (
        "+inv +proj=helmert "
        "+x=0.03054 +y=0.04606 +z=-0.07944 "
        "+rx=0.00141958 +ry=0.00015132 +rz=0.00150337 "
        "+s=0.003002 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)

def _etrf14_to_etrf96(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], code: str = "FIN") -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF2014 realization to ETRF97 at epoch 2000.0. This is the third step in the NKG2020 transformation
    from ITRF2020 to EUREF-FIN. Helmert parameters are taken from the NKG2020 paper.
    
    The code parameter can be used to specify area of interest: Finland (code='FIN') or Estonia (code='EST')"""
    if code not in {"FIN", "EST"}:
        raise ValueError("Specify area of interest: Finland (code='FIN') or Estonia (code='EST')")
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")
    if code == "FIN":
        # Helmert parameters for Finland
        proj_str = (
            "+proj=helmert "
            "+x=0.15651 +y=-0.10993 +z=-0.10935 "
            "+rx=-0.00312861 +ry=-0.00378935 +rz=0.00403512 "
            "+s=0.005290 +convention=position_vector"
        )
    else:
        # Helmert parameters for Estonia
        proj_str = (
            "+proj=helmert "
            "+x=-0.05027 +y=-0.11595 +z=0.03012 "
            "+rx=-0.00310814  +ry=0.00457237 +rz=0.00472406 "
            "+s=0.003191 +convention=position_vector"
        )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)
    
def _etrf96_to_etrf14(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], code: str = "FIN") -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF97 realization to ETRF2014 at epoch 2000.0. This is the second step in the 
    inverse NKG2020 transformation from EUREF-FIN to ITRF2020. Helmert parameters are taken from the NKG2020 paper.
    
    The code parameter can be used to specify area of interest: Finland (code='FIN') or Estonia (code='EST')"""
    if code not in {"FIN", "EST"}:
        raise ValueError("Specify area of interest: Finland (code='FIN') or Estonia (code='EST')")
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")
    if code == "FIN":
        # Helmert parameters for Finland
        proj_str = (
            "+inv +proj=helmert "
            "+x=0.15651 +y=-0.10993 +z=-0.10935 "
            "+rx=-0.00312861 +ry=-0.00378935 +rz=0.00403512 "
            "+s=0.005290 +convention=position_vector"
        )
    else:
        # Helmert parameters for Estonia
        proj_str = (
            "+inv +proj=helmert "
            "+x=-0.05027 +y=-0.11595 +z=0.03012 "
            "+rx=-0.00310814  +ry=0.00457237 +rz=0.00472406 "
            "+s=0.003191 +convention=position_vector"
        )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)

def _etrf14_to_etrf92(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF2014 realization to ETRF92 at epoch 2000.0. This is the third step in the NKG2020 transformation
    from ITRF2020 to EUREF-DK94. Helmert parameters are taken from the NKG2020 paper."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")

    proj_str = (
        "+proj=helmert "
        "+x=0.66818 +y=0.04453 +z=-0.45049 "
        "+rx=0.00312883 +ry=-0.02373423 +rz=0.00442969 "
        "+s=-0.003136 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)
    
def _etrf92_to_etrf14(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF92 realization to ETRF2014 at epoch 2000.0. This is the second step in the 
    inverse NKG2020 transformation from EUREF-DK94 to ITRF2020. Helmert parameters are taken from the NKG2020 paper."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")

    proj_str = (
        "+inv +proj=helmert "
        "+x=0.66818 +y=0.04453 +z=-0.45049 "
        "+rx=0.00312883 +ry=-0.02373423 +rz=0.00442969 "
        "+s=-0.003136 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)

def _etrf14_to_etrf00(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF2014 realization to ETRF2000 at epoch 2000.0. This is the third step in the NKG2020 transformation
    from ITRF2020 to LKS-94. Helmert parameters are taken from the NKG2020 paper."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")

    proj_str = (
        "+proj=helmert "
        "+x=0.36749 +y=0.14351 +z=-0.18472 "
        "+rx=0.00479140  +ry=-0.01027566  +rz=0.0276102 "
        "+s=-0.003684 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)
    
def _etrf00_to_etrf14(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF2000 realization to ETRF2014 at epoch 2000.0. This is the second step in the 
    inverse NKG2020 transformation from LKS-94 to ITRF2020. Helmert parameters are taken from the NKG2020 paper."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")

    proj_str = (
        "+inv +proj=helmert "
        "+x=0.36749 +y=0.14351 +z=-0.18472 "
        "+rx=0.00479140  +ry=-0.01027566  +rz=0.0276102 "
        "+s=-0.003684 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)

def _etrf14_to_etrf89(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF2014 realization to ETRF2000 at epoch 2000.0. This is the third step in the NKG2020 transformation
    from ITRF2020 to LKS-92. Helmert parameters are taken from the NKG2020 paper."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")

    proj_str = (
        "+proj=helmert "
        "+x=0.09745 +y=-0.69388 +z=0.52901 "
        "+rx=-0.01920690  +ry=0.01043272  +rz=0.02327169 "
        "+s=-0.049663 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)
    
def _etrf89_to_etrf14(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF2000 realization to ETRF2014 at epoch 2000.0. This is the second step in the 
    inverse NKG2020 transformation from LKS-92 to ITRF2020. Helmert parameters are taken from the NKG2020 paper."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")

    proj_str = (
        "+inv +proj=helmert "
        "+x=0.09745 +y=-0.69388 +z=0.52901 "
        "+rx=-0.01920690  +ry=0.01043272  +rz=0.02327169 "
        "+s=-0.049663 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)

def _etrf14_to_etrf93(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF2014 realization to ETRF93 at epoch 2000.0. This is the third step in the NKG2020 transformation
    from ITRF2020 to EUREF89. Corrections taken from cdn.proj.org."""

    def _extract_corrections(lon: float|np.ndarray, lat: float|np.ndarray) -> tuple[float|np.ndarray, ...]:
        """
        Extract corrections in the form of translations in ECEF coordinates from the internal
        no_kv_NKGETRF14_EPSG7922_2000 file. This is used instead of a Helmert transformation
        for EUREF89 in Norway.
        
        Returns:
            - X translation
            - Y translation
            - Z translation
        """
        # Convert to list of 2 tuples
        if isinstance(lat, (np.ndarray, list)):
            coordinates = list(zip(lon, lat))  # handles (lon_array, lat_array)
        else:
            coordinates = [(lon, lat)]     # handles (lon, lat)
        
        # Read velocities
        with resource(None, "NKG_CORR") as corr_raster:
            with rasterio.open(corr_raster) as src:
                # Create a VRT for on-the-fly reprojection and bilinear interpolation
                with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
                    # Sample all points at once
                    samples = list(vrt.sample(coordinates))
                    data = np.array(samples)  # shape (n_points, bands)
                    
                    # Replace masked values or invalid with NaN
                    if np.ma.is_masked(data):
                        data = data.filled(np.nan)

        # Returns shape (n_points, 3)      
        return data
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")
    
    lon, lat, _ = Transformer.from_crs("EPSG:8401", "EPSG:8403", always_xy=True).transform(*coordinates)

    # Shape (n_points, 3)
    translated_coords = np.asarray(coordinates).T + _extract_corrections(lon, lat)
    
    # Return 3-tuple
    if translated_coords.shape[0] == 1:
        return tuple(translated_coords.ravel())
    else:
        return tuple(translated_coords.T)
  
def _etrf93_to_etrf14(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in ETRF93 realization to ETRF2014 at epoch 2000.0. This is the second step in the 
    inverse NKG2020 transformation from EUREF89 to ITRF2020. Corrections taken from cdn.proj.org"""

    def _extract_corrections(lon: float|np.ndarray, lat: float|np.ndarray) -> tuple[float|np.ndarray, ...]:
        """
        Extract corrections in the form of translations in ECEF coordinates from the internal
        no_kv_NKGETRF14_EPSG7922_2000 file. This is used instead of a Helmert transformation
        for EUREF89 in Norway.
        
        Returns:
            - X translation
            - Y translation
            - Z translation
        """
        # Convert to list of 2 tuples
        if isinstance(lat, (np.ndarray, list)):
            coordinates = list(zip(lon, lat))  # handles (lon_array, lat_array)
        else:
            coordinates = [(lon, lat)]     # handles (lon, lat)
        
        # Read velocities
        with resource(None, "NKG_CORR") as corr_raster:
            with rasterio.open(corr_raster) as src:
                # Create a VRT for on-the-fly reprojection and bilinear interpolation
                with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
                    # Sample all points at once
                    samples = list(vrt.sample(coordinates))
                    data = np.array(samples)  # shape (n_points, bands)
                    
                    # Replace masked values or invalid with NaN
                    if np.ma.is_masked(data):
                        data = data.filled(np.nan)

        # Returns shape (n_points, 3)      
        return data
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")

    lon, lat, _ = Transformer.from_crs("EPSG:7922", "EPSG:7923", always_xy=True).transform(*coordinates)

    translated_coords = np.asarray(coordinates).T - _extract_corrections(lon, lat)
    
    # Return 3-tuple
    if translated_coords.shape[0] == 1:
        return tuple(translated_coords.ravel())
    else:
        return tuple(translated_coords.T)

def _deform(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], t_r: float|datetime|npt.NDArray[np.datetime64], rf: str, epoch: float|datetime|npt.NDArray[np.datetime64]|None = None) -> tuple[float|np.ndarray, ...]:
    """Model intraplate deformations with velocities from NKG_RF17vel, this is the second step as well as the final step
    of the NKG2020 transformation to Nat. ETRS89 in the Nordic region (t_r = target epoch of final transformation):
    - SWEREF99 in Sweden (ETRF97): t_r = 1999.5
    - EUREF89 in Norway (ETRF93): t_r = 1995.0
    - LKS 94 in Lithuania (ETRF2000): t_r = 2003.75
    - LKS 92 in Latvia (ETRF89): t_r = 1992.75
    - EUREF-FIN in Finland (ETRF96): t_r = 1997.0
    - EUREF-EST97 in Estonia (ETRF96): t_r = 1997.56
    - EUREF-DK94 in Denmark (ETRF92): t_r = 2015.829
    
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter, epoch (accepts datetime objects, UTC timezone).
    
    The rf parameter specifies the reference frame: ETRF2014 for the second step of the KGT2020 and the target
    ETRF implementation in the final step."""

    def _extract_velocities(lon: float|np.ndarray, lat: float|np.ndarray) -> tuple[float|np.ndarray, ...]:
        """
        Extract ENU velocities from internal NKG_RF17vel for given coordinates.
        This is used in the second and final steps of the NKG2020 transformation
        from ITRF2020 to Nat. ETRS89 in the Nordic region (t_r = target epoch of final transformation):
        - SWEREF99 in Sweden (ETRS97): t_r = 1999.5
        - EUREF89 in Norway (ETRS93): t_r = 1995.0
        - LKS 94 in Lithuania (ETRS2000): t_r = 2003.75
        - LKS 92 in Latvia (ETRS89): t_r = 1992.75
        - EUREF-FIN in Finland (ETRS96): t_r = 1997.0
        - EUREF-EST97 in Estonia (ETRS96): t_r = 1997.56
        - EUREF-DK94 in Denmark (ETRS92): t_r = 2015.829
        
        Returns:
            - X velocities
            - Y velocities
            - Z velocities
        """
        # Convert to list of 2 tuples
        if isinstance(lat, (np.ndarray, list)):
            coordinates = list(zip(lon, lat))  # handles (lon_array, lat_array)
        else:
            coordinates = [(lon, lat)]     # handles (lon, lat)
        
        # Read velocities
        with resource(None, "NKG_VEL") as vel_raster:
            with rasterio.open(vel_raster) as src:
                # Create a VRT for on-the-fly reprojection and bilinear interpolation
                with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
                    # Sample all points at once
                    samples = list(vrt.sample(coordinates))
                    data = np.array(samples)  # shape (n_points, bands)
                    
                    # Replace masked values or invalid with NaN
                    if np.ma.is_masked(data):
                        data = data.filled(np.nan)

                    # Convert from mm/year to m/year
                    data = data * 1e-3 

        # Returns shape (n_points, 3) converted to ECEF        
        return (ecef_to_enu(lon, lat, inverse=True) @ data[..., None]).squeeze()
    
    valid_rfs = ["ETRF2014", "ETRF97", "ETRF93", "ETRF2000", "ETRF89", "ETRF96", "ETRF92"]

    if rf not in valid_rfs:
        raise ValueError(f"Invalid reference frame {rf}. Valid reference frames: {valid_rfs}.")

    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 4:
        # Extract epoch 
        epoch = coordinates[3]

        # Extract spatial coordinates
        coordinates = coordinates[0:3]

    # Resolve datetime epoch
    if isinstance(epoch, datetime):
        # Normalize to UTC
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        else:
            epoch = epoch.astimezone(timezone.utc)

        # Convert to fractional year
        year = epoch.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        year_length = (end_of_year - start_of_year).total_seconds()
        seconds_into_year = (epoch - start_of_year).total_seconds()

        epoch = year + seconds_into_year / year_length

    # Resolve t_r
    if isinstance(t_r, datetime):
        # Normalize to UTC
        if t_r.tzinfo is None:
            t_r = t_r.replace(tzinfo=timezone.utc)
        else:
            t_r = t_r.astimezone(timezone.utc)

        # Convert to fractional year
        year = epoch.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        year_length = (end_of_year - start_of_year).total_seconds()
        seconds_into_year = (epoch - start_of_year).total_seconds()

        t_r = year + seconds_into_year / year_length

    match rf:
        case "ETRF2014":
            lon, lat, _ = Transformer.from_crs("EPSG:8401", "EPSG:8403", always_xy=True).transform(*coordinates)
        case "ETRF2000":
            lon, lat, _ = Transformer.from_crs("EPSG:7930", "EPSG:7931", always_xy=True).transform(*coordinates)
        case "ETRF97":
            lon, lat, _ = Transformer.from_crs("EPSG:7928", "EPSG:7929", always_xy=True).transform(*coordinates)
        case "ETRF96":
            lon, lat, _ = Transformer.from_crs("EPSG:7926", "EPSG:7927", always_xy=True).transform(*coordinates)
        case "ETRF93":
            lon, lat, _ = Transformer.from_crs("EPSG:7922", "EPSG:7923", always_xy=True).transform(*coordinates)
        case "ETRF92":
            lon, lat, _ = Transformer.from_crs("EPSG:7920", "EPSG:7921", always_xy=True).transform(*coordinates)
        case "ETRF89":
            lon, lat, _ = Transformer.from_crs("EPSG:7914", "EPSG:7915", always_xy=True).transform(*coordinates)
        
    # Shape (n_points, 3)
    deformed_coords = np.asarray(coordinates).T + np.asarray(t_r - epoch).reshape(-1,1) * _extract_velocities(lon, lat)
    
    # Return 3-tuple
    if deformed_coords.shape[0] == 1:
        return tuple(deformed_coords.ravel())
    else:
        return tuple(deformed_coords.T)

def _itrf20_to_etrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in the ITRF2020 reference frame to ETRF2020.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)
    
    Coordinate operation 4D EPSG:10573"""

    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 3:
        # Get time coordinate from epoch parameter
        if isinstance(epoch, datetime):
            # Normalize to UTC
            if epoch.tzinfo is None:
                epoch = epoch.replace(tzinfo=timezone.utc)
            else:
                epoch = epoch.astimezone(timezone.utc)

            # Convert to fractional year
            year = epoch.year
            start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
            end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            year_length = (end_of_year - start_of_year).total_seconds()
            seconds_into_year = (epoch - start_of_year).total_seconds()

            epoch = year + seconds_into_year / year_length
        
        coordinates = (*coordinates, np.full_like(coordinates[0], epoch))

    proj_str = ("+proj=helmert "
        "+x=0 +y=0 +z=0 "
        "+rx=0.002236 +ry=0.013494 +rz=-0.019578 +s=0 "
        "+dx=0 +dy=0 +dz=0 "
        "+drx=8.6e-05 +dry=0.000519 +drz=-0.000753 +ds=0 "
        "+t_epoch=2015 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)[0:3]

def _etrf20_to_itrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms between ECEF coordinates in the ETRF2020 reference frame to ITRF2020.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)
    
    Coordinate operation 4D EPSG:10573 (inverse)"""

    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 3:
        # Get time coordinate from epoch parameter
        if isinstance(epoch, datetime):
            # Normalize to UTC
            if epoch.tzinfo is None:
                epoch = epoch.replace(tzinfo=timezone.utc)
            else:
                epoch = epoch.astimezone(timezone.utc)

            # Convert to fractional year
            year = epoch.year
            start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
            end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            year_length = (end_of_year - start_of_year).total_seconds()
            seconds_into_year = (epoch - start_of_year).total_seconds()

            epoch = year + seconds_into_year / year_length
        
        coordinates = (*coordinates, np.full_like(coordinates[0], epoch))

    proj_str = ("+inv +proj=helmert "
        "+x=0 +y=0 +z=0 "
        "+rx=0.002236 +ry=0.013494 +rz=-0.019578 +s=0 "
        "+dx=0 +dy=0 +dz=0 "
        "+drx=8.6e-05 +dry=0.000519 +drz=-0.000753 +ds=0 "
        "+t_epoch=2015 +convention=position_vector"
    )

    return Transformer.from_pipeline(proj_str).transform(*coordinates)

def _itrf20_to_sweref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the NKG2020 transformation from ITRF2020 in the given epoch to SWEREF99.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    # Pipeline
    coordinates = _itrf20_to_etrf14(*coordinates, epoch=epoch)
    coordinates = _deform(*coordinates, t_r=2000.0, rf="ETRF2014")
    coordinates = _etrf14_to_etrf97(*coordinates)
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=1999.5, rf="ETRF97")    

    return coordinates

def _sweref_to_itrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from SWEREF99 to ITRF2020 in the given epoch.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 4:
        # Extract epoch 
        epoch = coordinates[3]

        # Extract spatial coordinates
        coordinates = coordinates[0:3]

    if isinstance(epoch, datetime):
        # Normalize to UTC
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        else:
            epoch = epoch.astimezone(timezone.utc)

        # Convert to fractional year
        year = epoch.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        year_length = (end_of_year - start_of_year).total_seconds()
        seconds_into_year = (epoch - start_of_year).total_seconds()

        epoch = year + seconds_into_year / year_length

    # Broadcast epoch
    epoch = np.full_like(coordinates[0], epoch)

    # Pipeline
    coordinates = _deform(*coordinates, epoch=1999.5, t_r=2000.0, rf="ETRF97")
    coordinates = _etrf97_to_etrf14(*coordinates)
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=epoch, rf="ETRF2014")
    coordinates = _etrf14_to_itrf20(*coordinates, epoch)
    
    return coordinates

def _etrf20_to_sweref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms from ETRF2020 in the given epoch to SWEREF99, by chaining the EPSG:10573 transformation with the NKG2020.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    # Pipeline
    coordinates = _etrf20_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_sweref(*coordinates)    

    return coordinates

def _sweref_to_etrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from SWEREF99 to ETRF2020 in the given epoch, by chaining the
    inverse NKGT2020 transformation with the inverse of EPSG:10573.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    # Pipeline
    coordinates = _sweref_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_etrf20(*coordinates)
    
    return coordinates

def _itrf20_to_finref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the NKG2020 transformation from ITRF2020 in the given epoch to EUREF-FIN.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _itrf20_to_etrf14(*coordinates, epoch=epoch)
    coordinates = _deform(*coordinates, t_r=2000.0, rf="ETRF2014")
    coordinates = _etrf14_to_etrf96(*coordinates, code="FIN")
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=1997.0, rf="ETRF96")    

    return coordinates

def _finref_to_itrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from EUREF-FIN to ITRF2020 in the given epoch.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 4:
        # Extract epoch 
        epoch = coordinates[3]

        # Extract spatial coordinates
        coordinates = coordinates[0:3]

    if isinstance(epoch, datetime):
        # Normalize to UTC
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        else:
            epoch = epoch.astimezone(timezone.utc)

        # Convert to fractional year
        year = epoch.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        year_length = (end_of_year - start_of_year).total_seconds()
        seconds_into_year = (epoch - start_of_year).total_seconds()

        epoch = year + seconds_into_year / year_length

    # Broadcast epoch
    epoch = np.full_like(coordinates[0], epoch)

    # Pipeline
    coordinates = _deform(*coordinates, epoch=1997.0, t_r=2000.0, rf="ETRF96")
    coordinates = _etrf96_to_etrf14(*coordinates, code="FIN")
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=epoch, rf="ETRF2014")
    coordinates = _etrf14_to_itrf20(*coordinates, epoch)
    
    return coordinates

def _etrf20_to_finref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms from ETRF2020 in the given epoch to EUREF-FIN, by chaining the EPSG:10573 transformation with the NKG2020.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _etrf20_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_finref(*coordinates)    

    return coordinates

def _finref_to_etrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from EUREF-FIN to ETRF2020 in the given epoch, by chaining the
    inverse NKGT2020 transformation with the inverse of EPSG:10573.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    coordinates = _finref_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_etrf20(*coordinates)
    
    return coordinates

def _itrf20_to_dkref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the NKG2020 transformation from ITRF2020 in the given epoch to EUREF-DK94.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _itrf20_to_etrf14(*coordinates, epoch=epoch)
    coordinates = _deform(*coordinates, t_r=2000.0, rf="ETRF2014")
    coordinates = _etrf14_to_etrf92(*coordinates)
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=2015.829, rf="ETRF92")    

    return coordinates

def _dkref_to_itrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from EUREF-DK94 to ITRF2020 in the given epoch.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 4:
        # Extract epoch 
        epoch = coordinates[3]

        # Extract spatial coordinates
        coordinates = coordinates[0:3]

    if isinstance(epoch, datetime):
        # Normalize to UTC
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        else:
            epoch = epoch.astimezone(timezone.utc)

        # Convert to fractional year
        year = epoch.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        year_length = (end_of_year - start_of_year).total_seconds()
        seconds_into_year = (epoch - start_of_year).total_seconds()

        epoch = year + seconds_into_year / year_length

    # Broadcast epoch
    epoch = np.full_like(coordinates[0], epoch)

    # Pipeline
    coordinates = _deform(*coordinates, epoch=2015.829, t_r=2000.0, rf="ETRF92")
    coordinates = _etrf92_to_etrf14(*coordinates)
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=epoch, rf="ETRF2014")
    coordinates = _etrf14_to_itrf20(*coordinates, epoch)
    
    return coordinates

def _etrf20_to_dkref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms from ETRF2020 in the given epoch to EUREF-DK94, by chaining the EPSG:10573 transformation with the NKG2020.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _etrf20_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_dkref(*coordinates)    

    return coordinates

def _dkref_to_etrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from EUREF-DK94 to ETRF2020 in the given epoch, by chaining the
    inverse NKGT2020 transformation with the inverse of EPSG:10573.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    coordinates = _dkref_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_etrf20(*coordinates)
    
    return coordinates

def _itrf20_to_litref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the NKG2020 transformation from ITRF2020 in the given epoch to LKS-94.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _itrf20_to_etrf14(*coordinates, epoch=epoch)
    coordinates = _deform(*coordinates, t_r=2000.0, rf="ETRF2014")
    coordinates = _etrf14_to_etrf00(*coordinates)
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=2003.75, rf="ETRF2000")    

    return coordinates

def _litref_to_itrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from LKS-94 to ITRF2020 in the given epoch.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 4:
        # Extract epoch 
        epoch = coordinates[3]

        # Extract spatial coordinates
        coordinates = coordinates[0:3]

    if isinstance(epoch, datetime):
        # Normalize to UTC
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        else:
            epoch = epoch.astimezone(timezone.utc)

        # Convert to fractional year
        year = epoch.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        year_length = (end_of_year - start_of_year).total_seconds()
        seconds_into_year = (epoch - start_of_year).total_seconds()

        epoch = year + seconds_into_year / year_length

    # Broadcast epoch
    epoch = np.full_like(coordinates[0], epoch)

    # Pipeline
    coordinates = _deform(*coordinates, epoch=2003.75, t_r=2000.0, rf="ETRF2000")
    coordinates = _etrf00_to_etrf14(*coordinates)
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=epoch, rf="ETRF2014")
    coordinates = _etrf14_to_itrf20(*coordinates, epoch)
    
    return coordinates

def _etrf20_to_litref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms from ETRF2020 in the given epoch to LKS-94, by chaining the EPSG:10573 transformation with the NKG2020.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _etrf20_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_litref(*coordinates)    

    return coordinates

def _litref_to_etrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from LKS-94 to ETRF2020 in the given epoch, by chaining the
    inverse NKGT2020 transformation with the inverse of EPSG:10573.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    coordinates = _litref_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_etrf20(*coordinates)
    
    return coordinates

def _itrf20_to_latref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the NKG2020 transformation from ITRF2020 in the given epoch to LKS-92.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _itrf20_to_etrf14(*coordinates, epoch=epoch)
    coordinates = _deform(*coordinates, t_r=2000.0, rf="ETRF2014")
    coordinates = _etrf14_to_etrf89(*coordinates)
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=1992.75, rf="ETRF89")    

    return coordinates

def _latref_to_itrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from LKS-92 to ITRF2020 in the given epoch.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 4:
        # Extract epoch 
        epoch = coordinates[3]

        # Extract spatial coordinates
        coordinates = coordinates[0:3]

    if isinstance(epoch, datetime):
        # Normalize to UTC
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        else:
            epoch = epoch.astimezone(timezone.utc)

        # Convert to fractional year
        year = epoch.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        year_length = (end_of_year - start_of_year).total_seconds()
        seconds_into_year = (epoch - start_of_year).total_seconds()

        epoch = year + seconds_into_year / year_length

    # Broadcast epoch
    epoch = np.full_like(coordinates[0], epoch)

    # Pipeline
    coordinates = _deform(*coordinates, epoch=1992.75, t_r=2000.0, rf="ETRF89")
    coordinates = _etrf89_to_etrf14(*coordinates)
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=epoch, rf="ETRF2014")
    coordinates = _etrf14_to_itrf20(*coordinates, epoch)
    
    return coordinates

def _etrf20_to_latref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms from ETRF2020 in the given epoch to LKS-92, by chaining the EPSG:10573 transformation with the NKG2020.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _etrf20_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_latref(*coordinates)    

    return coordinates

def _latref_to_etrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from LKS-92 to ETRF2020 in the given epoch, by chaining the
    inverse NKGT2020 transformation with the inverse of EPSG:10573.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    coordinates = _latref_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_etrf20(*coordinates)
    
    return coordinates

def _itrf20_to_estref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the NKG2020 transformation from ITRF2020 in the given epoch to EUREF-EST97.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _itrf20_to_etrf14(*coordinates, epoch=epoch)
    coordinates = _deform(*coordinates, t_r=2000.0, rf="ETRF2014")
    coordinates = _etrf14_to_etrf96(*coordinates, code="EST")
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=1997.56, rf="ETRF96")    

    return coordinates

def _estref_to_itrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from EUREF-EST97 to ITRF2020 in the given epoch.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 4:
        # Extract epoch 
        epoch = coordinates[3]

        # Extract spatial coordinates
        coordinates = coordinates[0:3]

    if isinstance(epoch, datetime):
        # Normalize to UTC
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        else:
            epoch = epoch.astimezone(timezone.utc)

        # Convert to fractional year
        year = epoch.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        year_length = (end_of_year - start_of_year).total_seconds()
        seconds_into_year = (epoch - start_of_year).total_seconds()

        epoch = year + seconds_into_year / year_length

    # Broadcast epoch
    epoch = np.full_like(coordinates[0], epoch)

    # Pipeline
    coordinates = _deform(*coordinates, epoch=1997.56, t_r=2000.0, rf="ETRF96")
    coordinates = _etrf96_to_etrf14(*coordinates, code="EST")
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=epoch, rf="ETRF2014")
    coordinates = _etrf14_to_itrf20(*coordinates, epoch)
    
    return coordinates

def _etrf20_to_estref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms from ETRF2020 in the given epoch to EUREF-EST97, by chaining the EPSG:10573 transformation with the NKG2020.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _etrf20_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_estref(*coordinates)    

    return coordinates

def _estref_to_etrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from EUREF-EST97 to ETRF2020 in the given epoch, by chaining the
    inverse NKGT2020 transformation with the inverse of EPSG:10573.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    coordinates = _estref_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_etrf20(*coordinates)
    
    return coordinates

def _itrf20_to_noref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the NKG2020 transformation from ITRF2020 in the given epoch to EUREF89.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _itrf20_to_etrf14(*coordinates, epoch=epoch)
    coordinates = _deform(*coordinates, t_r=2000.0, rf="ETRF2014")
    coordinates = _etrf14_to_etrf93(*coordinates)
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=1995.0, rf="ETRF93")    

    return coordinates

def _noref_to_itrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from EUREF89 to ITRF2020 in the given epoch.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not (len(coordinates) == 3 or len(coordinates) == 4):
        raise ValueError(f"Expected 3 or 4 coordinates, received {len(coordinates)}: {coordinates}")
    
    if len(coordinates) == 4:
        # Extract epoch 
        epoch = coordinates[3]

        # Extract spatial coordinates
        coordinates = coordinates[0:3]
    
    if isinstance(epoch, datetime):
        # Normalize to UTC
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        else:
            epoch = epoch.astimezone(timezone.utc)

        # Convert to fractional year
        year = epoch.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        year_length = (end_of_year - start_of_year).total_seconds()
        seconds_into_year = (epoch - start_of_year).total_seconds()

        epoch = year + seconds_into_year / year_length

    # Broadcast epoch
    epoch = np.full_like(coordinates[0], epoch)

    # Pipeline
    coordinates = _deform(*coordinates, epoch=1995.0, t_r=2000.0, rf="ETRF93")
    coordinates = _etrf93_to_etrf14(*coordinates)
    coordinates = _deform(*coordinates, epoch=2000.0, t_r=epoch, rf="ETRF2014")
    coordinates = _etrf14_to_itrf20(*coordinates, epoch)
    
    return coordinates

def _etrf20_to_noref(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Transforms from ETRF2020 in the given epoch to EUREF89, by chaining the EPSG:10573 transformation with the NKG2020.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)."""
    # Pipeline
    coordinates = _etrf20_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_noref(*coordinates)    

    return coordinates

def _noref_to_etrf20(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Implements the inverse NKG2020 transformation from EUREF89 to ETRF2020 in the given epoch, by chaining the
    inverse NKGT2020 transformation with the inverse of EPSG:10573.
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone)"""
    coordinates = _noref_to_itrf20(*coordinates, epoch=epoch)
    coordinates = _itrf20_to_etrf20(*coordinates)
    
    return coordinates

def change_rf(source_rf: str, target_rf: str, *coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
    """Changes the reference frame of the coordinates. The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone). Implemented reference frames are:
    - ITRF: alias for latest ITRF realization (ITRF2020)
    - ITRF2020
    - ETRF: alias for latest ETRF realization (ETRF2020)
    - ETRF2020
    - SWEREF: alias for SWEREF99
    - SWEREF99
    - FINREF: alias for EUREF-FIN
    - EUREF-FIN
    - DKREF: alias for EUREF-DK94
    - EUREF-DK94
    - LITREF: alias for LKS-94
    - LKS-94
    - LATREF: alias for LKS-92
    - LKS-92
    - ESTREF: alias for EUREF-EST97
    - EUREF-EST97
    - NOREF: alias for EUREF89
    - EUREF89
    
    The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or
    as a separate parameter (accepts datetime objects, UTC timezone).
    
    Note that national realizations cannot be changed into eachother.
    
    Returns 3D coordinates"""

    st = Settings()
    source_rf = st.resolve_frame(source_rf)
    target_rf = st.resolve_frame(target_rf)
    if source_rf == target_rf:
        return coordinates[0:3]
    
    transformer_map = {
        "ITRF2020": {
            "ETRF2020": _itrf20_to_etrf20,
            "SWEREF99": _itrf20_to_sweref,
            "EUREF-FIN": _itrf20_to_finref,
            "EUREF-DK94": _itrf20_to_dkref,
            "LKS-94": _itrf20_to_litref,
            "LKS-92": _itrf20_to_latref,
            "EUREF-EST97": _itrf20_to_estref,
            "EUREF89": _itrf20_to_noref,
        },
        "ETRF2020": {
            "ITRF2020": _etrf20_to_itrf20,
            "SWERFEF99": _etrf20_to_sweref,
            "EUREF-FIN": _etrf20_to_finref,
            "EUREF-DK94": _etrf20_to_dkref,
            "LKS-94": _etrf20_to_litref,
            "LKS-92": _etrf20_to_latref,
            "EUREF-EST97": _etrf20_to_estref,
            "EUREF89": _etrf20_to_noref,
        },
        "SWEREF99": {
            "ITRF2020": _sweref_to_itrf20,
            "ETRF2020": _sweref_to_etrf20
        },
        "EUREF-FIN": {
            "ITRF2020": _finref_to_etrf20,
            "ETRF2020": _finref_to_etrf20
        },
        "EUREF-DK94": {
            "ITRF2020": _dkref_to_itrf20,
            "ETRF2020": _dkref_to_etrf20
        },
        "LKS-94": {
            "ITRF2020": _litref_to_itrf20,
            "ETRF2020": _litref_to_etrf20
        },
        "LKS-92": {
            "ITRF2020": _latref_to_itrf20,
            "ETRF2020": _latref_to_etrf20
        },
        "EUREF-EST97": {
            "ITRF2020": _estref_to_itrf20,
            "ETRF2020": _estref_to_etrf20
        },
        "EUREF89": {
            "ITRF2020": _noref_to_itrf20,
            "ETRF2020": _noref_to_etrf20
        }
    }

    return transformer_map[source_rf][target_rf](*coordinates, epoch=epoch)[0:3]

# Coordinates
def ecef_to_geo(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], rf: str = None) -> tuple[float|np.ndarray, ...]:
    """Transforms ECEF coordinates to geodetic. The reference frame can be specified with the rf parameter.
    If not specified it will default to the REFERENCE_FRAME: TARGET in the Settings."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")

    rf = Settings().resolve_frame(rf)

    transformer_map = { #[source_crs, target_crs]
        "ITRF2020": ["EPSG:9988", "EPSG:9989"],
        "ETRF2020": ["EPSG:10569", "EPSG:10570"],
        "SWEREF99": ["EPSG:7928", "EPSG:7929"],
        "EUREF-FIN": ["EPSG:7926", "EPSG:7927"],
        "EUREF-DK94": ["EPSG:7920", "EPSG:7921"],
        "EUREF-EST97": ["EPSG:7926", "EPSG:7927"],
        "LKS-94": ["EPSG:7930", "EPSG:7931"],
        "LKS-92": ["EPSG:7914", "EPSG:7915"],
        "EUREF89": ["EPSG:7922", "EPSG:7923"]
    }
    return Transformer.from_crs(*transformer_map[rf], always_xy=True).transform(*coordinates)

def geo_to_ecef(*coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], rf: str = None) -> tuple[float|np.ndarray, ...]:
    """Transforms geodetic coordinates to ECEF."""
    if len(coordinates) == 1:
        coordinates = coordinates[0]
    if not len(coordinates) == 3:
        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")
    
    rf = Settings().resolve_frame(rf)

    transformer_map = { #[source_crs, target_crs]
        "ITRF2020": ["EPSG:9989", "EPSG:9988"],
        "ETRF2020": ["EPSG:10570", "EPSG:10569"],
        "SWEREF99": ["EPSG:7929", "EPSG:7928"],
        "EUREF-FIN": ["EPSG:7927", "EPSG:7926"],
        "EUREF-DK94": ["EPSG:7921", "EPSG:7920"],
        "EUREF-EST97": ["EPSG:7927", "EPSG:7926"],
        "LKS-94": ["EPSG:7931", "EPSG:7930"],
        "LKS-92": ["EPSG:7915", "EPSG:7914"],
        "EUREF89": ["EPSG:7923", "EPSG:7922"]
    }

    return Transformer.from_crs(*transformer_map[rf], always_xy=True).transform(*coordinates)

def _iutm(lat: np.ndarray|float, lon: np.ndarray|float) -> tuple[np.ndarray|float, np.ndarray|float]:
    """
    Returns the UTM EPSG code for a given latitude and longitude in ITRS.
    Handles Norway and Svalbard special cases.
    """
    if not isinstance(lat, np.ndarray):
        lat = np.asarray(lat)
    if not isinstance(lon, np.ndarray):
        lon = np.asarray(lon)

    # Get reference coordinates
    ref_lon = lon.flat[0]
    ref_lat = lat.flat[0]
    # Compute base zone
    zone = int((ref_lon + 180) / 6) + 1

    # Handle Norway and Svalbard exceptions
    # Norway: Zone 32 for 56°N–64°N and 3°E–12°E
    if 56 <= ref_lat < 64 and 3 <= ref_lon < 12:
        zone = 32
    # Svalbard: Zones 31–37 for 72°N–84°N
    if 72 <= ref_lat < 84:
        if ref_lon >= 0 and ref_lon < 9:
            zone = 31
        elif ref_lon < 21:
            zone = 33
        elif ref_lon < 33:
            zone = 35
        else:
            zone = 37

    # Hemisphere and EPSG code
    return 32600 + zone if ref_lat >= 0 else 32700 + zone

def _eutm(lat: np.ndarray|float, lon: np.ndarray|float) -> tuple[np.ndarray|float, np.ndarray|float]:
    """
    Returns the projected UTM coordinates for a given latitude and longitude in ETRS89
    Handles Norway and Svalbard special cases.
    """
    if not isinstance(lat, np.ndarray):
        lat = np.asarray(lat)
    if not isinstance(lon, np.ndarray):
        lon = np.asarray(lon)

    # Get reference coordinates
    ref_lon = lon.flat[0]
    ref_lat = lat.flat[0]
    # Compute base zone
    zone = int((ref_lon + 180) / 6) + 1

    # Handle Norway and Svalbard exceptions
    # Norway: Zone 32 for 56°N–64°N and 3°E–12°E
    if 56 <= ref_lat < 64 and 3 <= ref_lon < 12:
        zone = 32
    # Svalbard: Zones 31–37 for 72°N–84°N
    if 72 <= ref_lat < 84:
        if ref_lon >= 0 and ref_lon < 9:
            zone = 31
        elif ref_lon < 21:
            zone = 33
        elif ref_lon < 33:
            zone = 35
        else:
            zone = 37

    if zone < 27 or zone > 38:
        raise ValueError(f"The first coordinate pair found: ({ref_lat}, {ref_lon}) is not on within the ETRS89 zone")

    # Hemisphere and EPSG code
    if ref_lat < 0:
        raise ValueError(f"The first coordinate pair found: ({ref_lat}, {ref_lon}) is not on the northen hemisphere")
    return 25800 + zone

def _dktm(lon: float) -> int:
    """
    Select the correct DKTM zone EPSG code based on longitude.
    
    Zones:
    - DKTM1: EPSG 4093 (central meridian 9°E)
    - DKTM2: EPSG 4094 (central meridian 10°E)
    - DKTM3: EPSG 4095 (central meridian 11.75°E)
    - DKTM4: EPSG 4096 (central meridian 15°E)
    
    Args:
        lon (float): Longitude in degrees (ETRS89/EUREF-DK94).
    
    Returns:
        int: EPSG code for the selected DKTM zone.
    """
    if lon < 9.5:
        return 4093  # DKTM1
    elif lon < 10.9:
        return 4094  # DKTM2
    elif lon < 13.5:
        return 4095  # DKTM3
    else:
        return 4096  # DKTM4
    
def geo_to_map(lat: np.ndarray|float, lon: np.ndarray|float, rf: str|None = None) -> tuple[np.ndarray|float, np.ndarray|float]:
    """Transforms from geodetic coordinates to projected map coordinates in the specified Reference Frame.
    The reference frame can be specified using the rf parameter, and if not specified will default the the
    REFERENCE_FRAMES: TARGET in the Settings. The projected map coordiantes are dependent on the Reference
    Frame:
    - ITRF2020: UTM Zones (326xx, 327xx)
    - ETRF2020: UTM Zones (258xx)
    - SWEREF99: SWEREF99 TM
    - EUREF-FIN: EUREF-FIN / TM35FIN(E,N)
    - EUREF-DK94: ETRS89 / DKTMX (X in 1-4)
    - EUREF-EST97: EST97
    - LKS-94: LKS-94 / Lithuania TM
    - LKS-92: LKS-92 / Latvia TM"""
    rf = Settings().resolve_frame(rf)

    match rf: #(source_crs, target_crs)
        case "ITRF2020":
            epsg_codes = ("EPSG:9989", f"EPSG:{_iutm(lat=lat, lon=lon)}")
        case "ETRF2020":
            epsg_codes = ("EPSG:10570", f"EPSG:{_eutm(lat=lat, lon=lon)}")
        case "SWEREF99":
            epsg_codes = ("EPSG:7929", "EPSG:3006")
        case "EUREF-FIN":
            epsg_codes = ("EPSG:7927", "EPSG:3067")
        case "EUREF-DK94": 
            epsg_codes = ("EPSG:7921", f"EPSG:{_dktm(lon)}")
        case "EUREF-EST97": 
            epsg_codes = ("EPSG:7927", "EPSG:3301")
        case "LKS-94":
            epsg_codes = ("EPSG:7931", "EPSG:3346")
        case "LKS-92": 
            epsg_codes = ("EPSG:7915", "EPSG:3059")
        case "EUREF89": 
            epsg_codes = ("EPSG:7923", f"EPSG:{_eutm(lat=lat, lon=lon)}")

    return Transformer.from_crs(*epsg_codes, always_xy=True).transform(lon, lat)
    
def ecef_to_enu(lon: float|np.ndarray, lat: float|np.ndarray, inverse: bool = False, degrees: bool = True) -> np.ndarray:
    """
    Compute ENU rotation matrices for given longitude(s) and latitude(s).
    
    Parameters:
        lon, lat: float or array-like
            Longitude and latitude values.
        degrees: bool
            If True, convert from degrees to radians.
    
    Returns shape (3, 3) if single point and (n, 3, 3) if multiple points.
    """
    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)

    if len(lon.shape) > 1 or len(lat.shape) > 1:
        raise ValueError("ecef_to_enu expects scalars or 1D arrays")
    
    if degrees:
        lon = np.radians(lon)
        lat = np.radians(lat)
    
    n = lon.size

    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    mats = np.zeros((n, 3, 3))
    mats[:, 0, :] = np.stack([-sin_lon, cos_lon, np.zeros(n)], axis=1)
    mats[:, 1, :] = np.stack([-cos_lon * sin_lat, -sin_lon * sin_lat, cos_lat], axis=1)
    mats[:, 2, :] = np.stack([cos_lon * cos_lat, sin_lon * cos_lat, sin_lat], axis=1)
    
    if inverse:
        return mats[0].T if n == 1 else np.transpose(mats, axes=(0,2,1))
    return mats[0] if n == 1 else mats
