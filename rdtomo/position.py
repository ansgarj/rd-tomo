from __future__ import annotations
from pyproj import Transformer, CRS, datadir
from datetime import datetime, timedelta, timezone
import numpy as np
import numpy.typing as npt
from typing import Iterator, TypeVar, Type
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from .config import Settings, PACKAGE_PATH
from .manager import resource
from .utils import verify_dt64, verify_td64, Angles, IndexType, warn

class ReferenceFrame:
    _frame: str
    
    @property
    def name(self) -> str:
        return self._frame
    
    @name.setter
    def name(self, value: str|ReferenceFrame) -> None:
        self._frame = ReferenceFrame(value).name

    def __init__(self, frame: str) -> ReferenceFrame:
        """A Reference Frame Object initialized by a string representing the Frame. Implemented reference frames are:
        - ITRF: alias for latest ITRF realization (ITRF2020)
        - ITRF2020
        - ETRF: alias for latest ETRF realization (ETRF2020)
        - ETRF2020
        - SWEREF99
        - EUREF89
        - EUREF-FIN
        - EUREF-DK94
        - LKS-94
        - LKS-92
        - EUREF-EST97

        A reference frame can also be initialized by a string specifying a canonical location:
        - WGS84 (ITRF)
        - EUROPE (ETRF)
        - SWEDEN (SWEREF99)
        - NORWAY (EUREF89)
        - FINLAND (EUREF-FIN)
        - DENMARK (EUREF-DK94)
        - LITHUANIA (LKS-94)
        - LATVIA (LKS-92)
        - ESTONIA (EUREF-EST97)
        
        Pos is a Reference Frame aware position representation, and DeltaPos represents a position offset, while ReferenceFrame
        handles the Reference Frames (parallels datetime objects datetime, timedelta, timezone but with array representation
        instead of scalar)."""

        if isinstance(frame, ReferenceFrame):
            self._frame = frame.name
            return
        
        self._frame = Settings().resolve_frame(frame)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.name == ReferenceFrame(other).name
        if isinstance(other, ReferenceFrame):
            return self.name == other.name
        return NotImplemented
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"ReferenceFrame({self.name})"
    
    def as_frame(self, target: str|ReferenceFrame, *coordinates: float|np.ndarray|tuple[float|np.ndarray, ...], epoch: float|datetime|None = None) -> tuple[float|np.ndarray, ...]:
        """Transforms coordinate (arrays) from this Reference Frame to the one specified by the target parameter.
        The time coordinate can be passed in the same format as the spatial coordinates (decimal year), or as a
        separate parameter (accepts datetime objects, UTC timezone).
    
        For Pos objects use the reframe method."""
        target = ReferenceFrame(target)
        if target == self:
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
        return transformer_map[self.name][target.name](*coordinates, epoch=epoch)[0:3]
        
    @property
    def ecef_epsg(self) -> int:
        """Returns the EPSG code pointing the the ECEF coordinates of this frame."""
        epsg = { 
            "ITRF2020": 9988,
            "ETRF2020": 10569,
            "SWEREF99": 4976, 
            "EUREF-FIN": 10688,
            "EUREF-DK94": 10890,
            "EUREF-EST97": 4934,
            "LKS-94": 4950,
            "LKS-92": 4948,
            "EUREF89": 10873,
        }
        return epsg[self.name]
    
    @property
    def ecef_crs(self) -> CRS:
        """Returns the pyproj.CRS object of the ECEF coordinates of this frame."""
        return CRS.from_epsg(self.ecef_epsg)
    
    @property
    def llh_epsg(self) -> int:
        """Returns the EPSG code pointing the the LLH coordinates of this frame."""
        epsg = { 
            "ITRF2020": 9989,
            "ETRF2020": 10570,
            "SWEREF99": 4977, 
            "EUREF-FIN": 10689,
            "EUREF-DK94": 10891,
            "EUREF-EST97": 4935,
            "LKS-94": 4951,
            "LKS-92": 4949,
            "EUREF89": 10874,
        }
        return epsg[self.name]
    
    @property
    def llh_crs(self) -> CRS:
        """Returns the pyproj.CRS object of the LLH coordinates of this frame."""
        return CRS.from_epsg(self.llh_epsg)
    
    @property
    def geo_epsg(self) -> int:
        """Returns the EPSG code pointing the the 2D geodetic (lat, lon) coordinates of this frame."""
        epsg = {
            "ITRF2020": 9989,
            "ETRF2020": 10570,
            "SWEREF99": 4619,
            "EUREF-FIN": 10690,
            "EUREF-DK94": 10892,
            "EUREF-EST97": 4180,
            "LKS-94": 4669,
            "LKS-92": 4661,
            "EUREF89": 10875,
        }
        
        return epsg[self.name]
    
    @property
    def geo_crs(self) -> CRS:
        """Returns the pyproj.CRS object of the 2D geodetic (lat, lon) coordinates of this frame."""
        return CRS.from_epsg(self.geo_epsg)

    def proj_epsg(self, lat: float|None = None, lon: float|None = None) -> int:
        """Returns the EPSG code of the projected coordinate system matching lon and lat coordinates.
        The coordinates can be omitted for some national Reference Frames (those having only
        one implemented national projected coordinate system).
        
        ITRF and ETRF require both lat and lon to be specified, and EUREF-DK94 require lon to be specified."""
        match self.name:
            case "ITRF2020":
                epsg = _iutm(lat=lat, lon=lon)
            case "ETRF2020":
                epsg = _eutm(lat=lat, lon=lon)
            case "SWEREF99":
                epsg = 3006
            case "EUREF-FIN":
                epsg = 3067
            case "EUREF-DK94": 
                epsg =  _dktm(lon)
            case "EUREF-EST97": 
                epsg = 3301
            case "LKS-94":
                epsg = 3346
            case "LKS-92": 
                epsg = 3059
            case "EUREF89": 
                epsg = _eutm(lat=lat, lon=lon)

        return epsg
    
    def proj_crs(self, lat: float|None = None, lon: float|None = None) -> CRS:
        """Returns the pyproj.CRS object of the of the projected coordinate system matching lon and lat
        coordinates. The coordinates can be omitted for some national Reference Frames (those having only
        one implemented national projected coordinate system).
        
        ITRF and ETRF require both lat and lon to be specified, and EUREF-DK94 require lon to be specified."""

        return CRS.from_epsg(self.proj_epsg(lat=lat,lon=lon))

    @property
    def orthometric_epsg(self) -> int:
        """Currently available only for SWEREF99."""

        match self.name:
            case "SWEREF99":
                epsg = 5628
            case _:
                raise RuntimeError("Not implemented")

        return epsg

    @property
    def orthometric_crs(self) -> CRS:
        """Currently available only for SWEREF99."""
        return CRS.from_epsg(self.orthometric_epsg)

    
    def ecef_to_geo(self, *coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
        """Transforms ECEF coordinates to geodetic (lon, lat, h) in this Reference Frame"""
        
        return Transformer.from_crs(f"EPSG:{self.ecef_epsg}", f"EPSG:{self.llh_epsg}", always_xy=True).transform(*coordinates)
    
    def geo_to_ecef(self, *coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> tuple[float|np.ndarray, ...]:
        """Transforms geodetic (lon, lat, h) coordinates to ECEF in this Reference Frame"""
        return Transformer.from_crs(f"EPSG:{self.llh_epsg}", f"EPSG:{self.ecef_epsg}", always_xy=True).transform(*coordinates)
    
    def proj(self, lat: np.ndarray|float, lon: np.ndarray|float) -> tuple[np.ndarray|float, np.ndarray|float]:
        """Projects latitude/longitude pairs to the projected map coordinates in this Reference Frame. All points
        are assumed to be in the same projected coordinate zone, matching those of the first coordinate pair."""
        if isinstance(lat, float):
            ref_lat = lat
        else:
            ref_lat = np.asarray(lat)[0]

        if isinstance(lon, float):
            ref_lon = lon
        else:
            ref_lon = np.asarray(lon)[0]
        
        return Transformer.from_crs(f"EPSG:{self.geo_epsg}", f"EPSG:{self.proj_epsg(lat=ref_lat, lon=ref_lon)}", always_xy=True).transform(lon, lat)

    def orthometric(self, *coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> float|np.ndarray:
        """Converts LLH ellipsoidal height to orthometric height."""
        datadir.append_data_dir(str(PACKAGE_PATH / "resources"))
        return Transformer.from_crs(f"EPSG:{self.llh_epsg}", f"EPSG:{self.orthometric_epsg}", always_xy=True).transform(*coordinates)[2]

    def ellipsoidal(self, *coordinates: float|np.ndarray|tuple[float|np.ndarray, ...]) -> float|np.ndarray:
        """Converts LLH orthometric height to ellipsoidal height."""
        datadir.append_data_dir(str(PACKAGE_PATH / "resources"))
        return Transformer.from_crs(f"EPSG:{self.orthometric_epsg}", f"EPSG:{self.llh_epsg}", always_xy=True).transform(*coordinates)[2]


DeltaPosType = TypeVar('DeltaPosType', bound='DeltaPos')
class DeltaPos:
    __slots__ = ("_coords",)
    _coords: npt.NDArray[np.floating]           # 3D ENU coordinates [m], shape (n,3)

    def __new__(cls: Type[DeltaPosType], *coordinates, **kwargs) -> DeltaPosType:
            if coordinates:
                first = coordinates[0]
                if isinstance(first, cls):
                    return first
            return super().__new__(cls)

    def __init__(self, *coordinates: float|npt.NDArray[np.floating]|tuple[float|np.NDArray[np.floating], ...], east: float|npt.NDArray[np.floating] = 0., north: float|npt.NDArray[np.floating] = 0., up: float|npt.NDArray[np.floating] = 0.) -> DeltaPos:
        """Initiates DeltaPos: a 3D offset represented by East, North and Up (ENU) coordinates, relative some position.
        The coordinates accepts iterable objects.

        DeltaPos objects can be added/subtracted to each other or to Pos objects (numpy broadcasting applies).
        
        Pos is a Reference Frame aware position representation, and DeltaPos represents a position offset, while ReferenceFrame
        handles the Reference Frames (parallels datetime objects datetime, timedelta, timezone but with array representation
        instead of scalar)."""
        if coordinates:
            # Get coordinates
            if len(coordinates) == 1:
                coordinates = coordinates[0]

            # Return DeltaPos object if one is passed
            if isinstance(coordinates[0], DeltaPos):
                return

            # Validate dimensions
            if isinstance(coordinates, np.ndarray):
                if coordinates.ndim == 1:
                    if coordinates.size != 3:
                        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {coordinates}")
                    coordinates = coordinates.reshape(-1, 3)
                if coordinates.ndim == 2:
                    if coordinates.shape[1] !=3:
                        raise ValueError(f"Expected 3 coordinates, received {len(coordinates)}: {(*coordinates,)}")
                else:
                    raise ValueError("Incorrectly formatted coordinates")
            elif len(coordinates) == 3:
                coordinates = np.array(coordinates, dtype=float).T
                if coordinates.ndim == 1:
                    coordinates = coordinates.reshape(-1, 3)

            self._coords = coordinates
        else:
            self._coords = np.array([[0., 0., 0.]])

        if east != 0:
            self._coords[0] = east
        
        if north != 0:
            self._coords[1] = north
        
        if up != 0:
            self._coords[2] = up

    @property
    def east(self) -> npt.NDArray[np.floating]:
        return self._coords[:,0]
    
    @property
    def north(self) -> npt.NDArray[np.floating]:
        return self._coords[:,1]
    
    @property
    def up(self) -> npt.NDArray[np.floating]:
        return self._coords[:,2]
    
    @property
    def coords(self) -> npt.NDArray[np.floating]:
        return self._coords
    
    @property
    def azimuth(self) -> Angles:
        """Returns array with azimuth angle"""
        return Angles(np.atan2(self.east, self.north))
    
    def norm(self, horizontal: bool = False, vertical: bool = False) -> np.floating|npt.NDArray[np.floating]:
        if vertical:
            return np.abs(self.up)
        i = 2 if horizontal else 3
        return np.sqrt((self.coords[:,0:i]**2).sum(axis=1))
    
    def mean(self, dtype: type = float, out: None = None) -> DeltaPos:
        """Returns DeltaPos object which is mean of current."""
        return DeltaPos(self.coords.mean(axis=0, dtype=dtype, out=out))

    def __len__(self) -> int:
        """Number of points"""
        return self.coords.shape[0]
        
    def copy(self) -> Pos:
        return DeltaPos(self.coords.copy())
    
    def join(self, other: DeltaPos|npt.ArrayLike) -> None:
        """Serializes the other object as a DeltaPos object and joins it to the current."""
        other = DeltaPos(other)
        self._coords = np.vstack((self._coords, other.coords))
    
    def __iter__(self) -> Iterator[npt.NDArray[np.floating]]:
        """Iterates over ENU coordinates"""
        return iter(self.coords)
    
    def __eq__(self, other: object) -> bool:
        """If other is serializable as a DeltaPos object: returns True if coordinates match,
        otherwise returns False."""
        try:
            other = DeltaPos(other)
        except:
            return NotImplemented    
        return self.coords == other.coords
    
    def __lt__(self, other: object) -> bool:
        """If other is serializable as a DeltaPos object: returns True if norm of this object is lesser than
        the other otherwise returns False."""
        try:
            other = DeltaPos(other)
        except:
            return NotImplemented
        return self.norm() < other.norm()
    
    def __gt__(self, other: object) -> bool:
        """If other is serializable as a DeltaPos object: returns True if norm of this object is lesser than
        the other otherwise returns False."""
        try:
            other = DeltaPos(other)
        except:
            return NotImplemented
        return self.norm() > other.norm()
    
    def __add__(self, other: object) -> DeltaPos|Pos:
        if isinstance(other, Pos):
            return Pos(other.coords + (np.transpose(other.enu_rotation, axes=(0,2,1)) @ self.coords[..., None]).squeeze(), epoch=other.t.copy(), frame=other.frame)
        try:
            other = DeltaPos(other)
        except:
            return NotImplemented
        return DeltaPos(self.coords + other.coords)
        
    def __radd__(self, other: object) -> DeltaPos|Pos:
        if isinstance(other, Pos):
            return Pos(other.coords + (np.transpose(other.enu_rotation, axes=(0,2,1)) @ self.coords[..., None]).squeeze(), epoch=other.t.copy(), frame=other.frame)
        try:
            other = DeltaPos(other)            
        except:
            return NotImplemented
        return DeltaPos(self.coords + other.coords)

    def __sub__(self, other: object) -> DeltaPos:
        try:
            other = DeltaPos(other)
        except:
            return NotImplemented
        return DeltaPos(self.coords - other.coords)
    
    def __rsub__(self, other: object) -> DeltaPos:
        try:
            other = DeltaPos(other)
        except:
            return NotImplemented
        return DeltaPos(other.coords - self.coords)
    
    def __mul__(self, other: object) -> DeltaPos:
        other = np.asarray(other, dtype=float)
        if other.ndim == 1 and other.size == len(self):
            other = other.reshape(len(self), -1)
        if other.ndim == 0 or (other.ndim == 2 and len(other) == len(self) and (other.shape[1] == 3 or other.shape[1] == 1)):
            return DeltaPos(other * self.coords)
        return NotImplemented
    
    def __rmul__(self, other: object) -> DeltaPos:
        other = np.asarray(other, dtype=float)
        if other.ndim == 1 and other.size == len(self):
            other = other.reshape(len(self), -1)
        if other.ndim == 0 or (other.ndim == 2 and len(other) == len(self) and (other.shape[1] == 3 or other.shape[1] == 1)):
            return DeltaPos(other * self.coords)
        return NotImplemented
    
    def __getitem__(self, idx: IndexType) -> DeltaPos:
        """Returns DeltaPos object with a set of coordinates determined by idx."""
        return DeltaPos(self.coords[idx])

    def __setitem__(self, idx: IndexType, value: DeltaPos|npt.NDArray[np.floating]):
        obj = DeltaPos(value)
        if len(obj) == len(self[idx]):
            self._coords = obj.coords
        else:
            raise ValueError(f"The value must match the idx, and be serializable as a DeltaPos object, not {value}")

    def __str__(self) -> str:
        if len(self) == 1:
            distance = round(self.norm()[0])
        else:
            distance = self.norm().round(3)
        return f"{distance} m"
    
    def __repr__(self) -> str:
        return f"DeltaPos({self.coords}, {f'shape={self.coords.shape}, ' if len(self) > 333 else ''})"

PosType = TypeVar('PosType', bound='Pos')
class Pos:
    __slots__ = ("frame", "_epoch", "_time", "_ecef", "_llh", "_map", "_enu")
    frame: ReferenceFrame                       # Reference frame
    _epoch: np.datetime64                       # Nominal epoch
    _time: npt.NDArray[np.timedelta64]          # Time coordinate offset from _epoch, shape (n,)
    _ecef: npt.NDArray[np.floating]             # 3D ECEF coordinate array [X, Y, Z], shape (n, 3)
    _llh: npt.NDArray[np.floating]              # 3D geodetic coordinate array [lon, lat, h], shape (n, 3)
    _map: npt.NDArray[np.floating]              # 2D projected map coordinate array [Easting, Northing], shape (n, 2)
    _enu: npt.NDArray[np.floating]              # Transformation matrices from ECEF to ENU coordinates, shape (n, 3, 3)

    def __new__(cls: Type[PosType], *args, **kwargs) -> PosType:
        if args and isinstance(args[0], Pos):
            return args[0]
        obj = super().__new__(cls)
        obj.frame = None
        obj._epoch = None
        obj._time = None
        obj._ecef = None
        obj._llh = None
        obj._map = None
        obj._enu = None
        return obj

    # Standard init: ECEF coordinates
    def __init__(self, *coordinates: npt.ArrayLike[np.floating], epoch: float|datetime|np.datetime64|str|None = None, frame: str|ReferenceFrame|None = None, geodetic: bool = False, lat_first: bool = False) -> Pos:
        """Initiates Pos in ECEF coordinates by default; the geodetic parameter can be set to True to initiate in LLH coordinates
        (default is longitude first, use lat_first to set latitude first). The 4th dimension (time) is specified as a 4th coordinate
        (datetime, np.datetime64, datetime str or decimal year), or as a time offset (timedelta, np.timedelta64, duration string or
        number of seconds) from the  epoch parameter (datetime, np.datetime64, datetime string or decimal year). If no time coordinate
        is provided, the epoch must be specified and the time offset is set to 0 for all coordinate sets. Coordinates (but not epoch)
        accepts ArrayLike objects.
        
        If the Reference Frame is not specified via the frame parameter, defaults to the most recent ITRF realization.
        
        Pos is a Reference Frame aware position representation, and DeltaPos represents a position offset, while ReferenceFrame
        handles the Reference Frames (parallels datetime objects datetime, timedelta, timezone but with array representation
        instead of scalar)."""

        # Get coordinates
        if len(coordinates) == 1:
            coordinates = coordinates[0]

                # Return Coordinate object if one is passed
        
        if isinstance(coordinates, Pos):
            return # Skip initiation
        
        # Resolve and validate Reference Frame
        if frame is None:
            self.frame = ReferenceFrame("ITRF")
        else:
            self.frame = ReferenceFrame(frame)

        # Prepare coordinates and verify spatial coordinates
        t = None
        if isinstance(coordinates, np.ndarray):
            # Validate dimensions
            if coordinates.ndim == 1:
                if coordinates.size != 3 and coordinates.size != 4:
                    raise ValueError(f"Expected 3 or 4 coordinates, received {coordinates.size}: {(*coordinates,)}")

                if coordinates.size == 4:
                    # Time coordinate
                    t = coordinates[3]
                
                # Spatial coordinates 
                coordinates = coordinates[0:3].reshape(-1,3)

            if coordinates.ndim == 2:
                if coordinates.shape[1] !=3 and coordinates.shape[1] !=4:
                    raise ValueError(f"Expected 3 or 4 coordinates, received {coordinates.shape[1]}: {(*coordinates.T,)}")
                
                if coordinates.shape[1] == 4:
                    # Time coordinate
                    t = coordinates[:, 3]

                # Spatial coordinates
                coordinates = coordinates[:,0:3]

            else:
                raise ValueError("Incorrectly formatted coordinates")
        elif len(coordinates) == 3:
            # Spatial coordinates
            coordinates = np.array(coordinates, dtype=float).T
            if coordinates.ndim == 1:
                coordinates = coordinates.reshape(-1, 3)
        elif len(coordinates) == 4:
            # Time coordinate
            t = coordinates[3]

            # Spatial coordinates
            coordinates = np.array(coordinates[0:3], dtype=float).T
            if coordinates.ndim == 1:
                coordinates = coordinates.reshape(-1, 3)

        # Assign spatial coordinates
        if geodetic:
            if lat_first:
                coordinates = coordinates[:, [1,0,2]]
            self._llh = coordinates
        else:
            self._ecef = coordinates

        # Verify epoch and assign
        if epoch:
            self._epoch = verify_dt64(epoch)
        
        # Verify time coordinate and assign
        if t is not None:
            t = np.asarray(t)
            if t.size != len(self):
                raise ValueError(f"Size mismatch: {np.size(t)} time coordinate(s) for {len(self)} points")
            if t.ndim == 0:
                # Convert to 1D 
                t = t.reshape(-1)
            if t.ndim != 1:
                raise ValueError(f"Expected 1 dimensional time coordinate, received {t.ndim}: {t}")
        
            # Verify time coordinate
            if self._epoch:
                self._time = np.array([verify_td64(ti) for ti in t])
            else:
                dt = np.array([verify_dt64(ti) for ti in t])
                self._epoch = dt[0]
                self._time = dt - dt[0]
        else:
            if self._epoch:
                self._time = np.zeros_like(self.X, dtype='timedelta64[us]')
            else:
                raise ValueError("Time coordinate must be specified, either as a 4th dimension or as a static epoch.")
        
    @property
    def epoch(self) -> datetime:
        return self._epoch.astype(datetime)
    
    @property
    def t(self) -> npt.NDArray[np.float64]:
        """Time coordinate in seconds offset"""
        return self._time.copy().astype(float) * 1E-6
    
    def t_str(self) -> npt.NDArray[np.str_]:
        """Returns string array with t coordinate in '[d day[s], ]hh:mm:ss[.ffffff]' format"""
        return np.array([str(ti.astype(timedelta)) for ti in self._time])
    
    @property
    def dt(self) -> npt.NDArray[np.datetime64]:
        return self._epoch + self._time
    
    @property
    def years(self) -> npt.NDArray[np.float64]:
        dt_us = self.dt.astype('datetime64[us]') # Normalize type to microseconds for GNSS 
        years = self.dt.astype('datetime64[Y]').astype(int)
        
        # Build start/end-of-year as day-based boundaries (to avoid "average year" lengths)
        start_D = years.astype('datetime64[Y]').astype('datetime64[D]')
        end_D   = (years + 1).astype('datetime64[Y]').astype('datetime64[D]')

        # Convert to microseconds for exact duration arithmetic
        start_us = start_D.astype('datetime64[us]')
        end_us   = end_D.astype('datetime64[us]')

        # Elapsed and total year length in integer microseconds
        elapsed = (dt_us - start_us).astype('timedelta64[ns]').astype('int64')
        yearlen = (end_us - start_us).astype('timedelta64[ns]').astype('int64')

        # Fractional year and final decimal year
        years = years + elapsed / yearlen
        years = years.astype('float64')

        return years

    @property
    def coords(self) -> npt.NDArray[np.floating]:
        """ECEF coordinates"""
        if self._ecef is None:
            if self._llh is None:
                raise ValueError("coordinates defined in neither ECEF nor Geodetic system")
            self._ecef = np.vstack(self.frame.geo_to_ecef(*self.igeo())).T
        return self._ecef.copy()
    
    @property
    def geo(self) -> npt.NDArray[np.floating]:
        """Geodetic coordinates"""
        if self._llh is None:
            if self._ecef is None:
                raise ValueError("Coordinates defined in neither ECEF nor Geodetic system")
            self._llh = np.vstack(self.frame.ecef_to_geo(*self)).T
        return self._llh.copy()
    
    def igeo(self) -> Iterator[npt.NDArray[np.floating]]:
        """Iterator over geodetic coordinates"""
        return iter(self.geo.T)
    
    @property
    def proj(self) -> npt.NDArray[np.floating]:
        """Projected map coordinates"""
        if self._map is None:
            self._map = np.vstack(self.frame.proj(lat=self.lat, lon=self.lon)).T
        return self._map.copy()
    
    @property
    def X(self) -> npt.NDArray[np.floating]:
        """ECEF X"""
        return self.coords[:,0]
    
    @property
    def Y(self) -> npt.NDArray[np.floating]:
        """ECEF Y"""
        return self.coords[:,1]
    
    @property
    def Z(self) -> npt.NDArray[np.floating]:
        """ECEF Z"""
        return self.coords[:,2]
    
    @property
    def lon(self) -> npt.NDArray[np.floating]:
        """Longitude"""
        return self.geo[:,0]
    
    @property
    def lat(self) -> npt.NDArray[np.floating]:
        """Latitude"""
        return self.geo[:,1]
    
    @property
    def h(self) -> npt.NDArray[np.floating]:
        """Ellipsoidal height"""
        return self.geo[:,2]
    
    @property
    def easting(self) -> npt.NDArray[np.floating]:
        """Projected easting"""
        return self.proj[:,0]
    
    @property
    def northing(self) -> npt.NDArray[np.floating]:
        """Projected northing"""
        return self.proj[:,1]
    
    @property
    def enu_rotation(self) -> npt.NDArray[np.floating]:
        if self._enu is None:
            self._enu = ecef_to_enu(lon=self.lon, lat=self.lat)
            # Ensure shape (n, 3, 3)
            if self._enu.ndim == 2:
                self._enu = self._enu.reshape(1,3,3)

        return self._enu.copy()

    @property
    def orthometric(self) -> npt.NDArray[np.floating]:
        return self.frame.orthometric(self.lon, self.lat, self.h)

    @property
    def orthometric_llh(self) -> npt.NDArray[np.floating]:
        return np.hstack(self.lon, self.lat, self.orthometric)

    def __len__(self) -> int:
        """Number of points"""
        if self._ecef is not None:
            return self.X.size
        elif self._llh is not None:
            return self.lon.size
        raise ValueError("Coordinates defined in neither ECEF nor Geodetic system")
        
    def copy(self, retain_coords: bool = True, retain_geo: bool = True, retain_proj: bool = True, retain_enu: bool = True) -> Pos:
        if retain_coords and self._ecef is not None:
            cp = Pos(*self, self.t, epoch=self.epoch, frame=self.frame)
            if retain_geo and self._llh is not None:
                cp._llh = self.geo
        elif retain_geo and self._llh is not None:
            cp = Pos(*self.igeo(), self.t, epoch=self.epoch, frame=self.frame, geodetic=True)
        else:
            raise ValueError("Coordinates defined in neither ECEF nor Geodetic system")
        if retain_proj and self._map is not None:
            cp._map = self.proj
        if retain_enu and self._enu is not None:
            cp._enu = self.enu_rotation

        return cp
    
    def __iter__(self) -> Iterator[npt.NDArray[np.floating]]:
        """Iterates over ECEF coordinates"""
        return iter(self.coords.T)
    
    def __eq__(self, other: object) -> bool:
        """If other is serializable as a Pos object: returns True if coordinates match,
        otherwise returns False."""
        try:
            other = Pos(other, epoch=self.t, frame=self.frame)
        except:
            return NotImplemented
        other.reframe(self.frame)
        if self._ecef is not None and other._ecef is not None:
            return self.coords == other.coords
        elif self._llh is not None and other._llh is not None:
            return self.geo == other.geo
        return self.coords == other.coords
    
    def __add__(self, other: object) -> Pos:
        try: 
            other = DeltaPos(other)
        except:
            return NotImplemented
        return self.make(self.coords + (np.transpose(self.enu_rotation, axes=(0,2,1)) @ other.coords[..., None]).squeeze())
        
    def __radd__(self, other: object) -> Pos:
        try: 
            other = DeltaPos(other)
        except:
            return NotImplemented
        return self.make(self.coords + (np.transpose(self.enu_rotation, axes=(0,2,1)) @ other.coords[..., None]).squeeze())
    
    def __sub__(self, other: object) -> Pos|DeltaPos:
        if isinstance(other, DeltaPos):
            return Pos(self.coords.copy() - other.coords.copy(), epoch=self.t.copy(), frame=self.frame)
        try:
            other = self.make(other)
        except:
            return NotImplemented
        return DeltaPos((other.enu_rotation @ (self.coords.copy() - other.coords.copy())[..., None]).squeeze())
        
    def __rsub__(self, other: object) -> DeltaPos:
        try:
            other = self.make(other)
        except:
            return NotImplemented
        return DeltaPos((self.enu_rotation @ (other.coords.copy() - self.coords.copy())[..., None]).squeeze())
    
    def __getitem__(self, idx: IndexType) -> Pos:
        """Returns Pos object with a set of coordinates determined by idx."""
        if self._ecef is not None:
            pos = Pos(*self.coords[idx].T, self.t[idx], epoch=self.epoch, frame=self.frame)
            if self._llh is not None:
                geo = self.geo[idx]
                if geo.ndim == 1:
                    pos._llh = geo.reshape((1, 3))
                elif geo.ndim == 2:
                    pos._llh = geo
        elif self._llh is not None:
            pos = Pos(*self.geo[idx].T, self.t[idx], epoch=self.epoch, frame=self.frame, geodetic=True)
        if self._map is not None:
            map = self.map[idx]
            if map.ndim == 1:
                pos._map = map.reshape((1,3))
            if map.ndim == 2:
                pos._map = map
        if self._enu is not None:
            enu_rotation = self.enu_rotation[idx]
            if enu_rotation.ndim == 1:
                pos._enu = enu_rotation.reshape((1,3,3))
            if enu_rotation.ndim == 2:
                pos._enu = enu_rotation
        return pos

    def __setitem__(self, idx: IndexType, value: Pos|npt.NDArray[np.floating]):
        pos = self.make(value)
        if len(pos) == len(self[idx]):
            if self._ecef is not None:
                self._ecef[idx] = pos.coords
            if self._llh is not None:
                self._llh[idx] = pos.geo
            if self._map is not None:
                self._map[idx] = pos.map
            if self._enu is not None:
                self._enu[idx] = pos.enu_rotation
        else:
            raise ValueError(f"The value must match the idx, and be serializable as a Pos object, not {value}")

    def __bool__(self) -> bool:
        if self._ecef is None and self._llh is None:
            return False
        return True
    
    def __str__(self) -> str:
        if self:
            return str(self.geo)
        return "Empty Position"
    
    def __repr__(self) -> str:
        if self:
            return f"Pos({self.coords}, {f'shape={self.coords.shape}, ' if len(self) > 333 else ''}frame=({self.frame}), epoch={self.epoch})"
        return "Pos(None)"

    def make(self, obj: Pos|npt.ArrayLike[np.floating], geodetic: bool = False, lat_first: bool = False) -> Pos:
        """Returns the object as a Pos object if possible, in the same reference frame."""
        time = False
        if isinstance(obj, Pos):
            return obj.reframe(self.frame) # Skip initiation
        elif isinstance(obj, np.ndarray):
            # Validate dimensions
            if obj.ndim == 1:
                if obj.size != 3 and obj.size != 4:
                    raise ValueError(f"Expected 3 or 4 coordinates, received {obj.size}: {obj}")

                if obj.size == 4:
                    time = True

            elif obj.ndim == 2:
                if obj.shape[1] !=3 and obj.shape[1] !=4:
                    raise ValueError(f"Expected 3 or 4 coordinates, received {obj.shape[1]}: {obj}")
                
                if obj.shape[1] == 4:
                    # Time coordinate
                    time = True

            else:
                raise ValueError("Incorrectly formatted coordinates")
            
        elif len(obj) != 3 and len(obj) != 4:
            raise ValueError(f"Expected 3 or 4 coordinates, received {len(obj)}: {obj}")
        elif len(obj) == 4:
            time = True
        # Initialize
        if time:
            return Pos(obj, frame=self.frame, geodetic=geodetic, lat_first=lat_first)
        else:
            res =  Pos(obj, epoch=self._epoch, frame=self.frame, geodetic=geodetic, lat_first=lat_first)
            return res
    
    def reframe(self, frame: str) -> Pos:
        """Changes the Reference Frame of the coordinates to the one specified"""
        if frame == self.frame:
            return self
        self._ecef = np.vstack(self.frame.as_frame(frame, *self, self.years)).T
        self.frame = ReferenceFrame(frame)
        # Clear other coordinate systems 
        self._llh = None
        self._map = None
        self._enu = None
        return self

    def diff(self, *coordinates: Pos|npt.ArrayLike[np.floating]) -> DeltaPos:
        """Returns the input ECEF coordinates in the local ENU coordinates of this object.
        
        Returns DeltaPos object"""
        if len(coordinates) == 1:
            coordinates = coordinates[0]
        return self.make(coordinates) - self
    
    def add(self, *coordinates: DeltaPos|float|np.ndarray|tuple[float|np.ndarray, ...]) -> Pos:
        """Returns the input ENU coordinates, assumed to be in the local frame of this object,
        in the ECEF coordinates of the matching frame.
        
        Returns Pos object"""
        return self + DeltaPos(*coordinates)

    def mean(self, dtype: type = float, out: None = None) -> Pos:
        """Returns Pos object which is the arithmethic center of current object."""
        return self.make(self.coords.mean(axis=0, dtype=dtype, out=out))
    
    def join(self, other: Pos|npt.ArrayLike[np.floating]) -> None:
        """Serializes the other object as a Pos object and joins it to the current."""
        other = self.make(other)
        if self._ecef is not None:
            self._ecef = np.vstack((self._ecef, other.coords))
        if self._llh is not None:
            self._llh = np.vstack((self._llh, other.geo))
        if self._map is not None:
            self._map = np.vstack((self._llh, other.proj))
        if self._enu is not None:
            self._enu = np.vstack((self._enu, other.enu_rotation))
        time_diff = other._epoch - self._epoch
        self._time = np.hstack((self._time, other._time + time_diff))
    
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

# Projected coordinate zones
def _iutm(lat: np.ndarray|float, lon: np.ndarray|float) -> tuple[np.ndarray|float, np.ndarray|float]:
    """
    Returns the UTM EPSG code for a given latitude and longitude in ITRS.
    Handles Norway and Svalbard special cases.
    """
    if lat is None or lon is None:
        raise ValueError("Determining the UTM Zone requires longitude and latitude to be specified.")
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
    if lat is None or lon is None:
        warn("Determining the UTM Zone requires longitude and latitude to be specified.")
        return (25831, 25832, 2583, 25835, 25837)
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
    if lon is None:
        warn("Determining the DKTM Zone requires longitude to be specified.")
        return (4093, 4094, 4095, 4096)
    if lon < 9.5:
        return 4093  # DKTM1
    elif lon < 10.9:
        return 4094  # DKTM2
    elif lon < 13.5:
        return 4095  # DKTM3
    else:
        return 4096  # DKTM4

# ENU rotation from ECEF
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

