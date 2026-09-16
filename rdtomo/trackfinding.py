from __future__ import annotations
import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from multiprocessing import Pool
from datetime import datetime, timedelta
from scipy.optimize import minimize
from pathlib import Path
from matplotlib.figure import Figure
from typing import Type, TypeVar, overload, Iterable, Iterator
from abc import ABC, abstractmethod
import re

from .utils import Angles, IndexType, find_inliers, format_duration, add_meta, parse_datetime_string, gpst_to_dt, gpst, srf_reader, ascii_reader, slice_mask, infer_rf
from .position import Pos, DeltaPos, ReferenceFrame
from .dem import elevation as get_elevation
from .config import Frequencies, Settings
FREQUENCIES = Frequencies()

FlightType = TypeVar('FlightType', bound='RawFlight')

class RawFlight:
    __slots__ = ("_pos", "_vel", "_yaw", "_roll", "_pitch")
    _pos: Pos
    _vel: DeltaPos
    _yaw: Angles
    _roll: Angles
    _pitch: Angles

    def __new__(cls: Type[FlightType], *args, **kwargs) -> FlightType:
        return super().__new__(cls)

    @overload
    def __init__(self, pos: Pos, vel: DeltaPos, yaw: Angles, roll: Angles, pitch: Angles) -> RawFlight:
        ...

    @overload
    def __init__(self, flight: RawFlight) -> RawFlight:
        ...

    def __init__(self, arg1: Pos|RawFlight, vel: DeltaPos = None, yaw: Angles = None, roll: Angles = None, pitch: Angles = None) -> RawFlight:
        """Initiates a FlightType instance with already defined
        - position: Pos object
        - velocity: DeltaPos object
        - yaw: Angles object
        - roll: Angles object
        - pitch: Angles object"""

        # Check if first positional argument is RawFlight (subclass)
        if isinstance(arg1, RawFlight):
            pos = arg1.pos
            vel = arg1.vel
            yaw = arg1.yaw
            roll = arg1.roll
            pitch = arg1.pitch
        else:
            # Verify types
            if isinstance(arg1, Pos):
                pos = arg1
            else:
                raise TypeError(f"The first positional argument must be a RawFlight object or a Pos object, not {type(arg1)}")
            if not isinstance(vel, DeltaPos):
                raise TypeError(f"The second positional argument (vel) must be a DeltaPos object, not {type(vel)}")
            if not isinstance(yaw, Angles):
                raise TypeError(f"The third positional argument (yaw) must be Angles object, not {type(yaw)}")
            if not isinstance(roll, Angles):
                raise TypeError(f"The fourth positional argument (roll) must be Angles object, not {type(yaw)}")
            if not isinstance(roll, Angles):
                raise TypeError(f"The fifth positional argument (pitch) must be Angles object, not {type(yaw)}")
            
            # Verify compatible sizes
            if not len(pos) == len(vel):
                raise ValueError(f"Incompatible sizes: position of length {len(pos)} and velocity of length {len(vel)}")
            if not len(pos) == len(yaw):
                raise ValueError(f"Incompatible sizes: position of length {len(pos)} and velocity of length {len(yaw)}")
            if not len(pos) == len(roll):
                raise ValueError(f"Incompatible sizes: position of length {len(pos)} and velocity of length {len(roll)}")
            if not len(pos) == len(pitch):
                raise ValueError(f"Incompatible sizes: position of length {len(pos)} and velocity of length {len(pitch)}")
        
        self._pos = pos.reframe(Settings().TARGET_FRAME)
        self._vel = vel
        self._yaw = yaw
        self._roll = roll
        self._pitch = pitch

    @classmethod
    def from_log(cls: Type[FlightType], data: np.ndarray, reference_date: datetime, reference_frame: str|ReferenceFrame = "ITRF2020") -> FlightType:
        """Initiates a FlightType instance from a unimoco log."""
        # Initiate position
        dt = gpst_to_dt(data[:,0], reference_date=reference_date)
        lat = data[:,1]
        lon = data[:,2]
        alt = data[:,3]
        pos = Pos(lon, lat, alt, dt, frame=reference_frame, geodetic=True)

        # Initiate velocity
        vn = data[:,4]
        ve = data[:,5]
        vu = data[:,6]
        vel = DeltaPos(ve, vn, vu)

        # Initiate yaw, roll, pitch
        roll = Angles(data[:,7], degrees=True)
        pitch = Angles(data[:,8], degrees=True)
        yaw = Angles(data[:,9], degrees=True)

        self = cls(pos, vel, yaw, roll, pitch)

        return self

    @property
    def pos(self) -> Pos:
        return self._pos.copy()
    
    @property
    def vel(self) -> DeltaPos:
        return self._vel.copy()

    @property
    def yaw(self) -> Angles:
        return self._yaw.copy()
    
    @property
    def roll(self) -> Angles:
        return self._roll.copy()
    
    @property
    def pitch(self) -> Angles:
        return self._pitch.copy()

    def __len__(self) -> int:
        return len(self._pos)
    
    def __getitem__(self: FlightType, idx: IndexType) -> FlightType:
        """Returns FlightType object with a set of coordinates determined by idx."""
        cls = type(self)
        return cls(self.pos[idx], self.vel[idx], self.yaw[idx], self.roll[idx], self.pitch[idx])

    def __setitem__(self: FlightType, idx: IndexType, value: FlightType):
        cls = type(self)
        obj = cls(value)
        if len(obj) == len(self[idx]):
            self._pos[idx] = obj.pos
            self._vel[idx] = obj.vel
            self._yaw[idx] = obj.yaw
            self._roll[idx] = obj.roll
            self._pitch[idx] = obj.pitch
        else:
            raise ValueError(f"The value must match the idx, and be serializable as a FlightType object, not {value}")
        
    def astype(self, cls: Type[FlightType], idx: IndexType|None = None) -> FlightType:
        """Returns a RawFlight instance, or optionally a subset thereof, as another FlightType."""
        if idx is None:
            return cls(self.pos, self.vel, self.yaw, self.roll, self.pitch)
        else:
            return cls(self.pos[idx], self.vel[idx], self.yaw[idx], self.roll[idx], self.pitch[idx])
    
    def timestamps(self) -> tuple[str, str]:
        return format_duration(gpst(self.pos.dt[0].astype(datetime))), format_duration(gpst(self.pos.dt[-1].astype(datetime)))

    def copy(self) -> RawFlight:
        return type(self)(self.pos.copy(), self.vel.copy(), self.yaw.copy(), self.roll.copy(), self.pitch.copy())
     
    def save(self, path: str|Path) -> None:
        data = {
            "coords": np.hstack((self.pos.geo, self.pos._time.reshape(-1,1)/np.timedelta64(1, 's'))),
            "epoch": self.pos.epoch,
            "frame": self.pos.frame.name,
            "vel": self.vel.coords,
            "yaw": self.yaw.degs,
            "roll": self.roll.degs,
            "pitch": self.roll.degs,
        }
        np.savez(path, **data)

    @classmethod
    def load(cls: Type[FlightType], path: str|Path) -> FlightType:
        """Loads a FlightType object from a saved .npz file."""
        with np.load(path, allow_pickle=True) as data:
            pos = Pos(data['coords'], epoch=data['epoch'].item(), frame=str(data['frame']), geodetic=True)
            vel = DeltaPos(data['vel'])
            yaw = Angles(data['yaw'], degrees=True)
            roll = Angles(data['roll'], degrees=True)
            pitch = Angles(data['pitch'], degrees=True)
        
        return cls(pos, vel, yaw, roll, pitch)
    
    def dur(self) -> np.timedelta64:
        return (self.pos.dt[-1] - self.pos.dt[0]).astype('timedelta64[s]')
    
    def __str__(self) -> str:
        return f"RawFlight({len(self.pos)} data points)"

class Flight(RawFlight):
    __slots__ = ("_track")
    _track: Track

    @overload
    def __init__(self, pos: Pos, vel: DeltaPos, yaw: Angles, roll: Angles, pitch: Angles) -> Flight:
        ...

    @overload
    def __init__(self, flight: RawFlight) -> Flight:
        ...

    def __init__(self, arg1: Pos|RawFlight, vel: DeltaPos = None, yaw: Angles = None, roll: Angles = None, pitch: Angles = None) -> Flight:
        """Initiates a Flight instance with already defined
        - position: Pos object
        - velocity: DeltaPos object
        - yaw: Angles object
        - roll: Angles object
        - pitch: Angles object"""

        super().__init__(arg1, vel, yaw, roll, pitch)

        self._track = None

    def copy(self) -> Flight:
        cp = Flight(self.pos, self.vel, self.yaw, self.roll, self.pitch)
        if self._track:
            cp.track = self.track
   
    @property
    def track(self) -> Track:
        if self._track is None:
            self._track = Track(self)
        return self._track

    @property
    def type(self) -> str:
        if isinstance(self.track, Spiral):
            return "Spiral"
        if isinstance(self.track, Linear):
            return "Linear"
        if isinstance(self.track, Irregular):
            return "Irregular"

    def plot(self, ax: Axes|None = None, flight_id: str|int|None = None) -> tuple[Axes, Figure|None]:
        """
        Draw this flight's track on the given Axes. Creates Axes if None.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Target axis. If None, a new Figure and Axes are created.
        label : str or None
            Label for the line (defaults to flight ID).
        autoscale : bool
            Whether to recompute limits after plotting.
        margins : float or None
            Fractional padding for autoscale.
        **line_kw : dict
            Additional keyword arguments passed to ax.plot().
        
        Returns
        -------
        ax : matplotlib.axes.Axes
        fig_created : matplotlib.figure.Figure or None
            The Figure if created here, else None.
        """
        fig_created = None
        if ax is None:
            fig_created, ax = plt.subplots()

        # Plot the track
        ax.plot(self.pos.lon, self.pos.lat, 'r', label="Full flight")
        ax.plot(self.track.pos.lon, self.track.pos.lat, 'g', label="Track found")

        # Basic labels
        if flight_id:
            ax.set_title(f"{f'Flight {flight_id}: {self.type}' if isinstance(flight_id, int) else {flight_id}}")
        ax.set_xlabel("lon (deg)")
        ax.set_ylabel("lat (deg)")

        # Trigger redraw in interactive environments
        ax.figure.canvas.draw_idle()

        return ax, fig_created
    
    def __str__(self) -> str:
        return f"Flight({len(self.pos)} data points)"
    
class Track(RawFlight, ABC):
    _initialized: bool

    @overload
    def __new__(cls, pos: Pos, vel: DeltaPos, yaw: Angles, roll: Angles, pitch: Angles) -> Track:
        ...

    @overload
    def __new__(cls, flight: RawFlight) -> Track:
        ...

    def __new__(cls, arg1: Pos|RawFlight = None, vel: DeltaPos = None, yaw: Angles = None, roll: Angles = None, pitch: Angles = None) -> Track:
        if arg1 is None:
            return super().__new__(cls)
        if isinstance(arg1, RawFlight):
            pos = arg1.pos
            vel = arg1.vel
            yaw = arg1.yaw
            roll = arg1.roll
            pitch = arg1.pitch
        else:
            pos = arg1

        if cls is not Track:
            return super().__new__(cls, pos=pos, vel=vel, yaw=yaw, roll=roll, pitch=pitch)
         
        yaw_uw = yaw.unwrap()
        completed_turns = (np.max(yaw_uw) - np.min(yaw_uw)) / (2*np.pi)
        dt = np.gradient(pos.t) # Change in seconds
        time_step = dt.mean()

        # Classify and initiate
        required_turns = 2
        if completed_turns > required_turns: # Preliminary: Spiral
            step = int(10 / time_step)
            window = np.ones(step) / step
            tol = 3e-3  # tolerance for second derivative of yaw

            # Step 1: Smooth unwrapped yaw
            y = np.convolve(yaw_uw, window, mode='full')[:len(yaw_uw)]

            # Step 2: First and second derivatives
            dy = np.gradient(y) / dt 
            dy = np.convolve(dy, window, mode='full')[:len(dy)]
            ddy = np.gradient(dy) / dt

            # Step 3: Find indices with low second derivative
            idx = np.abs(ddy) < tol

            # Step 4: Split into segments
            segments = slice_mask(idx)
            
            if segments:
                min_flight_time = 60 # seconds
                
                # Step 5: Find longest segment
                longest = max(segments, key=lambda s: s.stop - s.start)
                flight_time = (longest.stop - longest.start) * time_step
                if flight_time >= min_flight_time:
                    # Step 6: Extract longest segment as preliminary track
                    ext = 2 * len(window)
                    start = max(0, longest.start - ext)
                    end = longest.stop
                    track = super().__new__(Spiral)
                    track.__init__(pos[start:end], vel[start:end], yaw[start:end], roll[start:end], pitch[start:end])
                    track._initialized = True

                    # Step 7: Gradient of azimuth
                    daz = np.gradient(track.azimuth.unwrap(degrees=True)) / np.gradient(track.pos.t)

                    plt.plot(range(len(daz)), daz, 'r')

                    # Step 8: Find change points in azimuth derivative
                    inliers = find_inliers(daz, min_samples=0.9, relative_threshold=0.4)
#                    plt.plot(inliers, daz[inliers], 'gx')
#                    plt.show()

                    # Step 9: Final track
                    return track[inliers] 

        # Attempt: Linear
        tol_const = 1.1
        min_flight_time = 5  # seconds per track

        # Derivative of heading
        heading = vel.azimuth.unwrap()
        dh = np.gradient(heading) / dt
        idx = np.abs(dh) < tol_const

        # Split into segments
        segments = slice_mask(idx)

        # Filter short segments
        segments = [seg for seg in segments if (seg.stop - seg.start) * time_step >= min_flight_time]

        # Remove first segment, corresponding to the drone flight to mission
        if len(segments) > 2:
            segments = segments[1:]

        # Find longest segment
        lengths = [seg.stop - seg.start for seg in segments]
        if lengths:
            # Initiate Linear tracks
            tracks = [super().__new__(Linear) for _ in segments]
            for track, seg in zip(tracks, segments):
                track.__init__(pos[seg], vel[seg], yaw[seg], roll[seg], pitch[seg])
                track._initialized = True

            parallel_tracks = tracks.pop(np.argmax(lengths))

            for track in tracks:
                if parallel_tracks.is_parallel(track):
                    parallel_tracks.join(track)
            
            return parallel_tracks

        # No Spiral or Linear tracks found
        track = super().__new__(Irregular)
        track.__init__(pos, vel, yaw, roll, pitch)
        track._initialized = True

        return track

    @overload
    def __init__(self, pos: Pos, vel: DeltaPos, yaw: Angles, roll: Angles, pitch: Angles) -> Track:
        ...

    @overload
    def __init__(self, flight: RawFlight) -> Track:
        ...

    def __init__(self, arg1: Pos|RawFlight, vel: DeltaPos = None, yaw: Angles = None, roll: Angles = None, pitch: Angles = None) -> Track:
        """Initiates a Track instance with already defined
        - position: Pos object
        - velocity: DeltaPos object
        - yaw: Angles object
        - roll: Angles object
        - pitch: Angles object"""
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        super().__init__(arg1, vel, yaw, roll, pitch)
        self._initialized = True

    @abstractmethod
    def info(self, elevation: float|None) -> dict:
        """Subclasses must provide a method to get rudimentary information in a dict."""
    
    def __str__(self) -> str:
        return f"Track({len(self.pos)} data points)"
    
class Spiral(Track):
    __slots__ = ("_center", "_dif", "_dem_path", "_model", "_initialized")
    _center: Pos
    _dif: DeltaPos
    _initialized: bool

    def __new__(cls, *args, **kwargs) -> Spiral:
        return super().__new__(cls, *args, **kwargs)

    @overload
    def __init__(self, pos: Pos, vel: DeltaPos, yaw: Angles, roll: Angles, pitch: Angles) -> Spiral:
        ...

    @overload
    def __init__(self, flight: RawFlight) -> Spiral:
        ...

    def __init__(self, arg1: Pos|RawFlight, vel: DeltaPos = None, yaw: Angles = None, roll: Angles = None, pitch: Angles = None) -> Spiral:
        """Initiates a Spiral track instance with already defined
        - position: Pos object
        - velocity: DeltaPos object
        - yaw: Angles object
        - roll: Angles object
        - pitch: Angles object"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        super().__init__(arg1, vel, yaw, roll, pitch)
        self._center = None
        self._dif = None
        self._dem_path = None
        self._model = None

    @property
    def center(self) -> Pos:
        if self._center is None:
            # Initial guess: centroid
            centroid = self.pos.mean()
            centroid_delta = self.pos.diff(centroid)
            
            # Objective function: deviation from linear radius vs angle
            def spiral_error(corr_delta: np.ndarray) -> np.floating:
                new_delta = centroid_delta + corr_delta
                radius = new_delta.norm(horizontal=True)
                angle = new_delta.azimuth.unwrap()
                p = np.polyfit(angle, radius, 1)
                fit_radius = np.polyval(p, angle)
                return np.mean((radius - fit_radius)**2)

            result = minimize(spiral_error, np.asarray([0,0,0]), method='Nelder-Mead')
            self._center = centroid + result.x
        return self._center.copy()
    
    def elevation(self, elevation: float|None = None, dem_path: None|Path|str = None) -> float:
        """Corrects the elevation value of the center point. If elevation is
        specified, this is the value set, otherwise the value is obtained from
        a DEM."""
        if elevation is None:
            elevation, self._dem_path = get_elevation(self.center, dem_path=dem_path)
        self._center = self.center.make([self.center.lon[0], self.center.lat[0], elevation], geodetic=True)

    @property
    def radius(self) -> npt.NDArray[np.float64]:
        if not self._dif:
            self._dif = self.pos - self.center
        return self._dif.norm(horizontal=True)
    
    @property
    def azimuth(self) -> Angles:
        if not self._dif:
            self._dif = self.pos - self.center
        return self._dif.azimuth
    
    @property
    def altitude(self) -> npt.NDArray[np.float64]:
        if not self._dif:
            self._dif = self.pos - self.center
        return self._dif.up

    @property
    def dem_path(self) -> Path|None:
        return self._dem_path

    def info(self, elevation: float|None = None) -> dict:
        """Returns a dict with basic information about the spiral:
        - t_start: timestamp for start of track
        - t_end: timestamp for end of track
        - center_lat: lat coordinate of center
        - center_lon: lon coordinate of center
        - reference_elevation: nominal elevation value of center
        - min_radius: minimum radius (m)
        - max_radius: maximum radius (m)
        - max_altitude: maximum altitude relative reference_elevation (m)
        - min_altitude: minimum altitude relative reference_elevation (m)
        
        If elevation is specified, updates the elevation value of the center."""
        
        if elevation:
            self.elevation(elevation)
        ts = self.timestamps()
        info = {
            "t_start": str(self.pos.dt[0]),
            "t_end": str(self.pos.dt[-1]),
            "center_lat": self.center.lat[0],
            "center_lon": self.center.lon[0],
            "reference_elevation": self.center.h[0],
            "min_radius": round(self.radius.min()),
            "max_radius": round(self.radius.max()),
            "max_altitude": round(self.altitude.max()),
            "min_altitude": round(self.altitude.min()),
        }
        return info

    def copy(self) -> Spiral:
        new_spiral = type(self)(self.pos.copy(), self.vel.copy(), self.yaw.copy(), self.roll.copy(), self.pitch.copy())
        new_spiral._center = self.center
        new_spiral._dif = self._dif.copy()
        new_spiral._dem_path = self._dem_path
        return new_spiral
    
    def save(self, path: str|Path) -> None:
        data = {
            "coords": np.hstack((self.pos.geo, self.pos._time.reshape(-1,1)/np.timedelta64(1, 's'))),
            "epoch": self.pos.epoch,
            "frame": self.pos.frame.name,
            "vel": self.vel.coords,
            "yaw": self.yaw.degs,
            "roll": self.roll.degs,
            "pitch": self.roll.degs,
            "center": self.center.geo,
            "dem": str(self._dem_path.resolve()) if self._dem_path else ""
        }
        np.savez(path, **data)

    @classmethod
    def load(cls: Type[Spiral], path: str|Path) -> Spiral:
        """Loads a Spiral object from a saved .npz file."""
        with np.load(path, allow_pickle=True) as data:
            pos = Pos(data['coords'], epoch=data['epoch'].item(), frame=str(data['frame']), geodetic=True)
            vel = DeltaPos(data['vel'])
            yaw = Angles(data['yaw'], degrees=True)
            roll = Angles(data['roll'], degrees=True)
            pitch = Angles(data['pitch'], degrees=True)
            center = pos.make(data['center'], geodetic=True)
            dem = Path(str(data['dem'])) if data['dem'] else None
        instance = cls(pos, vel, yaw, roll, pitch)
        instance._center = center
        instance._dem_path = dem

        return instance
    
    def __str__(self) -> str:
        return f"SpiralTrack({len(self.pos)} data points)"
    
class Linear(Track):
    __slots__ = ("_tracks", "_heading", "_initialized")
    _tracks: list[int]      # Starting indices of tracks, and final index
    _initialized: bool

    TOL_PAR: float = 0.57   # degrees mean heading is allowed to deviate for tracks to be parallel

    def __new__(cls, *args, **kwargs) -> Linear:
        return super().__new__(cls, *args, **kwargs)

    @overload
    def __init__(self, pos: Pos, vel: DeltaPos, yaw: Angles, roll: Angles, pitch: Angles) -> Linear:
        ...

    @overload
    def __init__(self, flight: RawFlight) -> Linear:
        ...

    def __init__(self, arg1: Pos|RawFlight, vel: DeltaPos = None, yaw: Angles = None, roll: Angles = None, pitch: Angles = None) -> Linear:
        """Initiates a single Linear track instance with already defined
        - position: Pos object
        - velocity: DeltaPos object
        - yaw: Angles object
        - roll: Angles object
        - pitch: Angles object"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        super().__init__(arg1, vel, yaw, roll, pitch)
        self._tracks = [0,len(self)]

    @property
    def heading(self) -> float:
        return [track._vel.azimuth.wrap(degrees=True).mean() for track in self.tracks] 

    @property
    def heading_std(self) -> float:
        return [track._vel.azimuth.wrap(degrees=True).std() for track in self.tracks] 

    @property
    def ntracks(self) -> int:
        return len(self._tracks) - 1

    @property
    def tracks(self) -> list[Linear]:
        return [self[start:end] for start, end in zip(self._tracks[:-1], self._tracks[1:])]

    def join(self, other: Linear) -> None:
        """Joins the other parallel Linear track to the current."""
        if not isinstance(other, Linear):
            raise TypeError(f"Only Linear tracks can be joined together, you attemted to join {other} of type {type(other)}")
        if self.is_parallel(other):
            self._pos.join(other._pos)
            self._vel.join(other._vel)
            self._yaw.join(other._yaw)
            self._roll.join(other._roll)
            self._pitch.join(other._pitch)
            self._tracks = self._tracks + [len(self)]
        else:
            raise ValueError(f"Unable to join parallel tracks with headings {self.heading} and {other.heading}")
    
    @classmethod
    def merge(cls: Linear, sequence: Iterable[Linear]) -> Linear:
        """Merges an iterable object of parallel Linear tracks together into a single object."""
        joined_track = None
        for track in sequence:
            if joined_track is None:
                joined_track = track
                continue
            joined_track.join(track)
        return joined_track
    
    def is_parallel(self, other: Linear|None = None) -> bool:
        """Returns True if all tracks in self are approximately parallel, otherwise False.
        
        If other is also passed, checks if it is also autoparallel, and returns False if not. 
        If both are autoparallel checks if they are approximately parallel and returns True if
        they are otherwise False."""
        headings = self.heading
        diff = abs(headings - headings[0]) % 180
        if np.all((diff < self.TOL_PAR) | (diff > 180 - self.TOL_PAR)):
            if other is None:
                return True
            elif isinstance(other,Linear):
                if other.is_parallel():
                    diff = abs(other.heading - headings[0]) % 180
                    if np.all((diff < self.TOL_PAR) | (diff > 180 - self.TOL_PAR)):
                        return True
                    else:
                        return False
                return False
            else:
                raise TypeError(f"Invalid type of other, expected Linear: {type(other)}")
        else:
            return False

    def info(self, elevation: float = 0.) -> dict:
        """Returns a dict with basic information about the tracks:
        - number_of_tracks: number of separate parallel tracks
        - X: track number with the following nested keys
            - t_start: timestamp for start of track
            - t_end: timestamp for end of track
            - altitude: mean and std of alitude
            - yaw: mean and std of yaw
            - heading: mean and std of heading
        
        The elevation parameter specifies a reference (ellipsoidal) elevation relative which flight altitude is counted.
        If not specified it is set to 0: altitude is equivalent to ellipsoidal height"""
        info = {
            "number_of_tracks": self.ntracks,
        }
        yaw = self.yaw.wrap(degrees=True)
        for i, track in enumerate(self.tracks):
            ts = track.timestamps()
            info[i] = {
                "t_start": ts[0],
                "t_end": ts[1],
                "altitude": {
                    "mean": self.pos.h.mean() - elevation,
                    "std": self.pos.h.std()
                },
                "yaw": {
                    "mean": yaw.mean(),
                    "std": yaw.std()
                },
                "heading": {
                    "mean": self.heading,
                    "std": self.heading_std
                }
            }
        
        return info

    def save(self, path: str|Path) -> None:
        data = {
            "coords": np.hstack((self.pos.geo, self.pos._time.reshape(-1,1)/np.timedelta64(1, 's'))),
            "epoch": self.pos.epoch,
            "frame": self.pos.frame.name,
            "vel": self.vel.coords,
            "yaw": self.yaw.degs,
            "roll": self.roll.degs,
            "pitch": self.roll.degs,
            "tracks": self._tracks,
        }
        np.savez(path, **data)

    @classmethod
    def load(cls: Type[Linear], path: str|Path) -> Linear:
        """Loads a Spiral object from a saved .npz file."""
        with np.load(path, allow_pickle=False) as data:
            pos = Pos(data['coords'], epoch=data['epoch'], frame=data['frame'], geodetic=True)
            vel = DeltaPos(data['vel'])
            yaw = Angles(data['yaw'], degrees=True)
            roll = Angles(data['roll'], degrees=True)
            pitch = Angles(data['pitch'], degrees=True)
            tracks = data['tracks']
        instance = cls(pos, vel, yaw, roll, pitch)
        instance._tracks = tracks

        return instance

    def __str__(self) -> str:
        return f"LinearTrack({len(self.pos)} data points)"

    def __iter__(self) -> Iterator[Linear]:
        return iter(self.tracks)

class Irregular(Track):
    __slots__ = ("_initialized",)

    def __new__(cls, *args, **kwargs) -> Irregular:
        return super().__new__(cls, *args, **kwargs)
    
    @overload
    def __init__(self, pos: Pos, vel: DeltaPos, yaw: Angles, roll: Angles, pitch: Angles) -> Irregular:
        ...

    @overload
    def __init__(self, flight: RawFlight) -> Irregular:
        ...

    def __init__(self, arg1: Pos|RawFlight, vel: DeltaPos = None, yaw: Angles = None, roll: Angles = None, pitch: Angles = None) -> Irregular:
        """Initiates a Linear track instance with already defined
        - position: Pos object
        - velocity: DeltaPos object
        - yaw: Angles object
        - roll: Angles object
        - pitch: Angles object"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        super().__init__(arg1, vel, yaw, roll, pitch)
    
    def info(self, elevation) -> dict:
        """Provides no information at the moment (empty dict)."""
        info = {}
        return info
    
    def __str__(self) -> str:
        return f"IrregularTrack({len(self.pos)} data points)"
    
# Find flights
def find_flights(data: np.ndarray, reference_date: datetime, reference_frame: str|ReferenceFrame = "ITRF2020") -> list[Flight]:
    
    # Parameters
    minimum_flight_alt = 30  # meters
    minimum_flight_dur = timedelta(minutes=1)
    minimum_boot_dur = timedelta(minutes=5)
    tol = 0.1  # tolerance for derivative of altitude

    # Time step and window size
    time_step = data[1,0] - data[0,0]
    step = int(timedelta(seconds=10).total_seconds() / time_step)
    window_size = np.ones(step) / step

    # Filter altitude signal
    alt = np.convolve(data[:,3], window_size, mode='same')
    da = np.diff(alt) / time_step

    # Segments of approximately constant altitude
    idx = np.abs(da) < tol
    segments = slice_mask(idx)

    # Find boot sequence
    durations = [(s.stop - s.start) * time_step for s in segments]
    boot_sequence = next((i for i, dur in enumerate(durations) if dur > minimum_boot_dur.total_seconds()), None)
    if boot_sequence is None:
        return [], time_step, window_size

    ground_alt = data[segments[boot_sequence], 3].mean()

    # Identify flight segments
    idx = data[:,3] > (ground_alt + minimum_flight_alt)

    # Extract flights
    flights = [Flight.from_log(data[s, :], reference_date, reference_frame) for s in slice_mask(idx)]

    # Remove spurious flights
    flights = [flight for flight in flights if flight.dur() > minimum_flight_dur]

    return flights

# Find tracks
def find_tracks(flights: list[Flight], npar: int = os.cpu_count()) -> dict[str, int]:

    with Pool(processes=npar) as pool:
        tracks = pool.map(_get_track, flights)

    counters = {
        "spiral": 0,
        "linear": 0,
        "irregular": 0,
    }
    for flight, track in zip(flights, tracks):
        flight._track = track
        match track:
            case Spiral():
                counters["spiral"] += 1
            case Linear():
                counters["linear"] += 1
            case Irregular():
                counters["irregular"] += 1
    
    return counters

def _get_track(flight: Flight) -> Track:
    return flight.track

# Rudimentary analysis
def analyze_tracks(flights: list[Flight], base_ele: float) -> dict:
    """Returns dict containing information about all tracks."""
    n_spiral = 0
    n_linear = 0
    info = {}
    for i, flight in enumerate(flights, start=1):
        if isinstance(flight.track, Spiral):
            n_spiral += 1
            if 'Spirals' not in info:
                info['Spirals'] = {}
            info['Spirals'][n_spiral] = flight.track.info(elevation=base_ele)
            info['Spirals'][n_spiral]['flight_num'] = i
        if isinstance(flight.track, Linear):
            n_linear += 1
            info[f'Linear_{n_linear}'] = flight.track.info(elevation=base_ele)
            info[f'Linear_{n_linear}']['flight_num'] = i
    
    return info

# Plot results
def plot_tracks(
    flights: list[Flight],
    ncols: int = 3,
    figsize: tuple[int, int] = (12, 8),
    suptitle: str|None = None,
    tight: bool = True,
) -> tuple[Figure, npt.ArrayLike[Axes]]:
    """
    Plot multiple Flight tracks in a grid of subplots.

    Parameters
    ----------
    flights : list[Flight]
        A list of Flight objects.
    ncols : int
        Number of columns in the grid.
    figsize : tuple[float, float]
        Figure size in inches.
    sharex : bool
        Share x-limits across subplots.
    sharey : bool
        Share y-limits across subplots.
    suptitle : str | None
        Optional figure-level title.
    tight : bool
        If True, apply fig.tight_layout() at the end.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray
        2D array of Axes. Unused axes are set invisible.
    """
    n = len(flights)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_visible(False)
        if suptitle:
            fig.suptitle(suptitle)
        if tight:
            fig.tight_layout()
        return fig, np.array([[ax]])

    nrows = (n + ncols - 1) // ncols  # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i, flight in enumerate(flights):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        # Call your method that draws into a specific Axes
        # It should accept ax=... and return (ax, fig_created) or just ax
        flight.plot(ax=ax, flight_id=i+1)

    # Hide any unused axes (e.g., when n is not a multiple of ncols)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle)

    if tight:
        fig.tight_layout()

    return fig, axes

# Modify the radar[...].inf file
def modify_radar_inf(path: Path, tracks: list[Spiral]|Linear, dt: datetime|None = None, dry: bool = False) -> Path:
    """
    Modify radar_logger_dat-[...].inf file with new track timestamps.
    
    Parameters:
        info: dict with start and end times as strings (dd:hh:mm:ss).
        path: Path to the imu_logger_dat-[...].moco file
    """
    t_start = []
    t_end = []
    for t in tracks:
        ts = t.timestamps()
        t_start.append(ts[0])
        t_end.append(ts[1])
    folder = path.parent.parent / "radar1"
    if not dt:
        match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', path.name)
        if match:
            dt = parse_datetime_string(match.group(1), require_datetime=True)
        else:
            raise RuntimeError(f"The file path does not contain a timestamp string, and none was provided: {path}")
    base_name = f"radar_logger_dat-{dt.strftime("%Y-%m-%d-%H-%M-%S")}.inf"
    inf_path = folder / base_name

    # Try ±1 second if file not found
    def try_alternatives() -> Path|None:
        for delta in [1, -1]:
            new_dt = dt + timedelta(seconds=delta)
            alt_name = f"radar_logger_dat-{new_dt.strftime("%Y-%m-%d-%H-%M-%S")}.inf"
            alt_path = folder / alt_name
            if alt_path.exists():
                return alt_path
        return None

    if not inf_path.exists():
        alt_path = try_alternatives()
        if alt_path:
            inf_path = alt_path

    radar_inf = [
        "Flight", "Info", "{CH}", "================", "Number", "of", "tracks", "{i}", ":", str(len(t_start)),
        "time", "begin", "of", "track", "[hh:mm:ss]", "{s}", ":",
        "time", "end", "of", "track", "[hh:mm:ss]", "{s}", ":",
        "time", "shift", "of", "moco", "data", "{f}", ":", "0"
    ]

    if dry:
        print()
        print(" ".join(radar_inf[:3]))
        print(radar_inf[3])
        print()
        print(" ".join(radar_inf[4:8]) + "                    " + " ".join(radar_inf[8:10]))
        print(" ".join(radar_inf[10:16]) + "      " + radar_inf[16], end=" ")
        print(" ".join(t_start))
        print(" ".join(radar_inf[17:19]) + "   " + " ".join(radar_inf[19:23]) + "      " + radar_inf[23], end=" ")
        print(" ".join(t_end))
        print(" ".join(radar_inf[24:-2]) + "             " + " ".join(radar_inf[-2:]))
        print()
        return
    with open(inf_path, "w") as f:
        f.write("\n")
        f.write(" ".join(radar_inf[:3]) + "\n")
        f.write(radar_inf[3] + "\n")
        f.write("\n")
        f.write(" ".join(radar_inf[4:8]) + "                    " + " ".join(radar_inf[8:10]) + "\n")
        f.write(" ".join(radar_inf[10:16]) + "      " + radar_inf[16] + " ")
        f.write(" ".join(t_start) + "\n")
        f.write(" ".join(radar_inf[17:19]) + "   " + " ".join(radar_inf[19:23]) + "      " + radar_inf[23] + " ")
        f.write(" ".join(t_end) + "\n")
        f.write(" ".join(radar_inf[24:-2]) + "             " + " ".join(radar_inf[-2:]))

    return inf_path

# Orchestrating functions
## trackfinder
def trackfinder(
        path: str|Path,
        linear: int = 0,
        dry: bool = False,
        npar: int = os.cpu_count()
) -> list[Spiral]|Linear:
    """Reads a moco file and segments it to find flights and identify their tracks, classified as 
    - Spiral: the track consists of the spiral part of the flight,
    - Linear: the track consists of the parallel linear segments,
    - Irregular: the track consists of the entire flight.

    If the linear parameter is set to 0 (default), the radar_logger_dat-[...].inf file will be modified to contain the
    time stamps of the spiral flights. By setting the value to a positive integer, the time stamps of the linear segments
    corresponding to the track selected will be inserted instead.
    
    Returns a list of Spirals by default, or the Linear segments of the track if the linear parameter is non-zero."""

    path=Path(path)
    print("Running trackfinder ...")

    # Get date and timestamp
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', path.name)
    if match:
        dt = parse_datetime_string(match.group(1), require_datetime=True)
    else:
        raise RuntimeError(f"The file path does not contain a timestamp string: {path}")


    # 1. Get data and reference frame
    print(f"--> Segmenting log: {path}", flush=True)
    try:
        data, rf = srf_reader(path)
    except RuntimeError:
        data, header, _ = ascii_reader(path)
        data = data[:,:10]
        rf, _ = infer_rf(header)

    rf = ReferenceFrame(rf)
    print(f"--> Working in reference frame: {rf}")

    # 2. Find flights
    flights = find_flights(data, dt, rf)
    print(f"--> {len(flights)} flights found ...", end=" ", flush=True)

    # 3. Find tracks
    counters = find_tracks(flights, npar=npar)
    print(f'{counters['spiral']} spiral, {counters['linear']} linear', end="")
    if counters['irregular'] > 0:
        print(f', and {counters['irregular']} irregular.')
    else:
        print(".")

    # 4. Plot tracks
    fig, _ = plot_tracks(flights, suptitle=f"{dt.strftime("%Y-%m-%d")}: tracks")
    if dry:
        plt.show()
   
    # 5. File generation
    else:
        # Save plot of tracks
        fig_path = path.with_name(dt.strftime("%Y-%m-%d-%H-%M-%S-trackfinder.svg"))
        fig.savefig(fig_path)
        print(f"--> Plot of tracks saved to {fig_path}", flush=True)
        
        # Extract relevant tracks
        if linear == 0:
            tracks = [f.track for f in flights if isinstance(f.track, Spiral)]
        else:
            tracks = [f.track for f in flights if isinstance(f.track, Linear)][linear-1]
            

        # Modify radar_inf file
        inf_path = modify_radar_inf(path, tracks, dt=dt, dry=dry)
        print(f"--> Timestamps saved to {inf_path}", flush=True)

        # Save tracks:
        print(f"--> Saving tracks ...", flush=True)
        for n, t in enumerate(tracks):
            file_name = path.with_name(dt.strftime(f"%Y-%m-%d-%H-%M-%S-{n+1:02}-track.npz"))
            print(f"     > {file_name}")
            t.save(file_name)
            
    print("--> All done.")
    return tracks