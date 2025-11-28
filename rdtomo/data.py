from __future__ import annotations
import re
from datetime import datetime, timedelta, date as datetype
from pathlib import Path
from contextlib import contextmanager, ExitStack
from typing import Iterator
from dataclasses import dataclass
from typing import KeysView, ValuesView, ItemsView, Any, Iterator
import numpy as np
import shutil
from matplotlib import pyplot as plt
from os import cpu_count

from .config import Settings
from .utils import warn, extract_datetime, drop_into_terminal, local
from .manager import tmp, read_only, DirExistsError, DirNotFoundError
from .gnss import reachz2rnx, fetch_swepos, extract_rnx_info, station_ppp, rtkp, ubx2rnx, splice_sp3, splice_clk, splice_inx, chc2rnx, reach2rnx, generate_mocoref
from .transformers import ecef_to_geo
from .core import TomoScene, TomoScenes, Scenes, tomoinfo
from .apperture import SARModel

#
# Abstract class that dispatches to DataDir, ProcessingDir, TomoDir or TomoArchive
class LoadDir(Path):
    def __new__(cls, *args, data: bool = False, processing: bool = False, tomo: bool = False, **kwargs):

        cached = kwargs.pop("cached", False)
        npar = kwargs.pop("npar", cpu_count())
        generate = kwargs.pop("generate", False) # LoadDir is intended for loading existing directories by default
        date = kwargs.pop("date", None)
        exist_ok = kwargs.pop("exist_ok", False)
        scenes = kwargs.pop("scenes", None)
        # Check if type was forced
        if data:
            if processing or tomo:
                raise ValueError(f"A directory cannot at the same time be a Data Directory{' and a Processing Directory' if processing else ''}{' and a Tomogram Directory' if tomo else ''}")
            return DataDir(*args, **kwargs)
        if processing:
            if tomo:
                raise ValueError("A directory cannot both be a Processing Directory and a Tomogram Directory")
            return ProcessingDir(*args, generate=generate, date=date, **kwargs)
        if tomo:
            return TomoScenes.load(*args, cached=cached, npar=npar, **kwargs)

        # Get path        
        path = Path(*args)

        # Check if path is .tomo dir
        if path.suffix == ".tomo" :
            return TomoDir(*args, cached=cached, npar=npar)
        
        # Check if path contains rawdata folder
        elif (path / "rawdata").is_dir():
            return ProcessingDir(*args, generate=generate, date=date)
        
        # Check if path contains .tomo folder(s)
        elif [d for d in path.glob('*.tomo') if d.is_dir()]:
            return TomoArchive(*args, generate=generate, exist_ok=exist_ok, scenes=scenes)
        
        # Assume Data Directory
        else:
            return DataDir(*args)

    @property
    def parent(self) -> Path:
        return Path(self).parent
    
    def is_relative_to(self, other: Path|str) -> bool:
        return Path(self).is_relative_to(other)
    
    def iterdir(self) -> Iterator[Path]:
        return Path(self).iterdir()
    
    def glob(self, pattern: str, case_sensitive: bool|None = None, recurse_symlinks: bool = False) -> Iterator[Path]:
        return Path(self).glob(pattern=pattern, case_sensitive=case_sensitive, recurse_symlinks=recurse_symlinks)
    
@dataclass(slots=True)
class DroneData():
    container: Path|None = None
    timestamp: datetime|None = None
    base_pos: np.ndarray|None = None
    base_start: datetime|None = None
    base_end: datetime|None = None
    drone_start: datetime|None = None
    drone_end: datetime|None = None
    drone_gnss_bin: Path|None = None
    drone_gnss_log: Path|None = None
    drone_imu_bin: Path|None = None
    drone_imu_log: Path|None = None
    drone_radar_bin: Path|None = None
    drone_radar_log: Path|None = None
    drone_radar_cmd: Path|None = None
    drone_rnx_obs: Path|None = None
    drone_rnx_nav: Path|None = None
    drone_rnx_sbs: Path|None = None
    base_obs: Path|None = None
    base_nav: Path|None = None
    mocoref: Path|None = None
    sp3: Path|None = None
    clk: Path|None = None
    inx: Path|None = None
    
    @property
    def files(self) -> dict[str, Path|None]:
        return {
            "drone_gnss_bin": self.drone_gnss_bin,
            "drone_gnss_log": self.drone_gnss_log,
            "drone_imu_bin": self.drone_imu_bin,
            "drone_imu_log": self.drone_imu_log,
            "drone_radar_bin": self.drone_radar_bin,
            "drone_radar_log": self.drone_radar_log,
            "drone_radar_cmd": self.drone_radar_cmd,
            "drone_rnx_obs": self.drone_rnx_obs,
            "drone_rnx_nav": self.drone_rnx_nav,
            "drone_rnx_sbs": self.drone_rnx_sbs,
            "base_obs": self.base_obs,
            "base_nav": self.base_nav,
            "mocoref": self.mocoref,
            "sp3": self.sp3,
            "clk": self.clk,
            "inx": self.inx
        }
    
    @files.setter
    def files(self, new_files: dict[str, Path|None]) -> None:
        for key, value in new_files.items():
            if key in self.keys() and self._valid_path(value):
                setattr(self, key, value)
            elif key in self.keys():
                raise ValueError(f"The files dict takes only None or Path objects as values. You attempted to assign {value} of type {type(value)}")
            else:
                raise KeyError(f"Invalid key {key}. Valid keys: {self.keys()}")
    
    @property
    def drone_files(self) -> dict[str, Path|None]:
        return {
            "drone_gnss_bin": self.drone_gnss_bin,
            "drone_gnss_log": self.drone_gnss_log,
            "drone_imu_bin": self.drone_imu_bin,
            "drone_imu_log": self.drone_imu_log,
            "drone_radar_bin": self.drone_radar_bin,
            "drone_radar_log": self.drone_radar_log,
            "drone_radar_cmd": self.drone_radar_cmd,
            "drone_rnx_obs": self.drone_rnx_obs,
            "drone_rnx_nav": self.drone_rnx_nav,
            "drone_rnx_sbs": self.drone_rnx_sbs,
        }
    
    def init(self, processing_dir: Path|str, generate: bool = True) -> ProcessingDir:
        processing_dir = ProcessingDir(processing_dir, date=self.timestamp.strftime('%Y%m%d'), generate=generate)

        # Initiate copy
        processing_dir.data = self.copy()
        processing_dir.data.container = None
        for key, file in self.items():
            # Get target directory
            if key == "mocoref":
                target_dir = processing_dir.mocoref_dir
            elif key in self.drone_files:
                target_dir = processing_dir.radar_dir
            else:
                target_dir = processing_dir.ground_dir

            # Copy and update path
            if file == target_dir / file.name:
                continue
            processing_dir.data[key] = shutil.copy2(file, target_dir)
        
        return processing_dir
    
    def overlap(self) -> timedelta:
        return min(self.base_end, self.drone_end) - max(self.base_start, self.drone_start)
    
    def base_epoch(self) -> datetime:
        """Returns a timestamp in the middle of the base_start and base_end
        as a nominal epoch."""
        return self.base_start + (self.base_end - self.base_start)/2
    
    def drone_rnx_files(self) -> tuple[Path|Path|Path]:
        if not all((self.drone_rnx_obs, self.drone_rnx_nav, self.drone_rnx_sbs)):
            self.drone_rnx_obs, self.drone_rnx_nav, self.drone_rnx_sbs = ubx2rnx(self.drone_gnss_bin, obs_file=self.container / self.drone_gnss_bin.with_suffix(".obs").name)
        return (self.drone_rnx_obs, self.drone_rnx_nav, self.drone_rnx_sbs)
            
    def _valid_path(self, path: Any) -> bool:
        return path is None or isinstance(path, Path)

    def keys(self) -> KeysView[str]:
        return self.files.keys()
    
    def paths(self) -> ValuesView[Path|None]:
        return self.files.values()
    
    def items(self) -> ItemsView[str, Path|None]:
        return self.files.items()
    
    def get(self, key: str, default: Path|None = None) -> Path|None:
        return self.files.get(key, default)
    
    def copy(self) -> DroneData:
        return DroneData(
            container=self.container,
            timestamp=self.timestamp,
            base_pos=self.base_pos.copy() if self.base_pos is not None else None,
            base_start=self.base_start,
            base_end=self.base_end,
            drone_start=self.drone_start,
            drone_end=self.drone_end,
            drone_gnss_bin=self.drone_gnss_bin,
            drone_gnss_log=self.drone_gnss_log,
            drone_imu_bin=self.drone_imu_bin,
            drone_imu_log=self.drone_imu_log,
            drone_radar_bin=self.drone_radar_bin,
            drone_radar_log=self.drone_radar_log,
            drone_radar_cmd=self.drone_radar_cmd,
            drone_rnx_obs=self.drone_rnx_obs,
            drone_rnx_nav=self.drone_rnx_nav,
            drone_rnx_sbs=self.drone_rnx_sbs,
            base_obs=self.base_obs,
            base_nav=self.base_nav,
            mocoref=self.mocoref,
            sp3=self.sp3,
            clk=self.clk,
            inx=self.inx,
        )
  
    def __iter__(self) -> Iterator[Path|None]:
        return iter(self.files.values())
    
    def __getitem__(self, key: str) -> Path|None:
        return self.files[key]

    def __setitem__(self, key: str, value: Path|None) -> None:
        if key in self.keys() and self._valid_path(value):
            setattr(self, key, value)
        elif key in self.keys():
            raise ValueError(f"Invalid file type: {value} of type {type(value)}")
        else:
            raise KeyError(f"Invalid key {key}. Valid keys are: {self.keys()}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __contains__(self, key: str|Path) -> bool:
        if key is None:
            raise ValueError
        return (key in self.files and bool(self.files[key])) or (key in self.paths())

class DataDir(LoadDir):
    def __new__(cls, *args) -> DataDir:
        path = Path(*args)
        if path.is_file():
            raise FileExistsError(f"{args[0]} is a file")
        if not path.exists():
            raise ValueError(f"{args[0]} does not exist")
        self = Path.__new__(cls, *args)
        return self

    def __init__(self, *args) -> None:
        super().__init__(*args)

    # Function to scan a data directory for files and extract what's necessary
    @contextmanager
    def open(
            self,
            atx: str|Path|None = None,
            receiver: str|Path|None = None,
            require_drone: bool = False,
            use_swepos: bool = False,
            use_ppp: bool = False,
            use_header: bool = False,
            is_zip: bool = False,
            is_mocoref: bool = False,
            is_csv: bool = False,
            is_llh: bool = False,
            is_json: bool = False,
            is_rnx: bool = False,
            is_hcn: bool = False,
            is_rtcm3: bool = False,
            csv_line: int = 1,
            offset: float = -0.079,
            download_attempts: int = 3,
            max_downloads: int = 10,
            elevation_mask: float|None = None,
            minimal_overlap: timedelta|float = timedelta(minutes=10)
    ) -> Iterator[DroneData]:
        """Searches recursively in the directory to find matching files:
        (1) Drone GNSS .bin and .log, and matching RINEX files;
        (2) Drone IMU .bin and .log;
        (3) Drone Radar .bin, .log and .cfg;
        (4) GNSS base station;
        (5) Mocoref data or precise position of GNSS base station; and
        (6) Data files for PPP and precise mode RTKP post processing.
        
        If the GNSS base station file is missing, can fetch files from the nearest Swepos station,
        and can supplement Mocoref data by performing static PPP on the base station.
        Note that the path must point to a directory which contains exactly one set of drone data.
        For other files, tomosar init will use the first matching file it finds (with an overlap of at least minimal_overlap
        for the base OBS).

        For the GNSS base station a RINEX OBS file is prioritized over other files: HCN files and RTCM3 files are also accepted,
        as well as Reach ZIP archives.

        For mocoref data a mocoref.moco file is prioritized followed by a JSON file, with the underlying assumption that these
        have been generated from raw mocoref data; then a LLH log is prioritized over a CSV file. If a Reach ZIP archive is
        used as the source of the GNSS base station file, the mocoref file will also be generated from there.

        Whereever new files are downloaded or generated, they are contained in a temporary data.tmp directory.
        
        Yields:
        - data: DroneData object containing results"""
    
        def matching_dt(dt1: datetime, dt2: datetime) -> bool:
            """Checks if two datetime objects are within 1 second of eachother"""
            if dt1 == dt2:
                return True
            if dt1 > dt2:
                return dt1 - dt2 == timedelta(seconds=1)
            if dt1 < dt2:
                return dt2 - dt1 == timedelta(seconds=1)

        def from_reachz(archive: Path) -> tuple[tuple[Path, bool], tuple[Path, bool], tuple[float, float, float], datetime, datetime]:
            """Extracts: base_obs, mocoref_file, base_pos, base_start and base_end from a Reach ZIP archive"""
            obs_data, (base_obs, mocoref_file, _) = reachz2rnx(archive, rnx_file=data.drone_rnx_obs, output_dir=data.container)
            if use_header:
                print(f"GNSS base generated from {local(archive, self)}")
            else:
                base_pos = obs_data[mocoref_file]
                print(f"Mocoref data and GNSS base generated from {local(archive, self)}")
            return base_obs, mocoref_file, base_pos

        if use_swepos and use_ppp:
            warn("Swepos files have an exact header position, will not perform PPP.")
            use_ppp = False
            
        # Patterns to look for drone files
        drone_patterns: dict[str, re.Pattern] = {
            "drone_gnss_bin": re.compile(r"^gnss_logger_dat-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.bin$"),
            "drone_gnss_log": re.compile(r"^gnss_logger_log-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.log$"),
            "drone_imu_bin": re.compile(r"^imu_logger_dat-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.bin$"),
            "drone_imu_log": re.compile(r"^imu_logger_log-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.log$"),
            "drone_radar_bin": re.compile(r"^radar_logger_dat-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.bin$"),
            "drone_radar_log": re.compile(r"^radar_logger_log-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.log$"),
            "drone_radar_cmd": re.compile(r"^radar_logger_cmd-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.cfg$"),
        }
        drone_rnx_patterns: dict[str, re.Pattern] = {
            "drone_rnx_obs": re.compile(r"^gnss_logger_dat-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.(obs|OBS)$"),
            "drone_rnx_nav": re.compile(r"^gnss_logger_dat-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.(nav|NAV)$"),
            "drone_rnx_sbs": re.compile(r"^gnss_logger_dat-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.(sbs|SBS)$"),
        }

        # Patterns to look for GNSS base station files
        gnss_patterns: dict[str, re.Pattern] = {
            "RINEX OBS": re.compile(r"^.+\.(\d{2}[Oo]|obs|OBS)$"),
            "HCN": re.compile(r"^.+\.(HCN|hcn)$"),
            "RTCM3": re.compile(r"^.+\.(RTCM3|rtcm3)$"),
            "Reach ZIP archive": re.compile(r"^Reach_\d+\.(zip|ZIP)$"),
        }

        nav_pattern = re.compile(r"^.+\.(\d{2}[Pp]|nav|NAV)")
        
        # Patterns to look for Mocoref data files
        mocoref_patterns: dict[str, re.Pattern] = {
            "mocoref": re.compile(r"^mocoref.moco$"),
            "JSON": re.compile(r"^.+\.(json|JSON)$"),
            "LLH": re.compile(r"^.+\.(llh|LLH)"),
            "CSV": re.compile(r"^.+\.(csv|CSV)$"),
        }

        precise_patterns: dict[str, re.Pattern] = {
            "SP3": re.compile(r"^.+\.(sp3|SP3)$"),
            "CLK": re.compile(r"^.+\.(clk|CLK)$"),
            "INX": re.compile(r"^.+\.(inx|INX)$")
        }
        
        # Dicts to store matches
        drone_files: dict[str, list[tuple[Path, datetime]]] = {key: [] for key in drone_patterns}
        drone_rnx_files: dict[str, list[tuple[Path, datetime]]] = {key: [] for key in drone_rnx_patterns}
        gnss_files: dict[str, list[Path]] = {key: [] for key in gnss_patterns}
        mocoref_files: dict[str, list[Path]] = {key: [] for key in mocoref_patterns}
        precise_files: dict[str, list[Path]] = {key: [] for key in precise_patterns}
        nav_files = []

        print(f"Opening directory: {self} ...")

        # Search recursively
        for p in self.rglob("*"):
            if p.is_file():
                for key, regex in drone_patterns.items():
                    match = regex.match(p.name)
                    if match:
                        dt = extract_datetime(p.name)
                        drone_files[key].append((p, dt))
                        break
                if match:
                    continue
                for key, regex in drone_rnx_patterns.items():
                    match = regex.match(p.name)
                    if match:
                        dt = extract_datetime(p.name)
                        drone_rnx_files[key].append((p, dt))
                        break
                if match:
                    continue
                if not use_swepos:
                    for key, regex in gnss_patterns.items():
                        match = regex.match(p.name)
                        if match:
                            gnss_files[key].append(p)
                            break
                    if match:
                        continue
                    if not use_header and not use_ppp:
                        for key, regex in mocoref_patterns.items():
                            match = regex.match(p.name)
                            if match:
                                mocoref_files[key].append(p)
                                break
                if match:
                    continue
                match = nav_pattern.match(p.name)
                if match:
                    nav_files.append(p)
                    continue
                for key, regex in precise_patterns.items():
                    match = regex.match(p.name)
                    if match:
                        precise_files[key].append(p)
                        break

        # Initiate DroneData storage
        data = DroneData()

        # Ensure that exactly one file is found for each drone type with matching datetimes and extract nominal datetime
        for key, matches in drone_files.items():
            # Ensure that exactly one file is found
            if require_drone and not matches:
                raise FileNotFoundError(f"{key} not found.")
            files, dts = zip(*matches)
            if len(files) > 1:
                raise RuntimeError(f"Multiple {key} files found: {local(files, self)}")
            
            # Ensure datetimes match and extract nominal datetime
            if data.timestamp == None:
                data.timestamp = dts[0]
            elif not matching_dt(data.timestamp, dts[0]):
                raise RuntimeError(f"Timestamps do not match: {data.timestamp} and {dts[0]}")
                
            # Store file
            data[key] = files[0]

        for key, matches in drone_rnx_files.items():
            matched = False
            for file, dt in matches:
                if matched:
                    continue
                # Ensure datetimes match
                if matching_dt(data.timestamp, dt):
                    data[key] = file
                    matched = True

        # Print drone files
        print("Found the following drone files:")
        for key, file in data.drone_files.items():
            if file:
                print(f"{" " * 3}- {key}: {local(file, self)}")
        
        if nav_files:
            data.base_nav = nav_files[0]
            print(f"External NAV data found: {local(data.base_nav, self)}")
        else:
            data.base_nav = None

        settings = Settings()
        with ExitStack() as stack:
            # Generate a temporary directory for file holding
            if Path("data.tmp").exists():
                raise FileExistsError(f"The temporary directory {Path("data.tmp")} already exists, and cannot be used.")
            data.container = stack.enter_context(tmp("data.tmp", allow_dir=True))

            # Work on precise files
            if len(precise_files["SP3"]) == 1:
                data.sp3 = precise_files["SP3"][0]
                print(f"SP3 file located: {data.sp3}")
            else:
                data.sp3 = splice_sp3(precise_files["SP3"], output_dir=data.container)
            if len(precise_files["CLK"]) == 1:
                data.clk = precise_files["CLK"][0]
                print(f"CLK file located: {data.clk}")
            else:
                data.clk = splice_clk(precise_files["CLK"], output_dir=data.container)    
            if len(precise_files["INX"]) == 1:
                data.inx = precise_files["INX"][0]
                print(f"INX file located: {data.inx}")
            else:
                data.inx = splice_inx(precise_files["INX"], output_dir=data.container)
            
            # Ensure drone RNX files exist
            data.drone_rnx_files()
            
            # Extract timestamps
            data.drone_start, data.drone_end, _, _ = extract_rnx_info(data.drone_rnx_obs)

            # Work on base OBS and mocoref.moco
            if use_swepos:
                if is_zip or is_mocoref or is_csv or is_json or is_llh:
                    warn("Fetching Swepos files: other mocoref options ignored")
                use_header = True
                data.base_obs, _ = fetch_swepos(data.drone_rnx_obs, output_dir=data.container)
                data.base_start, data.base_end, header_pos, _ = extract_rnx_info(data.base_obs)
            else:
                mocoref_data_file = None
                if use_ppp:
                    if is_zip or is_mocoref or is_csv or is_json or is_llh:
                        warn("PPP will be used: other mocoref options ignored")
                    mocoref_data = True
                    use_header = False
                # Check if Mocoref data was found
                elif use_header and not use_swepos:
                    if is_mocoref or is_csv or is_json or is_llh:
                        warn("Reading mocoref data from RINEX header: other mocoref options ignored.\nUse only if RINEX header is known to contain precise position.")
                    else:
                        warn("Reading mocoref data from RINEX header.\nUse only if RINEX header is known to contain precise position.")
                    mocoref_data = True
                else:
                    if is_rnx:
                        if is_zip or is_hcn or is_rtcm3:
                            raise ValueError("Only one of is_rnx, is_zip, is_hcn and is_rtcm3 can be used")
                        base_key = "RINEX OBS"
                    elif is_zip:
                        if is_hcn or is_rtcm3:
                            raise ValueError("Only one of is_rnx, is_zip, is_hcn and is_rtcm3 can be used")
                        if is_mocoref or is_csv or is_json or is_llh:
                            raise ValueError("Only one of is_zip, is_mocoref, is_csv, is_json and is_llh can be used")
                        base_key = "Reach ZIP archive"
                    elif is_hcn:
                        if is_rtcm3:
                            raise ValueError("Only one of is_rnx, is_zip, is_hcn and is_rtcm3 can be used")
                        base_key = "HCN"
                    elif is_rtcm3:
                        base_key = "RTCM3"
                    else:
                        base_key = None
                    if is_mocoref:
                        if is_csv or is_json or is_llh:
                            raise ValueError("Only one of is_zip, is_mocoref, is_csv, is_json and is_llh can be used")
                        mocoref_key = "mocoref"
                    elif is_csv:
                        if is_json or is_llh:
                            raise ValueError("Only one of is_zip, is_mocoref, is_csv, is_json and is_llh can be used")
                        mocoref_key = "CSV"
                    elif is_json:
                        if is_llh:
                            raise ValueError("Only one of is_zip, is_mocoref, is_csv, is_json and is_llh can be used")
                        mocoref_key = "JSON"
                    elif is_llh:
                        mocoref_key = "LLH"
                    else:
                        mocoref_key = None
                    mocoref_data = False
                    if not is_zip:
                        for key, files in mocoref_files.items():
                            if (mocoref_key is None or mocoref_key == key) and files:
                                mocoref_data = True
                                mocoref_data_file = files[0]
                                mocoref_key = key

                # Extract matching GNSS base station
                if isinstance(minimal_overlap, float):
                    minimal_overlap = timedelta(minutes=minimal_overlap)
                base_obs_file = False
                header_pos = None
                if mocoref_data:
                    for key, files in gnss_files.items():
                        if (base_key is None or base_key == key) and files:
                            base_obs_file = True
                            if mocoref_data:
                                match key:
                                    case "RINEX OBS":
                                        data.base_obs = files[0]
                                        print(f"Base OBS located: {local(data.base_obs, self)}")
                                    case "HCN":
                                        data.base_obs, _, _ = chc2rnx(files[0], obs_file=data.container / files[0].with_suffix(".obs").name)
                                        print(f"Base OBS generated from {local(files[0], self)}")
                                    case "RTCM3":  
                                        data.base_obs, _, _ = reach2rnx(files[0], obs_file=data.container / files[0].with_suffix(".obs").name, tstart=data.drone_start, tend=data.drone_end)
                                        print(f"GNSS base generated from {local(files[0], self)}")
                                    case "Reach ZIP archive":
                                        # Mocoref data is extracted from the ZIP archive
                                        mocoref_data_file = None 
                                        # Extract
                                        data.base_obs, data.mocoref, data.base_pos = from_reachz(files[0])
                                data.base_start, data.base_end, header_pos, _ = extract_rnx_info(data.base_obs)
                                if data.overlap() > minimal_overlap:
                                    break
                                print(f"Base OBS {'and mocoref ' if data.mocoref else ''} was discarded because of insufficient overlap with drone flight: {data.overlap()}")
                                data.base_obs = None
                                data.mocoref = None
                                data.base_pos = None
                                data.base_start = None
                                data.base_end = None
                                header_pos = None
                    # Generate mocoref from data file
                    if mocoref_data_file:
                        # Verify that base OBS with sufficient overlap was found
                        if not data.base_obs:
                            raise FileNotFoundError(f"Could not find valid base OBS.")
                        # Get mocoref data and generate mocoref.moco file if necessary
                        data.base_pos, data.mocoref = generate_mocoref(mocoref_data_file, timestamp=data.base_epoch(), type=mocoref_key, generate=True, line=csv_line, pco_diff=offset, output_dir=data.container)
                        if mocoref_key == "mocoref":
                            print(f"Mocoref located: {local(mocoref_data_file, self)}")
                        else:
                            print(f"Mocoref data extracted from {mocoref_key} file: {local(mocoref_data_file, self)}")
                else:
                    # Only Reach ZIP archive provides mocoref data
                    if gnss_files["Reach ZIP archive"]:
                        # Extract
                        i = 0
                        header_pos = None
                        while i < len(gnss_files["Reach ZIP archive"]):
                            data.base_obs, data.mocoref, data.base_pos  = from_reachz(gnss_files["Reach ZIP archive"][i])
                            data.base_start, data.base_end, header_pos, _ = extract_rnx_info(data.base_obs)
                            if data.overlap() > minimal_overlap:
                                break
                            print(f"Base OBS {'and mocoref ' if data.mocoref else ''} was discarded because of insufficient overlap with drone flight: {data.overlap()}")
                            i += 1
                            data.base_obs = None
                            data.mocoref = None
                            data.base_pos = None
                            data.base_start = None
                            data.base_end = None
                            header_pos = None
                    elif base_obs_file:
                        raise FileNotFoundError(f"Could not find mocoref data.")
                
                    # Verify that base OBS with sufficient overlap was found
                    if not data.base_obs:
                        raise FileNotFoundError(f"Could not find valid base OBS.")
            
                if use_ppp:
                    results = station_ppp(
                        obs_path=data.base_obs,
                        navglo_path=data.base_nav,
                        atx_path=atx,
                        antrec_path=receiver,
                        sp3_file=data.sp3,
                        clk_file=data.clk,
                        inx_file=data.inx,
                        max_downloads=max_downloads,
                        max_retries=download_attempts,
                        elevation_mask=elevation_mask,
                        out_path=data.container,
                        header=False,
                        retain=True,
                        make_mocoref=True
                    )
                    data.base_pos = results['itrf_position']
                    data.sp3 = results['sp3'] 
                    data.clk = results['clk']
                    data.inx = results['inx']
                    data.mocoref = results['mocoref_file']

            if use_header:
                lon, lat, h = ecef_to_geo(*header_pos, rf=settings.MOCOREF_FRAME) 
                mocoref_dict = {
                    settings.MOCOREF_LATITUDE: lat,
                    settings.MOCOREF_LONGITUDE: lon,
                    settings.MOCOREF_HEIGHT: h,
                    settings.MOCOREF_ANTENNA: 0.
                }
                data.base_pos, data.mocoref = generate_mocoref(mocoref_dict, timestamp=data.base_epoch(), generate=True, output_dir=data.container)
        
        yield data

    def init(
            self,
            processing_dir: str|Path,
            atx: str|Path|None = None,
            receiver: str|Path|None = None,
            use_swepos: bool = False,
            use_ppp: bool = False,
            use_header: bool = False,
            is_zip: bool = False,
            is_mocoref: bool = False,
            is_csv: bool = False,
            is_llh: bool = False,
            is_json: bool = False,
            is_rnx: bool = False,
            is_hcn: bool = False,
            is_rtcm3: bool = False,
            csv_line: int = 1,
            offset: float = -0.079,
            download_attempts: int = 3,
            max_downloads: int = 10,
            elevation_mask: float|None = None,
            minimal_overlap: timedelta|float = timedelta(minutes=10),
            dry: bool = False,
            rtkp_config: str|Path|None = None,
    ) -> ProcessingDir:
        with self.open(
            require_drone=True,
            atx=atx,
            receiver=receiver,
            use_swepos=use_swepos,
            use_ppp=use_ppp,
            use_header=use_header,
            is_zip=is_zip,
            is_mocoref=is_mocoref,
            is_csv=is_csv,
            is_llh=is_llh,
            is_json=is_json,
            is_hcn=is_hcn,
            is_rnx=is_rnx,
            is_rtcm3=is_rtcm3,
            csv_line=csv_line,
            offset=offset,
            download_attempts=download_attempts,
            max_downloads=max_downloads,
            elevation_mask=elevation_mask,
            minimal_overlap=minimal_overlap
        ) as tmp_data:
            if dry:
                print("All files located, setting up temporary processing directory ...", end="\n\n")
                with tmp(tmp_data.container.parent / "processing.tmp") as tmp_dir:
                    processing_dir = tmp_data.init(tmp_dir)
                    processing_dir.init(atx=atx, config=rtkp_config, receiver=receiver, elevation_mask=elevation_mask, minimal_overlap=minimal_overlap, download_attempts=download_attempts, max_downloads=max_downloads)
            else:
                print("All files located, dropping you into the processing directory ... ", end="\n\n")
                drop_into_terminal(processing_dir)
            return tmp_data.init(processing_dir)

    @property
    def content(self) -> list[Path]:
        return [f for f in self.rglob('*') if f.is_file()]
    
    @property
    def info(self) -> list[Path]:
        return self.content
        
class ProcessingDir(LoadDir):
    date: str
    rawdata: Path
    radar_dir: Path
    ground_dir: Path
    mocoref_dir: Path
    para_dir: Path
    m8t_5hz: Path
    config_gps_imu: Path
    process_config: Path
    config_para: Path
    srtm_para: Path
    cband_vv_para: Path
    lband_hh_para: Path
    lband_hv_para: Path
    pband_hh_para: Path
    cband_inf_para: Path
    pband_inf_para: Path
    lband_vv_para: Path
    lband_vh_para: Path
    pband_hv_para: Path
    pband_vv_para: Path
    pband_vh_para: Path
    processing: Path
    cross: Path
    data: DroneData

    # Set of attributes that cannot be changed
    immutable = {"date", "rawdata", "radar_dir", "ground_dir", "mocoref_dir", "para_dir", "m8t_5hz", "config_gps_imu",
                 "process_config", "config_para", "srtm_para", "cband_vv_para", "lband_hh_para", "lband_hv_para",
                 "pband_hh_para", "cband_inf_para", "pband_inf_para", "lband_vv_para", "lband_vh_para", "pband_hv_para",
                 "pband_vv_para", "pband_vh_para", "processing", "cross"}

    def __new__(cls, *args, generate: bool = True, date: str|datetime|datetype|None = None, exist_ok: bool = False):
        # Ensure a valid path was passed
        path = Path(*args)
        if path.is_file():
            raise FileExistsError(f"{args[0]} is a file")
        if generate:
            path.mkdir(exist_ok=True, parents=True)
        if not path.exists():
            raise ValueError(f"{args[0]} does not exist")
        # Initiate
        self = Path.__new__(cls, *args)

        # Initiate rawdata directory
        if not generate or not date:
            if not (path / "rawdata").exists():
                raise DirNotFoundError(f"{path} does not contain a rawdata folder")
            content = [d for d in (path / "rawdata").glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]") if d.is_dir()]
            if len(content) == 1:
                object.__setattr__(self, "rawdata", content[0])
                object.__setattr__(self, "date", self.rawdata.name)
            elif content:
                raise DirNotFoundError(f"Multiple date subfolders of the rawdata directory found: {content}. Specify a date (date=).")
            else:
                raise ValueError(f"No date subfolder of the rawdata diectory found, specify a date (date=){' and set generate=True' if not generate else ''}.")
        else:
            if isinstance(date, (datetime, datetype)):
                date = date.strftime("%Y%m%d")        
            object.__setattr__(self, "date", date)
            object.__setattr__(self, "rawdata", path / "rawdata" / date)
        # Initiate rawdata subfolders
        object.__setattr__(self, "radar_dir", self.rawdata / "radar1")
        object.__setattr__(self, "ground_dir", self.rawdata / "ground1")
        object.__setattr__(self, "mocoref_dir", self.rawdata / "mocoref")
        if generate:
            self.radar_dir.mkdir(exist_ok=exist_ok, parents=True)
            self.ground_dir.mkdir(exist_ok=exist_ok)
            self.mocoref_dir.mkdir(exist_ok=exist_ok)
        
        # Initiate parameter directory
        object.__setattr__(self, "para_dir", path / "parameter")
        object.__setattr__(self, "m8t_5hz", self.para_dir / "config" / "m8t_5hz.conf")
        object.__setattr__(self, "config_gps_imu", self.para_dir / "config" / "config_gps_imu.txt")
        object.__setattr__(self, "process_config", self.para_dir / "config" / "process.config")
        object.__setattr__(self, "config_para", self.para_dir / "config" / "config.para")
        object.__setattr__(self, "srtm_para", self.para_dir / "srtm" / "srtm.para")
        object.__setattr__(self, "cband_vv_para", self.para_dir / "1" / "cband_vv.para")
        object.__setattr__(self, "lband_hh_para", self.para_dir / "2" / "lband_hh.para")
        object.__setattr__(self, "lband_hv_para", self.para_dir / "3" / "lband_hv.para")
        object.__setattr__(self, "pband_hh_para", self.para_dir / "4" / "phand_hh.para")
        object.__setattr__(self, "cband_inf_para", self.para_dir / "5" / "cband_dem.para")
        object.__setattr__(self, "pband_inf_para", self.para_dir / "6" / "pband_inf.para")
        object.__setattr__(self, "lband_vv_para", self.para_dir / "7" / "lband_vv.para")
        object.__setattr__(self, "lband_vh_para", self.para_dir / "8" / "lband_vh.para")
        object.__setattr__(self, "pband_vh_para", self.para_dir / "9" / "pband_vh.para")
        object.__setattr__(self, "pband_hv_para", self.para_dir / "a" / "pband_hv.para")
        object.__setattr__(self, "pband_vv_para", self.para_dir / "b" / "pband_vv.para")

        # Initiate processing directory
        object.__setattr__(self, "processing", path / "processing")
        object.__setattr__(self, "cross", self.processing / "cross")
        # ...

        # Initiate DroneData
        self.data = DroneData()
        return self
    
    def __init__(self, *args, generate: bool = True, date: str|datetime|datetype|None = None, exist_ok: bool = False) -> None:
        super().__init__(*args)

    def __setattr__(self, name: str, value: Any):
        if name in self.immutable:
            raise AttributeError(f"{name} is immutable")
        object.__setattr__(self, name, value)

    def open(self, atx: str|Path|None = None, receiver: str|Path|None = None, minimal_overlap: float|timedelta = timedelta(minutes=10)) -> None:
        with DataDir(self).open(atx=atx, receiver=receiver, require_drone=True, is_rnx=True, is_mocoref=True, minimal_overlap=minimal_overlap) as data:
            for key, file in data.files.items():
                if key == "mocoref":
                    target_dir = self.mocoref_dir
                elif key in data.drone_files:
                    target_dir = self.radar_dir
                else:
                    target_dir = self.ground_dir   
                if file and not file.resolve().parent == target_dir.resolve():
                    raise RuntimeError(f"The file {local(file, self)} was not located inside the correct folder: {local(target_dir, self)}")
            self.data = data

    def init(
            self,
            config: str|Path|None = None,
            atx: str|Path|None = None,
            receiver: str|Path|None = None,
            use_precise: bool = True,
            download_attempts: int = 3,
            max_downloads: int = 10,
            elevation_mask: float|None = None,
            minimal_overlap: timedelta|float = timedelta(minutes=10)
        ) -> None:
        print("Running init")
        self.open(atx=atx, receiver=receiver, minimal_overlap=minimal_overlap)
        
        results = rtkp(
            rover_obs=self.data.drone_rnx_obs,
            base_obs=self.data.base_obs,
            nav_file=self.data.drone_rnx_nav,
            sbs_file=self.data.drone_rnx_sbs,
            sp3_file=self.data.sp3 if use_precise else None,
            clk_file=self.data.clk,
            inx_file=self.data.inx,
            atx_file=atx,
            receiver_file=receiver,
            precise=use_precise,
            out_path=self.data.drone_rnx_obs.with_suffix(".pos"),
            download_dir=self.ground_dir,
            config_file=config,
            elevation_mask=elevation_mask,
            mocoref_file=self.data.mocoref,
            retain=True,
            max_downloads=max_downloads,
            max_retries=download_attempts
        )
        coords, gpst, q = results["coordinates"], results["gpst"], results["quality"]
        fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(8, 8))
        axs = axs.flatten()
        ax = axs[0]
        ax.plot(gpst[q==1], coords[2,q==1], 'g')
        ax.plot(gpst[q!=1], coords[2,q!=1], 'r+')
        ax.set_xlabel("GPST (s)")
        ax.set_ylabel("Ellipsoidal Height (m)")
        ax = axs[1]
        ax.plot(*coords[:,q==1], 'g')
        ax.plot(*coords[:,q!=1], 'r+')
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        fig_name = self.data.timestamp.strftime("%Y-%m-%d-%H-%M-%S-position.svg")
        fig.savefig(self.radar_dir / fig_name, format="svg")

        plt.show()

        # Continue with IMU and unimoco ... 

    @property
    def track_count(self) -> int:
        return len(list(self.cross.iterdir())) if self.cross.is_dir() else None
    
    @property
    def preprocessing_done(self) -> bool:
        True if self.processing.is_dir() else False

    @property
    def info(self) -> dict:
        info = {
            "Preprocessing Done": self.preprocessing_done,
            "Track Count": self.track_count,
        }
        return info
    
class TomoDir(LoadDir):
    _scene: TomoScene|None
    # yyyy-mm-dd-HH-MM-SS-filename_tag.tomo/
    #   |-- flight_info.json
    #   |-- moco_cut.csv
    #   |-- phh
    #   |    |-- processing_parameters.json
    #   |    |-- raw_tomogram.tif
    #   |    |-- multilooked_tomogram.tif
    #   |    |-- filtered_tomogram.tif
    #   |    |-- raw_statistics.csv
    #   |    |-- multilooked_statistics.csv
    #   |    |-- filtered_statistics.csv
    #   |    |-- masked_statistics/
    #   |    |       |-- <mask1>_raw_statistics.csv
    #   |    |       |-- <mask1>_multilooked_statistics.csv
    #   |    |       |-- <mask1>_filtered_statistics.csv
    #   |    |       |-- <mask2>_raw_statistics.csv
    #   |    |       |-- ...
    #   |    |-- cached_masks/
    #   |    |       |-- <mask1>.npy
    #   |    |       |-- <mask1>.json
    #   |    |       |-- <mask2>.npy
    #   |    |       |-- ...
    #   |    |-- .slices/
    #   |    |       |-- ...
    #   |-- cvv
    #   |    |-- ...
    #   |-- lhh
    #   |    |-- ...
    #   |-- ...

    def __new__(cls, *args, generate: bool = True, exist_ok: bool = False, scene: TomoScene|None = None) -> TomoDir:
        path = Path(*args)
        if path.is_file():
            raise FileExistsError(f"{args[0]} is a file")
        if generate:
            path.mkdir(exist_ok=exist_ok)
            read_only(path)
        if not path.exists():
            raise ValueError(f"{args[0]} does not exist")
        if not path.suffix == ".tomo":
            raise ValueError(f"{args[0]} is not a .tomo directory")
        self = Path.__new__(cls, *args)
        if generate:
            object.__setattr__(self, "_scene", scene)
        else:
            object.__setattr__(self, "_scene", None)

        return self
    
    def __init__(self, *args, generate: bool = True, exist_ok: bool = False, scene: TomoScene|None = None) -> None:
        super().__init__(*args)

    @property
    def scene(self) -> TomoScene:
        if not self._scene:
            object.__setattr__(self, "_scene", TomoScene.load(self))
        return self._scene
    
    @property
    def info(self) -> dict:
        if self._scene:
            return self.scene.info
        return tomoinfo(self)

    @property
    def bands(self) -> list[str]:
        return self.scene.bands

    @property
    def model(self) -> SARModel:
        return self.scene.model

    def open(self, cached: bool = True, npar: int = cpu_count()) -> None:
        object.__setattr__(self, "_scene", TomoScene.load(self, cached=cached, npar=npar))

    def update(self) -> None:
        if self._scene:
            self.scene.update()
        else:
            self.load(cached=False)

    def save(self) -> None:
        if not self._scene:
            raise ValueError(f"The TomoDir {self} has not been loaded and cannot be saved")
        self.scene.save(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_scene":
            raise AttributeError("The _scene attribute is immutable")
        object.__setattr__(self, name, value)

class TomoArchive(LoadDir):
    _scenes: TomoScenes|None

    def __new__(cls, *args, generate: bool = True, exist_ok: bool = False, scenes: TomoScenes|None = None) -> TomoDir:
        path = Path(*args)
        if path.is_file():
            raise FileExistsError(f"{args[0]} is a file")
        if generate:
            path.mkdir(exist_ok=exist_ok)
        if not path.exists():
            raise ValueError(f"{args[0]} does not exist")
        if not generate:
            content = [d for d in path.glob('*.tomo') if d.is_dir()]
            if not content:
                raise DirNotFoundError(f"Could not find any .tomo directories inside {path}")
        self = Path.__new__(cls, *args)
        if generate:
            object.__setattr__(self, "_scenes", scenes)
        else:
            object.__setattr__(self, "_scenes", None)

        return self
    
    def __init__(self, *args, generate: bool = True, exist_ok: bool = False, scenes: TomoScenes|None = None) -> None:
        super().__init__(*args)

    @property
    def parents(self) -> list[TomoArchive]:
        return [d for d in super().parents if isinstance(super(d), TomoArchive)]
    
    @property
    def children(self) -> list[TomoArchive]:
        return [d for d in self.rglob('*') if d.is_dir() and isinstance(super(d), TomoArchive)]

    @property
    def scenes(self) -> TomoScenes:
        if not self._scenes:
            object.__setattr__(self, "_scenes", TomoScenes.load(self))
        return self._scenes
    
    @property
    def content(self) -> tuple[TomoDir, ...]:
        return (TomoDir(d) for d in self.glob('*.tomo') if d.is_dir())
    
    @property
    def info(self) -> dict:
        info = {}
        i = 0
        for d in self.content:
            info[d] = d.info
            i += 1
        info["Scene Count"] = i
    
    def add(self, *scenes: Scenes) -> None:
        self.scenes.add(*scenes)

    def list(self) -> None:
        self.scenes.list()

    def open(self, cached: bool = True, npar: int = cpu_count()) -> None:
        object.__setattr__(self, "_scenes", TomoScenes.load(self, cached=cached, npar=npar))

    def update(self, *scenes: Scenes|TomoDir) -> None:
        if self._scenes:
            self.scenes.update()
        else:
            self.load(cached=False)
        for scene_obj in scenes:
            if isinstance(scene_obj, TomoDir):
                if scene_obj in self.content:
                    raise DirExistsError(f"{scene_obj} is already in the archive {self}")
                path = scene_obj
                scene_obj = scene_obj.scene
            else:
                path = None
            self.scenes.add(scene_obj)
            if path:
                shutil.copytree(path, self / path.name)
            else:
                scene_obj.save(self)

    def save(self) -> None:
        if not self._scene:
            raise ValueError(f"The TomoArchive {self} has not been loaded and cannot be saved")
        self.scenes.save(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_scenes":
            raise AttributeError("The _scenes attribute is immutable")
        object.__setattr__(self, name, value)