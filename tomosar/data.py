from __future__ import annotations
import re
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager, ExitStack
from typing import Iterator
from dataclasses import dataclass
from typing import KeysView, ValuesView, ItemsView, Any
import numpy as np
import shutil

from .config import Settings
from .utils import warn, generate_mocoref, extract_datetime, drop_into_terminal
from .binaries import ubx2rnx, tmp, local, splice_sp3, splice_clk, splice_inx, chc2rnx, reach2rnx
from .gnss import reachz2rnx, fetch_swepos, extract_rnx_info, station_ppp
from .transformers import ecef_to_geo

@dataclass
class DataExtraction():
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
    
    def initiate(self, processing_dir: Path|str) -> DataExtraction:
        processing_dir = Path(processing_dir)
        rawdata_dir = processing_dir / "rawdata" / self.timestamp.strftime('%Y%m%d')
        radar_dir = rawdata_dir  / "radar1"
        ground_dir = rawdata_dir / "ground1"
        mocoref_dir = rawdata_dir / "mocoref"

        # Initiate copy
        init = self.copy()
        init.container = None
        for key, file in self.items():
            # Get target directory
            if key == "mocoref":
                target_dir = mocoref_dir
            elif key in self.drone_files:
                target_dir = radar_dir
            else:
                target_dir = ground_dir

            # Copy and update path
            if file == target_dir / file.name:
                continue
            init[key] = shutil.copy2(file, target_dir)
        
        return init
    
    def overlap(self) -> timedelta:
        return min(self.base_end, self.drone_end) - max(self.base_start, self.drone_start)
    
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
    
    def copy(self) -> DataExtraction:
        return DataExtraction(
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

class DataDir(Path):

    def __new__(cls, *args, **kwargs) -> DataDir:
        if not Path(*args).is_dir():
            raise ValueError(f"{args[0]} is not a directory")
        self = super().__new__(cls, *args, **kwargs)
        return self

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
    ) -> Iterator[DataExtraction]:
        """Searches recursively in the directory to find matching files:
        (1) Drone GNSS .bin and .log, and matching RINEX files;
        (2) Drone IMU .bin and .log;
        (3) Drone Radar .bin, .log and .cfg;
        (4) GNSS base station;
        (5) Mocoref data or precise position of GNSS base station; and optionally
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
        - data: DataExtraction object containing results"""
    
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
                print(f"GNSS base generated from {archive}")
            else:
                base_pos = obs_data[mocoref_file]
                print(f"Mocoref data and GNSS base generated from {archive}")
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

        print(f"Opening Data Directory: {self} ...")

        # Search recursively
        for p in self.rglob("*"):
            if p.is_file():
                for key, regex in drone_patterns.items():
                    match = regex.match(p.name)
                    if match:
                        dt = extract_datetime(p.name)
                        drone_files[key].append((p, dt))
                for key, regex in drone_rnx_patterns.items():
                    match = regex.match(p.name)
                    if match:
                        dt = extract_datetime(p.name)
                        drone_files[key].append((p, dt))
                if not use_swepos:
                    for key, regex in gnss_patterns.items():
                        match = regex.match(p.name)
                        if match:
                            gnss_files[key].append(p)
                    if not use_header and not use_ppp:
                        for key, regex in mocoref_patterns.items():
                            match = regex.match(p.name)
                            if match:
                                mocoref_files[key].append(p)
                match = nav_pattern.match(p.name)
                if match:
                    nav_files.append(p)
                for key, regex in precise_patterns.items():
                    match = regex.match(p.name)
                    if match:
                        precise_files[key].append(p)

        # Initiate DataExtraction storage
        data = DataExtraction()

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
                print(f"Located SP3 file: {precise_files['SP3']}")
            else:
                data.sp3 = splice_sp3(precise_files["SP3"], output_dir=data.container)
            if len(precise_files["CLK"]) == 1:
                data.clk = precise_files["CLK"][0]
                print(f"Located CLK file: {precise_files['CLK']}")
            else:
                data.clk = splice_clk(precise_files["CLK"], output_dir=data.container)    
            if len(precise_files["INX"]) == 1:
                data.inx = precise_files["INX"][0]
                print(f"Located INX file: {precise_files['INX']}")
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
                        base_key = "RNX OBS"
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
                                        print(f"Base OBS located: {data.base_obs}")
                                    case "HCN":
                                        data.base_obs, _, _ = chc2rnx(files[0], obs_file=data.container / files[0].with_suffix(".obs").name)
                                        print(f"Base OBS generated from {files[0]}")
                                    case "RTCM3":  
                                        data.base_obs, _, _ = reach2rnx(files[0], obs_file=data.container / files[0].with_suffix(".obs").name, tstart=data.drone_start, tend=data.drone_end)
                                        print(f"GNSS base generated from {files[0]}")
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
                        # Get mocoref data and generate mocoref.moco file if necessary
                        data.base_pos, data.mocoref = generate_mocoref(mocoref_data_file, type=mocoref_key, generate=True, line=csv_line, pco_offset=offset, output_dir=data.container)
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
                    data.base_pos, _, (data.sp3, data.clk, data.inx) = station_ppp(
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
                        retain=True
                    )
                    lon, lat, h = ecef_to_geo.transform(*data.base_pos)
                    mocoref_dict = {
                        settings.MOCOREF_LATITUDE: lat,
                        settings.MOCOREF_LONGITUDE: lon,
                        settings.MOCOREF_HEIGHT: h,
                        settings.MOCOREF_ANTENNA: 0.,
                    }
                    _, data.mocoref = generate_mocoref(mocoref_dict, generate=True, output_dir=data.container)

            if use_header:
                lon, lat, h = ecef_to_geo.transform(*header_pos) 
                mocoref_dict = {
                    settings.MOCOREF_LATITUDE: lat,
                    settings.MOCOREF_LONGITUDE: lon,
                    settings.MOCOREF_HEIGHT: h,
                    settings.MOCOREF_ANTENNA: 0.
                }
                data.base_pos, data.mocoref = generate_mocoref(mocoref_dict, generate=True, output_dir=data.container)
        
        yield data

    def initiate(
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
            minimal_overlap: timedelta|float = timedelta(minutes=10)
    ) -> DataExtraction:
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
            print("All files located, dropping you into the processing directory ... ", end="\n\n")
            drop_into_terminal(processing_dir)
            return tmp_data.initiate(processing_dir)