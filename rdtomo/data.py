from __future__ import annotations
import re
import os
from datetime import datetime, timedelta, date as datetype
from pathlib import Path
from contextlib import contextmanager, ExitStack
from typing import Iterator
from dataclasses import dataclass
from typing import KeysView, ValuesView, ItemsView, Any, Iterator
import shutil
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from os import cpu_count
from abc import ABC
import subprocess
from time import sleep
from multiprocessing import Pool
import json

from .utils import warn, extract_datetime, drop_into_terminal, local, srf_reader, srf_writer, gpst_to_dt, parse_datetime_string, ascii_reader
from .manager import tmp, DirExistsError, DirNotFoundError, gdl, run
from .gnss import reachz2rnx, fetch_swepos, extract_rnx_info, station_ppp, ppk, ubx2rnx, splice_sp3, splice_clk, splice_inx, chc2rnx, reach2rnx, generate_mocoref, read_rnx2rtkp_out
from .core import TomoScene, TomoScenes, Scenes, tomoinfo
from .apperture import SpiralModel
from .position import Pos, ReferenceFrame
from .trackfinding import trackfinder, Spiral
from .config import Settings

# Helper function to populate processing subdirectories
def _proc(band: int, cross_dir: Path):
    gdl(f"proc,{band}", capture=True)
    (cross_dir / f"processing_{band}.done").touch()

# Abstract class that dispatches to DataDir, ProcessingDir, TomoDir or TomoArchive
class LoadDir(Path, ABC):
    _initialized: bool

    def __new__(cls, *args, data: bool = False, processing: bool = False, tomo: bool = False, archive: bool = False, **kwargs):
        generate = kwargs.pop("generate", False) # LoadDir is intended for loading existing directories by default
        date = kwargs.pop("date", None)
        exist_ok = kwargs.pop("exist_ok", False)
        scene = kwargs.pop("scene", None)
        scenes = kwargs.pop("scenes", None)
        # Check if type was forced
        instance = None
        if sum([data, processing, tomo, archive]) > 1:
            raise ValueError("A directory can only be of one type.")
        if data:
            instance = super().__new__(DataDir)
        if processing:
            instance = super().__new__(ProcessingDir)
        if tomo:
            instance = super().__new__(TomoDir)
        if archive:
            instance = super().__new__(TomoArchive)

        # Get path        
        path = Path(*args)
        if path.is_file():
            raise FileExistsError(f"{path} is a file")
        if generate:
            path.mkdir(exist_ok=True, parents=True)
        if not path.exists():
            raise ValueError(f"{path} does not exist")

        if not instance:
            # Check if path is .tomo dir
            if path.suffix == ".tomo" :
                instance = super().__new__(TomoDir)
            
            # Check if path contains rawdata folder
            elif (path / "rawdata").is_dir():
                instance = super().__new__(ProcessingDir)
            
            # Check if path contains .tomo folder(s)
            elif [d for d in path.glob('*.tomo') if d.is_dir()]:
                instance = super().__new__(TomoArchive)
            
            # Assume Data Directory
            else:
                instance = super().__new__(DataDir)
        
        if isinstance(instance, DataDir):
            instance.__init__(*args)
        
        if isinstance(instance, ProcessingDir):
            instance.__init__(*args, generate=generate, date=date, exist_ok=exist_ok)

        if isinstance(instance, TomoDir):
            instance.__init__(*args, generate=generate, exist_ok=exist_ok, scene=scene)

        if isinstance(instance, TomoArchive):
            instance.__init__(*args, generate=generate, exist_ok=exist_ok, scenes=scenes)
        
        instance._initialized = True
        return instance
     
    def __init__(self, *args, **kwargs) -> None:
        if hasattr(self, '_initialized') and self._initialized:
            return
        super().__init__(*args, **kwargs)

    def __truediv__(self, key) -> Path:
        return Path().__truediv__(key)

    @property
    def parent(self) -> Path:
        return Path(self).parent
    
    def is_relative_to(self, other: Path|str) -> bool:
        return Path(self).is_relative_to(other)
    
    def iterdir(self) -> Iterator[Path]:
        return Path(self).iterdir()
    
    def glob(self, pattern: str, case_sensitive: bool|None = None, recurse_symlinks: bool = False) -> Iterator[Path]:
        return Path(self).glob(pattern=pattern, case_sensitive=case_sensitive, recurse_symlinks=recurse_symlinks)
    
    def rglob(self, pattern, *, case_sensitive = None):
        return Path(self).rglob(pattern, case_sensitive=case_sensitive)
    
@dataclass(slots=True)
class DroneData():
    container: Path|None = None
    timestamp: datetime|None = None
    base_pos: Pos|None = None
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
    
    @property
    def nav_files(self) -> list[Path]:
        nav_files = []
        if self.base_nav:
            nav_files.append(self.base_nav)
        
        if self.drone_rnx_files():
            nav_files.append(self.drone_rnx_nav)
        return nav_files

    def init(self, processing_dir: Path|str, exist_ok: bool = True) -> ProcessingDir:
        processing_dir = Path(processing_dir)
        if not processing_dir.exists():
            processing_dir.mkdir(parents=True)

        processing_dir = ProcessingDir(processing_dir, date=self.timestamp.strftime('%Y%m%d'), exist_ok=exist_ok)

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
            if file is None:
                continue
            if file == target_dir / file.name:
                continue
            processing_dir.data[key] = Path(shutil.copy2(file, target_dir))
          
        return processing_dir
    
    def ppk(self,
            config: str|Path|None = None,
            use_precise: bool = True,
            atx: str|Path|None = None,
            receiver: str|Path|None = None,
            elevation_mask: float|None = None,
            minimal_overlap: timedelta|float = timedelta(minutes=10),
            download_attempts: int = 3,
            max_downloads: int = 10,
            raw: bool = False,
            unify_input_frames: bool = True,
        ) -> tuple[Pos|dict]:
        """Performs PPK processing on data, using the internal config file unless an external
        one is provided.
        
        If use_precise is True and no SP3 file is provided then an attempt is made to download
        from ESA (number of parallel downloads specified by max_downloads and each file is
        attempted up to max_retries times). The atx and receiver files can be specified to
        provide alternative ATX lists (receiver is used only for the receiver). The
        elevation_mask parameter overrides the elevation mask configuration parameter from
        the config file if set.
        
        The raw parameter can be set to True in order NOT to use internal rd-tomo resources.
        
        If unify_input_frames is set to False, the BASE position will be input in whatever
        Reference Frame it is provided in, otherwise it will be explicitly reframed to ITRF.
        
        Returns:
        - pos: a Pos object with PPK solution for the drone position
        - result: a dict with the following keys:
            - "SD": Standard deviation of pos solution in ENU as a DeltaPos object,
            - "ratio": AR ratio of each point,
            - "gps_week": GPS week of each point
            - "gpst": GPST (s) of each point, as seconds into current week
            - "quality": Q number
            - "sp3": .SP3 Path or None
            - "clk": .CLK Path or None
            - "path": .pos Path or None"""
        
        # Check if target .pos file exists
        target_file = self.drone_gnss_bin.with_suffix(f".pos")
        if target_file.is_file():
            print(f"Target .pos file located: {target_file}")
            print("--> Will not run PPK")
            pos, results = read_rnx2rtkp_out(target_file)
            results["path"] = target_file
            return pos, results
        
        # Ensure sufficient overlap in data
        if isinstance(minimal_overlap, float):
            minimal_overlap = timedelta(minutes=minimal_overlap)
        if self.overlap() < minimal_overlap:
            raise ValueError(f"Data overlap insufficient: {self.overlap()}")
        
        # Prepare out paths
        if self.container:
            out_path = self.container / self.drone_rnx_obs.with_suffix(".pos").name
            download_dir = self.container
        else:
            out_path = self.drone_rnx_obs.with_suffix(".pos")
            download_dir = self.base_obs.resolve().parent

        # Run PPK
        return ppk(
            rover_obs=self.drone_rnx_obs,
            base_obs=self.base_obs,
            nav_file=self.nav_files,
            out_path=out_path,
            config_file=config,
            sbs_file = self.drone_rnx_sbs,
            sp3_file = self.sp3 if use_precise else None,
            clk_file= self.clk,
            atx_file=atx,
            receiver_file=receiver,
            elevation_mask=elevation_mask,
            precise=use_precise,
            mocoref_pos=self.base_pos,
            mocoref_file=self.mocoref,
            retain=True,
            download_dir=download_dir,
            max_downloads=max_downloads,
            max_retries=download_attempts,
            raw=raw,
            unify_input_frames=unify_input_frames,
        )

    def imuconv(self) -> Path:
        """Returns the imu_logger_dat-[...]_ts+00_il_ie_ad.bin file for unimoco. Runs GDL>imuconv
        on the IMU binary log and GDL>imuie2ad, if necessary, in order to produce it."""

        target_file = self.drone_imu_bin.resolve().parent / (self.drone_imu_bin.stem + "_ts+00_il_ie.bin")
        if not target_file.is_file():
            print("Converting IMU data ...", end=" ", flush=True)
            gdl(["imuconv", self.drone_imu_bin], capture=True)
            print("done.", flush=True)
        else:
            print(f"Found file: {target_file}")
            print("--> Will not convert IMU data.")

        final_file = self.drone_imu_bin.resolve().parent / (self.drone_imu_bin.stem + "_ts+00_il_ie_ad.bin")
        if not final_file.is_file():
            print("Integrating converted IMU data ...", end=" ", flush=True)
            gdl(["imuie2ad", target_file], capture=True)
            print("done.", flush=True)
        else:
            print(f"Found file: {final_file}")
            print("--> Will not integrate converted IMU data.")

        return final_file

    def unimoco(self, pos_file: str|Path, config_file: str|Path, out_path: str|Path) -> Path:
        """Returns the unimoco data. Runs unimoco first, if necessary,
        in order to produce it."""

        out_path = Path(out_path)

        # Target reference frame
        tf = ReferenceFrame(Settings().TARGET_FRAME)

        # Check if out_path exists
        if out_path.is_file():
            print(f"Found file: {out_path}")
            print("--> Will not run unimoco.", flush=True)
            data, rf = srf_reader(out_path)
            if tf != rf:
                print(f"Reference frame inferred from {out_path} ({rf}) does not match target frame ({tf}).")
                print(f"--> Converting ...", end=" ", flush=True)
                rf = ReferenceFrame(rf)

                # Get date and timestamp
                match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', out_path.name)
                if match:
                    dt = parse_datetime_string(match.group(1), require_datetime=True)
                else:
                    raise RuntimeError(f"The file path does not contain a timestamp string: {out_path}")
                
                # Reframe
                dt = gpst_to_dt(data[:,0], reference_date=dt)
                data[:,1:4] = rf.as_frame(tf, *rf.geo_to_ecef(*data[:,1:4].T), dt)
                srf_writer(out_path, data, ref_frame=tf)
        else:
            target_file = self.drone_imu_bin.resolve().parent / (self.drone_imu_bin.stem + "_ts+00_il_ie_ad.moco")
            if not target_file.is_file():
                # Get reference frame from .pos file
                with open(pos_file, 'r') as f:
                    for line in f:
                        if line.startswith("% REF FRAME"):
                            rf = line.split()[-1]
                            break
                        if not line.startswith("%"):
                            # EOH: no information provided, assume standard RTKP output (ITRF2020)
                            rf = "ITRF2020"
                            break
                if tf != rf:
                    print(f"Reference frame inferred from {pos_file} ({rf}) does not match target frame ({tf}).")
                    print(f"--> Converting ...", end=" ", flush=True)
                    read_rnx2rtkp_out(Path(pos_file))
                    print("done.", flush=True)

                # Parameters
                para_file = target_file.with_name("unimoco_parameters.txt")
                # Default values
                q = 1
                mode = 1
                si = 0
                ti = 0
                tf = 0
                if para_file.is_file():
                    with open(para_file, 'r') as f:
                        flag = False
                        for line in f:
                            if not flag:
                                if line.startswith("[unimoco]"):
                                    flag = True
                            if line.startswith("Qscale"):
                                _, _, q = line.partition("=")
                                q = int(q.strip())
                            if line.startswith("mode"):
                                _, _, mode = line.partition("=")
                                mode = int(mode.strip())
                            if line.startswith("saveIntermediate"):
                                _, _, si = line.parition("=")
                                si = int(si.strip())
                            if line.startswith("ti"):
                                _, _, ti = line.partition("=")
                                try:
                                    ti = float(ti.strip())
                                except ValueError:
                                    pass
                            if line.startswith("tf"):
                                _, _, tf = line.partition("=")
                                try:
                                    tf = float(tf.strip())
                                except ValueError:
                                    pass
                    
                print("Running unimoco ...", flush=True)
                run(["unimoco",
                    config_file,
                    self.imuconv(),
                    pos_file,
                    "-ti",
                    str(ti),
                    "-tf",
                    str(tf),
                    "-Qscale",
                    str(q),
                    "-mode",
                    str(mode),
                    "-saveIntermediate",
                    str(si),
                    ">>",
                    self.drone_imu_bin.with_suffix(".tmp")
                ], capture=False)

                # Save radaz-style copy
                shutil.copy2(target_file, self.drone_radar_bin.with_suffix(".moco"))

                # Clean-up
                for f in Path.cwd().glob("temp_*.moco"):
                    f.unlink(missing_ok=True)

                # unimoco parameters
                ti = "default" if ti==0 else ti
                tf = "default" if tf==0 else tf
                with open(para_file, "w") as f:
                    f.write("[unimoco]\n")
                    f.write("# Qscale options: 1e-4 or 1e-3 or 1e-2 or 1e-1 or 1 (default)\n")
                    f.write(f"Qscale={q}\n")
                    f.write("# mode options: 1=fwd/bck (default), 2=fwd, 3,bck\n")
                    f.write(f"mode={mode}\n")
                    f.write("# separate options: 1 -> saveIntermediate\n")
                    f.write(f"saveIntermediate={si}\n")
                    f.write("# initial and final time:\n")
                    f.write(f"ti={ti}\n")
                    f.write(f"tf={tf}\n")

                # Check unimoco quality
                fig, axs = plt.subplots(2, 1, figsize=(12,12))
                ax = axs[0]
                try:
                    res, _, _ = ascii_reader(self.data.drone_imu_bin.resolve().parent / (self.data.drone_imu_bin.stem + "_ts+00_il_ie_ad.res"))
                    ax.plot(res[:,0], res[:,1], label="x", lw=".2")
                    ax.plot(res[:,0], res[:,2], label="y", lw=".2")
                    ax.plot(res[:,0], res[:,3], label="z", lw=".2")
                    ax.set_title("innovation")
                    ax.legend()
                except FileNotFoundError:
                    pass

                ax = axs[1]
                try:
                    sep, _, _ = ascii_reader(self.data.drone_imu_bin.resolve().parent / (self.data.drone_imu_bin.stem + "_ts+00_il_ie_ad.sep"))
                    ax.plot(sep[:,0], sep[:,1], label="North", lw=".2")
                    ax.plot(sep[:,0], sep[:,2], label="East", lw=".2")
                    ax.plot(sep[:,0], sep[:,3], label="Up", lw=".2")
                    ax.set_title("filter separation")
                    ax.legend()
                except FileNotFoundError:
                    pass

                fig.savefig(self.moco_file.with_suffix(".png"))

            else:
                print(f"Found file: {target_file}")
                print("--> Will not run unimoco.", flush=True)

            print("Converting moco file to binary format ...", end=" ", flush=True)
            data, _, _ = ascii_reader(target_file)
            data = data[:,:10]
            srf_writer(out_path, data, ref_frame=tf.name)

            # Radaz-style copy
            shutil.copy2(out_path, self.drone_radar_bin.with_suffix(".mocob"))
            print("done.")

        return data

    def overlap(self) -> timedelta:
        return min(self.base_end, self.drone_end) - max(self.base_start, self.drone_start)
    
    def base_epoch(self) -> datetime:
        """Returns a timestamp in the middle of the base_start and base_end
        as a nominal epoch."""
        return self.base_start + (self.base_end - self.base_start)/2

    def getmac(self) -> str:
        """Returns radar serial number from the radar log."""
        with open(self.drone_radar_log, 'r') as log:
            for line in log:
                if line.startswith("ID"):
                    sn = line.split()[1][-4:]
                    if int(sn[0]) == 0:
                        sn = sn[1:]
                    return sn
        return ''

    def cmdver(self) -> str:
        """Returns the radar cmd version from the radar log."""
        with open(self.drone_radar_log, 'r') as log:
            for line in log:
                if line.startswith("Radar CMD file"):
                    cmdver = line.split()[-1]
                    match = re.search(r'(\d+(?:_\d+)*)', cmdver)
                    if match:
                        return match.group(1)
        return ''

    def drone_rnx_files(self) -> bool:
        if not all((self.drone_rnx_obs, self.drone_rnx_nav, self.drone_rnx_sbs)):
            if not self.drone_gnss_bin:
                return False
            self.drone_rnx_obs, self.drone_rnx_nav, self.drone_rnx_sbs = ubx2rnx(self.drone_gnss_bin, obs_file=self.container / self.drone_gnss_bin.with_suffix(".obs").name)
        return True
            
    def _valid_path(self, path: Any) -> bool:
        return path is None or isinstance(Path(path), Path)

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

    def __bool__(self) -> bool:
        """Returns False if all paths are None, otherwise True."""
        for path in self:
            if path is not None:
                return True
        return False

class DataDir(LoadDir):
    def __new__(cls, *args, **kwargs) -> DataDir:
        return super().__new__(cls, *args, data=True, **kwargs)

    def __init__(self, *args) -> None:
        super().__init__(*args)
        if hasattr(self, '_initialized') and self._initialized:
            pass # Subclass specific

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
        (6) Data files for PPP and precise mode PPK post processing.
        
        If the GNSS base station file is missing, DataDir can fetch files from the nearest Swepos station,
        and can supplement Mocoref data by performing static PPP on the base station.
        Note that the path must point to a directory which contains exactly one set of drone data.
        For other files, DataDir will use the first matching file it finds (with an overlap of at least minimal_overlap
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

        def from_reachz(archive: Path) -> tuple[Path, Path, Path, Pos]:
            """Extracts: base_obs, mocoref_file, base_pos, base_start and base_end from a Reach ZIP archive"""
            obs_data, (base_obs, mocoref_file, base_nav) = reachz2rnx(archive, rnx_file=data.drone_rnx_obs, output_dir=data.container)
            base_pos = obs_data[mocoref_file]
            if use_header:
                print(f"Base OBS and NAV generated from {local(archive, self)}")
            else:
                print(f"Mocoref data and base OBS and NAV generated from {local(archive, self)}")
            return base_obs, base_nav, mocoref_file, base_pos

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

        nav_pattern = re.compile(r"^.+\.(\d{2}[Pp]|nav|NAV)$")
        
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
            if not data.drone_rnx_files():
                raise FileNotFoundError(f"GNSS drone data missing: {data.drone_gnss_bin}")
            
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
                            match key:
                                case "RINEX OBS":
                                    data.base_obs = files[0]
                                    print(f"Base OBS located: {local(data.base_obs, self)}")
                                case "HCN":
                                    data.base_obs, data.base_nav, _ = chc2rnx(files[0], obs_file=data.container / files[0].with_suffix(".obs").name, nav=True)
                                    print(f"Base OBS and NAV generated from {local(files[0], self)}")
                                case "RTCM3":  
                                    data.base_obs, data.base_nav, _ = reach2rnx(files[0], obs_file=data.container / files[0].with_suffix(".obs").name, tstart=data.drone_start, tend=data.drone_end, nav=True)
                                    print(f"Base OBS and NAV generated from {local(files[0], self)}")
                                case "Reach ZIP archive":
                                    # Mocoref data is extracted from the ZIP archive
                                    mocoref_data_file = None 
                                    # Extract
                                    data.base_obs, data.base_nav, data.mocoref, data.base_pos = from_reachz(files[0])
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
                            data.mocoref = mocoref_data_file
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
                            data.base_obs, data.base_nav, data.mocoref, data.base_pos  = from_reachz(gnss_files["Reach ZIP archive"][i])
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
                    data.base_pos, results = station_ppp(
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
                    data.sp3 = results['sp3'] 
                    data.clk = results['clk']
                    data.inx = results['inx']
                    data.mocoref = results['mocoref_file']

            if use_header:
                data.base_pos, data.mocoref = generate_mocoref(header_pos, generate=True, output_dir=data.container)

            print()
            yield data

        data.container = None

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
            ppk_config: str|Path|None = None,
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
                    processing_dir.init(atx=atx, config=ppk_config, receiver=receiver, elevation_mask=elevation_mask, minimal_overlap=minimal_overlap, download_attempts=download_attempts, max_downloads=max_downloads)
            else:
                print("All files located, dropping you into the processing directory ... ", end="\n\n")
                drop_into_terminal(processing_dir)
            return tmp_data.init(processing_dir, exist_ok=False)

    @property
    def content(self) -> list[Path]:
        return [f for f in self.rglob('*') if f.is_file()]
    
    @property
    def info(self) -> list[str]:
        return [str(p) for p in self.content]

    @property
    def name(self) -> str:
        return "Data Directory"

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
    _moco_file: Path|None

    # Set of attributes that cannot be changed
    immutable = {"date", "rawdata", "radar_dir", "ground_dir", "mocoref_dir", "para_dir", "m8t_5hz", "config_gps_imu",
                 "process_config", "config_para", "srtm_para", "cband_vv_para", "lband_hh_para", "lband_hv_para",
                 "pband_hh_para", "cband_inf_para", "pband_inf_para", "lband_vv_para", "lband_vh_para", "pband_hv_para",
                 "pband_vv_para", "pband_vh_para", "processing", "cross", "_moco_file"}
    
    def __new__(cls, *args, **kwargs) -> ProcessingDir:
        return super().__new__(cls, *args, processing=True, **kwargs)
    
    def __init__(self, *args, date: str|datetime|datetype|None = None, exist_ok: bool = True, **kwargs) -> None:
        super().__init__(*args)
        path = Path(*args)

        # Initiate rawdata directory
        if hasattr(self, '_initialized') and self._initialized:
            # Convert provided date to string
            if isinstance(date, (datetime, datetype)):
                date = date.strftime("%Y%m%d")
            
            # Check if rawdata folder exists and contains date folder
            if (path / "rawdata").exists():
                content = [d for d in (self / "rawdata").glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]") if d.is_dir()]
                if len(content) == 1:
                    object.__setattr__(self, "rawdata", content[0])
                    object.__setattr__(self, "date", self.rawdata.name)
                    if date and date != self.date:
                        warn(f"Provided date {date} does not match content of rawdata directory {self.rawdata}. Ignoring provided date.")
                elif content:
                    raise DirNotFoundError(f"Multiple date subfolders of the rawdata directory found: {content}. Only one is allowed.")
                elif date:
                    object.__setattr__(self, "date", date)
                    object.__setattr__(self, "rawdata", path / "rawdata" / date)
                    self.rawdata.mkdir()
                else:
                    raise ValueError(f"No date subfolder of the rawdata diectory found, specify a date (date=).")
            elif date:
                object.__setattr__(self, "date", date)
                object.__setattr__(self, "rawdata", path / "rawdata" / date)
                self.rawdata.mkdir(parents=True)
            else:
                raise ValueError(f"No date specified (date=), and date directory does not exist.")
                
            # Initiate rawdata subfolders
            object.__setattr__(self, "radar_dir", self.rawdata / "radar1")
            object.__setattr__(self, "ground_dir", self.rawdata / "ground1")
            object.__setattr__(self, "mocoref_dir", self.rawdata / "mocoref")
            self.radar_dir.mkdir(exist_ok=True)
            self.ground_dir.mkdir(exist_ok=True)
            self.mocoref_dir.mkdir(exist_ok=True)
            object.__setattr__(self,"_moco_file", None)
            
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
            object.__setattr__(self, "processing", self / "processing")
            object.__setattr__(self, "cross", self.processing / "cross")
            # ...

            # Initiate DroneData
            self.data = DroneData()

    def __setattr__(self, name: str, value: Any):
        if name in self.immutable:
            raise AttributeError(f"{name} is immutable")
        object.__setattr__(self, name, value)

    def open(self) -> None:
        with DataDir(self).open(require_drone=True, is_rnx=True, is_mocoref=True) as data:
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
        
    def gather(self) -> None:
        """Gathers data and fetches Radaz parameter files."""
        
        # Gather data
        if not self.data:
            self.open()

        # Check if parameters file already exist
        if (self / "parameter" / '1' / 'cband_vv.para').is_file():
            return
        if (self / "parameter" / '1' / 'cband_hh.para').is_file():
            return

        # Locate template
        paraname = '*_' + self.data.getmac() + '_' + self.data.cmdver()
        parameter_folders = [p for p in (Settings().RADAZ_CONFIG / "parameterfiles").glob(paraname)]
        if len(parameter_folders) == 1:
            shutil.copytree(parameter_folders[0], self, dirs_exist_ok=True)
        elif len(parameter_folders) > 1:
            raise DirExistsError(f"Multiple ({len(parameter_folders)}) parameter folders found: {parameter_folders}")
        else:
            raise DirNotFoundError(f"No parameter folder found matching: {paraname}")   

    @property
    def moco_file(self) -> Path:
        if self._moco_file is None:
            self.gather()
            (self.rawdata / "moco").mkdir(exist_ok=True)
            object.__setattr__(self,"_moco_file", self.rawdata / "moco" / (self.data.drone_imu_bin.stem + "_ts+00_il_ie_ad.moco"))
        return self._moco_file
            
    def init(
            self,
            config: str|Path|None = None,
            atx: str|Path|None = None,
            receiver: str|Path|None = None,
            use_precise: bool = True,
            download_attempts: int = 3,
            max_downloads: int = 10,
            elevation_mask: float|None = None,
            minimal_overlap: timedelta|float = timedelta(minutes=10),
            linear: int = 0
        ) -> None:

        # Gather data and ensure parameter files are present
        self.gather()

        if self.moco_file.is_file():
            print(f"Found file: {self.moco_file}")
            print("--> Will not perform direct georeferencing of drone.")
        else:
            # Run PPK
            coords, results = self.data.ppk(
                config=config,
                use_precise=use_precise,
                atx=atx,
                receiver=receiver,
                elevation_mask=elevation_mask,
                max_downloads=max_downloads,
                download_attempts=download_attempts,
                minimal_overlap=minimal_overlap,
            )
            gpst, q = results["gpst"], results["quality"]
            
            # Plot flights
            fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(8, 8))
            axs = axs.flatten()
            ax = axs[0]
            ax.plot(gpst[q==1], coords.h[q==1], 'g')
            ax.plot(gpst[q!=1], coords.h[q!=1], 'r+')
            ax.set_xlabel("GPST (s)")
            ax.set_ylabel("Ellipsoidal Height (m)")
            ax.set_title(coords.frame.name)
            ax = axs[1]
            ax.plot(coords.easting[q==1], coords.northing[q==1], 'g')
            ax.plot(coords.easting[q!=1], coords.northing[q!=1], 'r+')
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            fig_name = self.data.timestamp.strftime("%Y-%m-%d-%H-%M-%S-gnss-position.png")
            fig.savefig(self.radar_dir / fig_name, format="png")

            # Run unimoco on position output
            self.data.unimoco(results['path'], self.config_gps_imu, self.moco_file)

            # Finishing touches using GDL procedures
            (self / "mocos.done").touch()

        # Trackfinding
        track_files = [f for f in self.moco_file.parent.glob("*track.npz")]
        if track_files:
            print(f"Found {len(track_files)} track files (*track.npz).")
            print("--> Will not run trackfinder")
        else:
            trackfinder(self.moco_file, linear=linear)

        if linear == 0:
            print("Setting things up ...", end=" ", flush=True)
            gdl("construct,/para",capture=True)
            sleep(0.2)
            args = [(band, self.cross) for band in Settings().RADAR_BANDS if not (self.cross / f"processing_{band}.done").exists()]
            if len(args) > 1:
                with Pool(processes=min(os.cpu_count(), len(args))) as pool:
                    pool.starmap(_proc, args)
            elif len(args) == 1:
                _proc(args[0])
            print("done.")

            for track in self.track_dirs:
                (track / "track.npz").unlink(missing_ok=True)
                (track / "track.npz").symlink_to(self.rawdata.resolve() / "moco" / self.dt.strftime(f"%Y-%m-%d-%H-%M-%S-{track.name}-track.npz"))
                self.dem(track.name).unlink(missing_ok=True)
                dem = self._set_dem(track.name)
                if dem is None:
                    warn(f"--> No valid DEM found for track {track.name}, will not inspect.")
                else:
                    self.dem(track.name).symlink_to(dem.resolve())
                    print(f"--> Inspecting track {track.name} ...", end=" ", flush=True)
                    fig, data = self.inspect(track=track.name)
                    print("done.")
                
                    fig.savefig(track / "sar_parameters.svg")
                    print(f"        > {track / "sar_parameters.svg"}")
                    with open(track / "meta_data.json", "w") as f:
                        json.dump(data, f, indent=4)
                    print(f"        > {track / "metadata.json"}")
            (self / "tracks").unlink(missing_ok=True)
            (self / "tracks").symlink_to(self.cross)
        
            print("Starting taskmon ...", end=" ", flush=True)
            subprocess.run(
                ["bash",shutil.which('taskmon'), "&"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            print("done.")
            print("--> Taskmon will now manage processes to populate processing subfolders.")

            print("\nAll done.")
        else:
            print("Starting Radaz linear processor ...", flush=True)
            gdl("proz")

    def dem(self, track: int|str) -> Path:
        return self.track_dir(track) / "DEM"

    def _set_dem(self, track: int|str, dem_path: str|Path|None = None) -> Path|None:
        t = self.track(track)
        t.elevation(dem_path=dem_path)
        t.save(self.track_file(track))
        return t.dem_path

    def get_dem(self, track: int|str, dem_path: str|Path|None = None) -> Path:
        """Sets DEM for track to dem_path if passed. If not passed, sets DEM for track
        from settings if not already set. Returns DEM path."""
        if isinstance(track, str):
            track = int(track)
        path = self.dem(track)
        if dem_path:
            dem_path = Path(dem_path)
            if dem_path.is_file():
                if path.resolve() == dem_path:
                    return dem_path
                else:
                    self._set_dem(track, dem_path=dem_path)
                    path.unlink()
                    path.symlink_to(dem_path)
                    return dem_path
            else:
                raise RuntimeError(f"Target file does not exist: {dem_path}")
        elif path.is_symlink():
            return path.resolve()
        dem_path = self._set_dem(track)
        if dem_path is None:
            raise RuntimeError("No valid DEM found")
        path.symlink_to(dem_path)
        return dem_path

    def inspect(self, track: int|str, dem_path: str|Path|None = None) -> tuple[Figure, dict]:
        if self.track_file(track) is None:
            raise RuntimeError(f"Could not locate track file")
        self.get_dem(track, dem_path=dem_path)
        return self.model(track).evaluate()
         
    @property
    def dt(self) -> datetime:
        # Get date and timestamp
        self.gather()
        match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', self.data.drone_imu_bin.name)
        if match:
            return parse_datetime_string(match.group(1), require_datetime=True)
        else:
            raise RuntimeError("Could not parse datetime")

    @property
    def dt_string(self) -> str:
        return self.dt.strftime("%Y-%m-%d-%H-%M-%S")
        
    @property
    def track_count(self) -> int|None:
        return len(self.track_dirs) if self.cross.is_dir() else None

    @property
    def track_list(self) -> list[int]|None:
        if self.track_count:
            return list(range(1,self.track_count+1))
        return None

    @property
    def track_dirs(self) -> list[Path]|None:
        return [p for p in self.cross.iterdir() if p.is_dir()] if self.cross.is_dir() else None

    def track_dir(self, track: int|str) -> Path:
        if isinstance(track, str):
            track = int(track)
        return self.cross / f"{track:02d}"
    
    def track_file(self, track: int|str) -> Path:
        if isinstance(track, str):
            track = int(track)
        file = (self.track_dir(track) / "track.npz").resolve()
        if file.is_file():
            return file
        else:
            return None

    def track(self, track: int|str) -> Spiral:
        return Spiral.load(self.track_file(track))

    def model(self, track: int|str) -> SpiralModel:
        return SpiralModel.load(self.track_file(track))
    
    @property
    def band_count(self) -> int|None:
        first_track = self.cross / "01"
        return len([p for p in first_track.iterdir() if p.is_dir()]) if first_track.is_dir() else None

    @property
    def band_list(self) -> list[int]|None:
        if self.band_count:
            return list(range(1,self.band_count+1))
        return None

    def band(self, track: int, band: int) -> Path:
        return self.cross / f"{track:02d}" / hex(band)

    def bands(self, only_processing: bool = False) -> list[Path]|None:
        if only_processing:
            processing_bands = Settings().RADAR_BANDS
            return [p for p in self.cross.glob('*/*') if p.is_dir() and int(p.name, 16) in processing_bands] if self.cross.is_dir() else None
        return [p for p in self.cross.glob('*/*') if p.is_dir()] if self.cross.is_dir() else None

    def band_info(self, band: int|str) -> str:
        map = {
            1: "CVV",
            2: "LHH",
            3: "LHV",
            4: "PHH",
            5: "CVV(i)",
            6: "PHH(i)",
            7: "LVV",
            8: "LHH",
            9: "PHV",
            10: "PVV",
            11: "PVH"
        }

        if isinstance(band, str):
            band = int(band, 16)

        return map[band]
        
    def metadata(self, track: int|str) -> dict:
        try:
            with open(self.track_dir(track) / "metadata.json", 'r') as f:
                return json.load(f)
        except:
            return {"info": "Missing."}
    
    @property
    def preprocessing_done(self) -> bool:
        True if self.processing.is_dir() else False

    @property
    def info(self) -> dict:
        track_dirs = self.track_dirs
        info = {}
        if track_dirs is None:
            return
        else:
            for track_dir in track_dirs:
                info[track_dir.name] = self.metadata(track_dir.name)["info"]

        return info

    @property
    def name(self) -> str:
        return "Processing Directory"
    
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

    def __new__(cls, *args, **kwargs) -> TomoDir:
        return super().__new__(cls, *args, tomo=True, **kwargs)

    def __init__(self, *args, generate: bool = True, exist_ok: bool = False, scene: TomoScene|None = None) -> None:
        super().__init__(*args)
        if hasattr(self, '_initialized') and self._initialized:
            if not self.suffix == ".tomo":
                raise ValueError(f"{self} is not a .tomo directory")

            if generate:
                object.__setattr__(self, "_scene", scene)
            else:
                object.__setattr__(self, "_scene", None)

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
    def model(self) -> SpiralModel:
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

    @property
    def name(self) -> str:
        return "Tomogram Directory"
    
class TomoArchive(LoadDir):
    _scenes: TomoScenes|None

    def __new__(cls, *args, **kwargs) -> TomoArchive:
        return super().__new__(cls, *args, archive=True, **kwargs)

    def __init__(self, *args, generate: bool = True, exist_ok: bool = False, scenes: TomoScenes|None = None) -> TomoDir:
        super().__init__(*args)
        if hasattr(self, '_initialized') and self._initialized:
            if not generate:
                content = [d for d in self.glob('*.tomo') if d.is_dir()]
                if not content:
                    raise DirNotFoundError(f"Could not find any .tomo directories inside {self}")
            
            if generate:
                object.__setattr__(self, "_scenes", scenes)
            else:
                object.__setattr__(self, "_scenes", None)
    
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

    @property
    def name(self) -> str:
        return "Tomogram Archive"