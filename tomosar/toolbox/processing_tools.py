import click
from pathlib import Path
import os
import time as Time
from datetime import datetime, timedelta, date
import re
import shutil
from matplotlib import pyplot as plt

from ..gnss import fetch_swepos as run_fetch_swepos, station_ppp as run_station_ppp, rtkp, extract_rnx_info, reachz2rnx
from ..trackfinding import trackfinder as run_trackfinder
from .. import ImageInfo, TomoScenes, Settings
from ..utils import interactive_console, extract_datetime, local, warn, drop_into_terminal, generate_mocoref
from ..forging import tomoforge
from ..binaries import chc2rnx, reach2rnx, ubx2rnx, tmp, merge_eph
from ..transformers import ecef_to_geo

@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path.cwd())
@click.option("-f", "--force", is_flag=True, help="Force generation of processing directory (this may overwrite existing directories, but has no effect if PATH is a processing directory)")
@click.option("--swepos", is_flag=True, help="Substitute for GNSS base station with files from nearest Swepos station")
@click.option("--ppp", is_flag=True, help="Subsitute for mocoref data by running static PPP on GNSS base station")
@click.option("-z", "-zip", "is_zip", is_flag=True, help="Force GNSS base station and mocoref.moco files to be generated from a Reach ZIP archive")
@click.option("--mocoref", "is_mocoref", is_flag=True, help="Force mocoref data to be read from mocoref.moco file")
@click.option("--csv", "is_csv", is_flag=True, help="Force mocoref data to be read from CSV file")
@click.option("--json", "is_json", is_flag=True, help="Force mocoref data to be read from JSON file")
@click.option("--llh", "is_llh", is_flag=True, help="Force mocoref data to be read from LLH file")
@click.option("-h", "--header", is_flag=True, help="Read mocoref data from RINEX header (no separate file, use ONLY if RINEX header is known to contain precise position)")
@click.option("-p", "--processing", "is_processing_dir", is_flag=True, help="Force the specified PATH to be interpreted as a processing directory")
@click.option("--precise", "use_precise", is_flag=True, help="Use precise ephemeris data (NOTE: this may degrade solution, run tomosar test precise-rktp to test)")
@click.option("-t", "--tag", default = "", flag_value=date.today().strftime('%Y%m%d'), help="Tag processing directory with specified string (default: the date of today)")
@click.option("-k", "--config", type=click.Path(exists=True, path_type=Path), default=None, help="Specify external config file for rnx2rtkp")
@click.option("-a", "--atx", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the satellite antenna .atx file")
@click.option("-r", "--receiver", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the .atx file containing receiver antenna info")
@click.option("--downloads", type=int, default=10, help="Max number of parallel downloads (default: 10)")
@click.option("--attempts", type=int, default=3, help="Max number of attempts for each file (default: 3)")
@click.option("-m", "--mask", "elevation_mask", type=float, default=None, help="Elevation mask for satellites")
@click.option("-l", "--line", type=int, default=1, help="Line in CSV file to read data from (default=1)")
@click.option("--offset", type=float, default=-0.079, help="Specify vertical PCO between mocoref data log receiver and drone processing receiver (default=-0.079) for CSV files")
def init(
    path: Path,
    force: bool,
    swepos: bool,
    ppp: bool,
    is_zip: bool,
    is_mocoref: bool,
    is_csv: bool,
    is_json: bool,
    is_llh: bool,
    header: bool,
    is_processing_dir: bool,
    use_precise: bool,
    tag: str,
    config: Path | None,
    atx: Path | None,
    receiver: Path | None,
    downloads: int,
    attempts: int,
    elevation_mask: float|None,
    line: int,
    offset: float,
) -> None:
    """Searches recursively from the PATH (default: CWD) to find matching files:
    (1) Drone GNSS .bin and .log;
    (2) Drone IMU .bin and .log;
    (3) Drone Radar .bin, .log and .cfg;
    (4) GNSS base station; and
    (5) Mocoref data or precise position of GNSS base station.
    
    If the GNSS base station file is missing, can fetch files from the nearest Swepos station,
    and can supplement Mocoref data by performing static PPP on the base station.
    Note that the path must point to a directory which contains exactly one set of drone data.
    For other files, tomosar init will use the first matching file it finds.

    For the GNSS base station a RINEX OBS file is prioritized over other files: HCN files and RTCM3 files are also accepted,
    as well as Reach ZIP archives.

    For mocoref data a mocoref.moco file is prioritized followed by a .json file, with the underlying assumption that these
    have been generated from raw mocoref data; then a .llh log is prioritized over a .csv file. If a Reach ZIP archive is
    used as the source of the GNSS base station file, the mocoref file will also be generated from there. 
    
    The files are converted where applicable and copied/moved into a processing directory, in such a way that the content
    of the data directory where tomosar init was initiated is left unaltered. Then preprocessing is initiated [ONLY GNSS IMPLEMENTED].
    By default tomosar init will use precise ephemeris data for the RTKP post processing, and will download this data if not
    available (disable by running with --broadcast)
    
    Note that tomosar init can also be run inside a processing directory, in which case it simply initiates preprocessing [ONLY GNSS IMPLEMENTED].
    Any directory inside the settings specified PROCESSING_DIRS is assumed to be a processing directory, and any directory
    outside is by default assumed to be a data directory (this behaviour can be overridden by the --processing option)."""
    
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
        with tmp(archive.parent / "obs_file.tmp") as obs_tmp:
            ubx2rnx(drone_files["Drone GNSS file"], nav=False, sbs=False, obs_file=obs_tmp)
            obs_data, (base_obs, mocoref_file, _) = reachz2rnx(files[0], rnx_file=obs_tmp, output_dir=tmp_dir)
        base_start, base_end = obs_data[base_obs]["TIME OF FIRST OBS"], obs_data[base_obs]["TIME OF LAST OBS"]
        if header:
            print(f"GNSS base generated from {archive}")
        else:
            base_pos = obs_data[mocoref_file]
            print(f"Mocoref data and GNSS base generated from {archive}")
        return base_obs, mocoref_file, base_pos, base_start, base_end
    
    def initiate_file(source: Path|tuple[Path, ...], target_dir: Path) -> Path:
        if isinstance(source, Path):
            source=(source,)
        target = []
        for file in source:
            if file == target_dir / file.name:
                target.append(file)
                continue
            target.append(Path(shutil.copy2(file, target_dir)))
            if tmp_dir in file.parents:
                file.unlink()
        if len(target) == 1:
            return target[0]
        else:
            return tuple(target)


    if swepos and ppp:
        warn("Both --swepos and --ppp cannot be used, ignoring -ppp.")
        ppp = False
        
    # Patterns to look for drone files
    drone_patterns: dict[str, re.Pattern] = {
        "Drone GNSS file": re.compile(r"^gnss_logger_dat-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.bin$"),
        "Drone GNSS log": re.compile(r"^gnss_logger_log-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.log$"),
        "Drone IMU file": re.compile(r"^imu_logger_dat-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.bin$"),
        "Drone IMU log": re.compile(r"^imu_logger_log-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.log$"),
        "Drone radar file": re.compile(r"^radar_logger_dat-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.bin$"),
        "Drone radar log": re.compile(r"^radar_logger_log-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.log$"),
        "Drone radar command": re.compile(r"^radar_logger_cmd-(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.cfg$"),
    }

    # Patterns to look for GNSS base station files
    gnss_patterns: dict[str, re.Pattern] = {
        "RINEX OBS": re.compile(r"^.+\.(\d{2}[Oo]|obs|OBS)$"),
        "HCN": re.compile(r"^.+\.(HCN|hcn)$"),
        "RTCM3": re.compile(r"^.+\.(RTCM3|rtcm3)$"),
        "Reach ZIP archive": re.compile(r"^Reach_\d+\.(zip|ZIP)$"),
    }

    navglo_pattern = re.compile(r"^.+\.(\d{2}[PpGg]|nav|NAV)")
    
    # Patterns to look for Mocoref data files
    mocoref_patterns: dict[str, re.Pattern] = {
        "mocoref": re.compile(r"^mocoref.moco$"),
        "JSON": re.compile(r"^.+\.(json|JSON)$"),
        "LLH": re.compile(r"^.+\.(llh|LLH)"),
        "CSV": re.compile(r"^.+\.(csv|CSV)$"),
    }

    sp3_patterns: dict[str, re.Pattern] = {
        "SP3": re.compile(r"^.+\.(sp3|SP3)$"),
        "CLK": re.compile(r"^.+\.(clk|CLK)$")
    }
    
    # Dicts to store matches
    drone_files: dict[str, list[tuple[Path, datetime]]] = {key: [] for key in drone_patterns}
    gnss_files: dict[str, list[Path]] = {key: [] for key in gnss_patterns}
    mocoref_files: dict[str, list[Path]] = {key: [] for key in mocoref_patterns}
    sp3_files: dict[str, list[Path]] = {key: [] for key in sp3_patterns}
    navglo = []

    # Search recursively
    for p in path.rglob("*"):
        if p.is_file():
            for key, regex in drone_patterns.items():
                match = regex.match(p.name)
                if match:
                    radar_dir = p.parent
                    dt = extract_datetime(p.name)
                    drone_files[key].append((p, dt))
            if not swepos:
                for key, regex in gnss_patterns.items():
                    match = regex.match(p.name)
                    if match:
                        gnss_files[key].append(p)
                if ppp:
                    match = navglo_pattern.match(p.name)
                    if match:
                        navglo.append(p)
                if not header and not ppp:
                    for key, regex in mocoref_patterns.items():
                        match = regex.match(p.name)
                        if match:
                            mocoref_files[key].append(p)
            if use_precise:
                for key, regex in sp3_patterns.items():
                    match = regex.match(p.name)
                    if match:
                        sp3_files[key].append(p)

    # Ensure that exactly one file is found for each drone type with matching datetimes and extract nominal datetime
    dt = None
    for key, matches in drone_files.items():
        # Ensure that exactly one file is found
        if not matches:
            raise FileNotFoundError(f"{key} not found.")
        files, dts = zip(*matches)
        if len(files) > 1:
            raise RuntimeError(f"Multiple {key}s found: {local(files, path)}")
        
        # Ensure datetimes match and extract nominal datetime
        if dt == None:
            dt = dts[0]
        else:
            if not matching_dt(dt, dts[0]):
                raise RuntimeError(f"Timestamps do not match: {dt} and {dts[0]}")
            
        # Store file
        drone_files[key] = files[0]

    drone_files: dict[str, Path]
    dt: datetime

    # Print files
    print("Found the following drone files:")
    for key, file in drone_files.items():
        print(f"{" " * 2}- {key}: {local(file, path)}")

    with tmp("tmp", allow_dir=True) as tmp_dir:
        # Work on SP3 files
        if sp3_files["SP3"]:
            eph_files = sp3_files["SP3"]
            eph_files.extend(sp3_files["CLK"])
            sp3_file, clk_file = merge_eph(eph_files, output_dir=tmp_dir)
            if sp3_file and tmp_dir not in sp3_file.parents:
                print(f"Located SP3 file: {sp3_file}")
            if clk_file and tmp_dir not in clk_file.parents:
                print(f"Located CLK file: {clk_file}")
        else:
            sp3_file = None
            clk_file = None
        
        # Convert drone GNSS to RINEX
        rover_obs, rover_nav, rover_sbs = ubx2rnx(drone_files["Drone GNSS file"], obs_file=tmp_dir / drone_files["Drone GNSS file"].with_suffix(".obs").name)
        
        # Work on base OBS and mocoref.moco
        if swepos:
            if is_zip or is_mocoref or is_csv or is_json or is_llh:
                warn("--swepos used: other mocoref options ignored")
            header = True
            base_obs, _ = run_fetch_swepos(drone_files["Drone GNSS file"], output_dir=tmp_dir)
            ground_dir = radar_dir.parent / "ground1"
            mocoref_dir = radar_dir.parent / "mocoref"
        else:
            mocoref_data_file = None
            if ppp:
                if is_zip or is_mocoref or is_csv or is_json or is_llh:
                    warn("--ppp used: other mocoref options ignored")
                mocoref_data = True
                header = False
            # Check if Mocoref data was found
            elif header:
                if is_mocoref or is_csv or is_json or is_llh:
                    warn("--header used: other mocoref options ignored.\nReading mocoref data from RINEX header. Use only if RINEX header is known to contain precise position.")
                else:
                    warn("Reading mocoref data from RINEX header. Use only if RINEX header is known to contain precise position.")
                mocoref_data = True
            else:
                if is_zip:
                    if is_mocoref or is_csv or is_json or is_llh:
                        raise ValueError("Only one of --zip, --mocoref, --csv, --json and --llh can be used")
                if is_mocoref:
                    if is_csv or is_json or is_llh:
                        raise ValueError("Only one of --zip, --mocoref, --csv, --json and --llh can be used")
                    mocoref_key = "mocoref"
                elif is_csv:
                    if is_json or is_llh:
                        raise ValueError("Only one of --zip, --mocoref, --csv, --json and --llh can be used")
                    mocoref_key = "CSV"
                elif is_json:
                    if is_llh:
                        raise ValueError("Only one of --zip, --mocoref, --csv, --json and --llh can be used")
                    mocoref_key = "JSON"
                elif is_llh:
                    mocoref_key = "LLH"
                else:
                    mocoref_key = None
                mocoref_data = False
                if not is_zip:
                    for key, files in mocoref_files.items():
                        if files and (mocoref_key is None or mocoref_key == key):
                            mocoref_data = True
                            mocoref_data_file = files[0]
                            mocoref_key = key

            # Extract matching GNSS base station
            base_obs = None
            if mocoref_data:
                for key, files in gnss_files.items():
                    if files:
                        # Used only if processing dir
                        ground_dir = files[0].parent
                        match key:
                            case "RINEX OBS":
                                # Copy, don't move
                                base_obs = files[0]
                                print(f"GNSS base located: {base_obs}")
                            case "HCN":
                                base_obs, _, _ = chc2rnx(files[0], obs_file=tmp_dir/files[0].with_suffix(".obs").name)
                                print(f"GNSS base generated from {files[0]}")
                            case "RTCM3":
                                with tmp(file.parent / "obs_file.tmp") as obs_tmp:
                                    ubx2rnx(drone_files["Drone GNSS file"], nav=False, sbs=False, obs_file=obs_tmp)
                                    tstart, tend, _, _ = extract_rnx_info(obs_tmp)
                                base_obs, _, _ = reach2rnx(files[0], obs_file=tmp_dir/files[0].with_suffix(".obs").name, tstart=tstart, tend=tend)
                                print(f"GNSS base generated from {files[0]}")
                            case "Reach ZIP archive":
                                # Mocoref data is extracted from the ZIP archive
                                mocoref_data_file = None 
                                # Used only if processing dir
                                mocoref_dir = files[0].parent.parent / "mocoref"
                                # Extract
                                base_obs, mocoref_file, base_pos, base_start, base_end = from_reachz(files[0])
                        break
                if mocoref_data_file:
                    # Used only if processing dir
                    mocoref_dir = mocoref_data_file.parent
                    # Get mocoref data and generate mocoref.moco file if necessary
                    base_pos, mocoref_file = generate_mocoref(mocoref_data_file, type=mocoref_key, generate=True, line=line, pco_offset=offset, output_dir=tmp_dir)
                    mocoref_data = True
                    print(f"Mocoref data extracted from {key} file: {local(mocoref_data_file, path)}")
                    if not mocoref_file:
                        # Occurs only if mocoref.moco file
                        mocoref_file = mocoref_data_file
            else:
                # Only Reach ZIP archive provides mocoref data
                if gnss_files["Reach ZIP archive"]:
                    # Used only if processing dir 
                    ground_dir, mocoref_dir = files[0].parent, files[0].parent.parent / "mocoref"
                    # Extract
                    base_obs, mocoref_file, base_pos, base_start, base_end = from_reachz(files[0])
                else:
                    raise FileNotFoundError(f"Could not find mocoref data.")

            if not base_obs:
                raise FileNotFoundError(f"Could not find GNSS base.")
            
            settings = Settings()
            if ppp:
                ground_dir = base_obs.parent
                mocoref_dir = ground_dir.parent / "mocoref"
                if navglo:
                    navglo = navglo[0]
                else:
                    navglo = None
                base_pos, sp3_file, clk_file = run_station_ppp(
                    obs_path=base_obs,
                    navglo_path=navglo,
                    atx_path=atx,
                    antrec_path=receiver,
                    max_downloads=downloads,
                    max_retries=attempts,
                    elevation_mask=elevation_mask,
                    out_path=tmp_dir,
                    header=False,
                    retain=True
                )
                lon, lat, h = ecef_to_geo.transform(*base_pos)
                mocoref_dict = {
                    settings.MOCOREF_LATITUDE: lat,
                    settings.MOCOREF_LONGITUDE: lon,
                    settings.MOCOREF_HEIGHT: h,
                    settings.MOCOREF_ANTENNA: 0.,
                }
                _, mocoref_file = generate_mocoref(mocoref_dict, generate=True, output_dir=tmp_dir)

        base_start, base_end, header_pos, _ = extract_rnx_info(base_obs)
        if header:
            ground_dir = base_obs.parent
            mocoref_dir = ground_dir.parent / "mocoref"
            lon, lat, h = ecef_to_geo.transform(*header_pos) 
            mocoref_dict = {
                settings.MOCOREF_LATITUDE: lat,
                settings.MOCOREF_LONGITUDE: lon,
                settings.MOCOREF_HEIGHT: h,
                settings.MOCOREF_ANTENNA: 0.
            }
            base_pos, mocoref_file = generate_mocoref(mocoref_dict, generate=True, output_dir=tmp_dir)
        
        # Check if processing directory
        if not is_processing_dir:
            if path.is_relative_to(settings.PROCESSING_DIRS):
                is_processing_dir = True
        
        if is_processing_dir:
            processing_dir = path
        else:
            processing_dir = settings.PROCESSING_DIRS / (path.name + "_" + tag)
            processing_dir.mkdir(exist_ok=force)
            
            rawdata_dir = processing_dir / "rawdata" / dt.strftime('%Y%m%d')
            rawdata_dir.mkdir(exist_ok=force, parents=True)
            radar_dir = rawdata_dir  / "radar1"
            radar_dir.mkdir(exist_ok=force)
            ground_dir = rawdata_dir / "ground1"
            ground_dir.mkdir(exist_ok=force)
            mocoref_dir = rawdata_dir / "mocoref"
            mocoref_dir.mkdir(exist_ok=force)

            # Copy drone files
            for key, file in drone_files.items():
                drone_files[key] = shutil.copy2(file, radar_dir)
        
        # Initiate drone RINEX files
        rover_obs, rover_nav, rover_sbs = initiate_file((rover_obs, rover_nav, rover_sbs), radar_dir)
        
        # Initiate base_obs and mocoref_file
        base_obs = initiate_file(base_obs, ground_dir)
        mocoref_file = initiate_file(mocoref_file, mocoref_dir)

        # If station-ppp was used initate SP3 and CLK files
        if ppp:
            sp3_file = initiate_file(sp3_file, radar_dir)
            clk_file = initiate_file(clk_file, radar_dir)
        
        # Move into processing dir
        os.chdir(processing_dir)

    print("All files located, dropping you into the processing directory ... ", end="\n\n")
    drop_into_terminal(processing_dir)
    
    # Initiate preprocessing
    if swepos and not elevation_mask:
        if use_precise or sp3_file:
            elevation_mask = 20
        else:
            elevation_mask = 5
    coords, gpst, q = rtkp(
        rover_obs=rover_obs,
        base_obs=base_obs,
        nav_file=rover_nav,
        sbs_file=rover_sbs,
        sp3_file=sp3_file,
        clk_file=clk_file,
        precise=use_precise,
        out_path=rover_obs.with_suffix(".pos"),
        config_file=config,
        elevation_mask=elevation_mask,
        mocoref_file=mocoref_file,
        retain=True
    )

    fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(8, 8))
    axs = axs.flatten()
    ax = axs[0]
    ax.plot(gpst[q==1], coords[:,2][q==1], 'g+')
    ax.plot(gpst[q!=1], coords[:,2][q!=1], 'r+')
    ax.set_xlabel("GPST (s)")
    ax.set_ylabel("Ellipsoidal Height (m)")
    ax = axs[1]
    ax.plot(coords[:,1][q==1], coords[:,0][q==1], 'g+')
    ax.plot(coords[:,1][q!=1], coords[:,0][q!=1], 'r+')
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    fig_name = dt.strftime("%Y-%m-%d-%H-%M-%S-position.pdf")
    fig.savefig(radar_dir / fig_name, format="pdf")

    # Continue with IMU and unimoco ... 

@click.command()
@click.argument("archive", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", "output_dir", type=click.Path(exists=False, file_okay=False, path_type=Path), default=None, help="Extract files into given folder (default: parent of archive)")
@click.option("--rover", "rover_obs", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Target rover OBS for RTKP processing")
@click.option("-n", "--nav", "extract_nav", is_flag=True, help="Extract NAV file")
@click.option("--verbose", is_flag=True, help="Verbose mode.")
def extract_reach(archive: Path, output_dir: Path, rover_obs: Path, extract_nav: bool, verbose: bool) -> None:
    """Extracts a Reach ZIP archive to produce:
    (1) A RINEX OBS file for a single site,
    (2) A mocoref.moco for the OBS file, and
    (3) A RINEX NAV file (optional).
    
    Optionally takes a RINEX OBS file as input to extract from the archive the OBS file which has the greatest overlap with the
    input RINEX file. Otherwise extracts the longest segment."""

    reachz2rnx(archive=archive, output_dir=output_dir, rnx_file=rover_obs, nav=extract_nav, verbose=verbose)

@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("-d","--dry", is_flag=True, help="Do not generate file (displays result)")
@click.option("--csv", is_flag=True, help="File is a CSV file (default for .csv and .CSV files)")
@click.option("--json", is_flag=True, help="File is a JSON file (default for .json and .JSON files)")
@click.option("--llh", is_flag=True, help="File is an LLH log (default for .llh and .LLH files)")
@click.option("-l", "--line", type=int, default=1, help="Line in CSV file to read data from (default=1)")
@click.option("--offset", type=float, default=-0.079, help="Specify vertical PCO between mocoref data log receiver and drone processing receiver (default=-0.079) for CSV files")
def mocoref(path, dry, csv, json, llh, line, offset) -> None:
    """Generate a mocoref file from a data file"""
    if csv:
        if json or llh:
            raise ValueError("Only one of --csv, --json and --llh can be used")
        type = "CSV"
    elif json:
        if llh:
            raise ValueError("Only one of --csv, --json and --llh can be used")
        type = "JSON"
    elif llh:
        type = "LLH"
    else:
        type = None
    _, mocoref_file = generate_mocoref(data=path, type=type, line=line, generate=not dry, verbose=dry, pco_offset=offset)
    print(f"Mocoref file {local(mocoref_file)} generated from {local(path)}")

@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--stations", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the SWEPOS coordinate list CSV")
@click.option("--downloads", type=int, default=10, help="Max number of parallel downloads (default: 10)")
@click.option("--attempts", type=int, default=3, help="Max number of attempts for each file (default: 3)")
@click.option("-o", "--output", type=click.Path(path_type=Path), default="SWEPOS", help="Output directory for SWEPOS RINEX files")
@click.option("-d", "--dry", is_flag=True, help="Dry run without downloads")
@click.option("--cont", is_flag=True, help="Continue run after downloads complete")
@click.option("-n","--nav", is_flag=True, help="Also fetch nav files.")
def fetch_swepos(path, stations, downloads, attempts, output, dry, cont, nav) -> None:
    """Extract GNSS info from rover gnss log or RINEX observation file and find nearest SWEPOS station.
    Then download files into output directory."""
    run_fetch_swepos(
        filepath=path,
        stations_path=stations,
        max_downloads=downloads,
        max_retries=attempts,
        dry=dry,
        output_dir=output,
        fetch_nav=nav,
        cont=cont
    )

@click.command()
@click.argument("obs_file", type=click.Path(exists=True, path_type=Path))
@click.option("-n", "--navglo", type=click.Path(exists=True, path_type=Path), default=None, help="Path to navigation data for GLONASS (can be general/merged NAV file)")
@click.option("-a", "--atx", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the satellite antenna .atx file")
@click.option("-r", "--receiver", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the .atx file containing receiver antenna info")
@click.option("--downloads", type=int, default=10, help="Max number of parallel downloads (default: 10)")
@click.option("--attempts", type=int, default=3, help="Max number of attempts for each file (default: 3)")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=False, flag_value=True, help="Generate .out file at specified path (default: next to OBS file)")
@click.option("-d", "--dry", is_flag=True, help="Dry run without downloads")
@click.option("-x", "--no-header", 'header', is_flag=True, default=True, flag_value=False, help="Do not modify OBS file header with new position.")
@click.option("--retain", is_flag=True, help="Retain ephemeris data files after finishing")
@click.option("-m", "--mocoref", "make_mocoref", is_flag=True, help="Generate mocoref.moco file with position")
def station_ppp(
    obs_file: Path,
    navglo: Path|None,
    atx: Path|None,
    receiver: Path|None,
    downloads: int,
    attempts: int,
    output: Path|bool,
    dry: bool,
    header: bool,
    retain: bool,
    make_mocoref: bool
) -> None:
    """Attempt to determine base station position using PPP post processing."""
    if output == True:
        output = obs_file.with_suffix(".out")
    if output == False:
        output = None
    run_station_ppp(
        obs_path=obs_file,
        navglo_path=navglo,
        atx_path=atx,
        antrec_path=receiver,
        max_downloads=downloads,
        max_retries=attempts,
        dry=dry,
        out_path=output,
        header=header,
        retain=retain,
        mocoref=make_mocoref
    )

@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("-l", "--linear", type=int, default=0, help="Specify linear track index to modify radar-[...].inf (0 for spiral flights)")
@click.option("-v", "--verbose", is_flag=True, help="Print detailed output")
@click.option("-d", "--dry", is_flag=True, help="Don't save or modify files")
@click.option("--dem", type=click.Path(exists=True, path_type=Path), default=None, help="Path to DEM file or folder to combine with DEMS_GROUND")
@click.option("--npar", type=int, default=None, help="Number of parallel processes (default: CPU count)")
def trackfinder(path, linear, verbose, dry, dem, npar) -> None:
    """Run trackfinder on a .moco file containing integrated GNSS and IMU data."""
    run_trackfinder(
        path=path,
        dem_path=dem,
        linear=linear,
        verbose=verbose,
        dry=dry,
        npar=npar
    )

@click.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--out", type=click.Path(path_type=Path), default=".", help="Output directory (default: '.')")
@click.option("-t", "--tag", type=str, default="", flag_value=None, help="Tag output .tomo directory with an extra string (default: latest slice processing date)")
@click.option("--single", is_flag=True, help="Process 1-look data in interferometric bands")
@click.option("--nopair", is_flag=True, help="Avoid processing 2-look data in interferometric bands")
@click.option("--RR", is_flag=True, help="Estimate RR and SSF in multilooked tomogram")
@click.option("--fused", is_flag=True, help="Process only fused tomograms")
@click.option("--sub", is_flag=True, help="Process only subsurface tomograms")
@click.option("--sup", is_flag=True, help="Process only supersurface tomograms")
@click.option("--canopy", is_flag=True, help="Process only canopy tomograms")
@click.option("--phh", is_flag=True, help="Only process files from P-band")
@click.option("--lxx", is_flag=True, help="Only process files from L-band")
@click.option("--lhh", is_flag=True, help="Only process L-band files with HH-pol")
@click.option("--lvv", is_flag=True, help="Only process L-band files with VV-pol")
@click.option("--lhv", is_flag=True, help="Only process L-band files with HV-pol")
@click.option("--lvh", is_flag=True, help="Only process L-band files with VH-pol")
@click.option("--cvv", is_flag=True, help="Only process files from C-band")
@click.option("--load", is_flag=True, help="Load generated tomogram scenes into an interactive Python console")
@click.option("-m", "--masks", type=str, default="", help="Folder containing shapefile masks (in addition to TOMOMASKS)")
@click.option("-n", "--npar", type=int, default=os.cpu_count(), help="Number of parallel threads")
@click.option("--folder", type=str, default=None, help="Filter all files not in the provided folder")
@click.option("-d", "--date", type=str, default=None, help="Filter all files where the flight date does not match")
@click.option("-t", "--time", type=str, default=None, help="Filter all files where the flight time does not match")
@click.option("-s", "--spiral", type=int, default=None, help="Filter all files where the spiral ID does not match")
@click.option("-w", "--width", type=float, default=None, help="Filter all files where the processed width does not match")
@click.option("-r", "--res", type=float, default=None, help="Filter all files where the processing resolution does not match")
@click.option("-f", "--refr", type=float, default=None, help="Filter all files where the refractive index does not match")
@click.option("--lat", type=float, default=None, help="Filter all files where the central latitude does not match")
@click.option("--lon", type=float, default=None, help="Filter all files where the central longitude does not match")
@click.option("--thresh", type=float, default=None, help="Filter all files where the processing threshold does not match")
@click.option("--smo", type=float, default=None, help="Filter all files where the smoothing parameter does not match")
@click.option("--ham", type=float, default=None, help="Filter all files where the Hamming window parameter does not match")
@click.option("--squint", type=float, default=None, help="Filter all files where the squint parameter does not match")
@click.option("--text", type=str, default=None, help="Filter all files which do not contain a matching text tag")
@click.option("--DC", type=float, default=None, help="Filter all files where DC parameter does not match")
@click.option("--DL", type=float, default=None, help="Filter all files where the DL parameter does not match")
@click.option("--HC", type=float, default=None, help="Filter all files where the HC parameter does not match")
@click.option("--HV", type=float, default=None, help="Filter all files where the HV parameter does not match")
def forge(paths, single, nopair, RR, fused, sub, sup, canopy,
         phh, lxx, lhh, lvv, lhv, lvh, cvv, load,
         out, masks, npar, folder, date, time, spiral, width, res, refr,
         lat, lon, thresh, smo, ham, squint, text, DC, DL, HC, HV) -> TomoScenes:
    """Forge slices into Tomogram Directories."""

    time_start = Time.time()

    print("Input paths:", paths)
    print("Output directory:", out)
    print("Mask directory:", masks)
    print("Parallel threads:", npar)

    # Construct filter
    folder = os.path.abspath(folder) if folder else None
    date_obj = datetime.strptime(date, "%Y-%m-%d") if date else datetime.strptime("1900-01-01", "%Y-%m-%d")
    if time:
        timestamp = datetime.strptime(time, "%H:%M:%S")
        date_obj = date_obj.replace(hour=timestamp.hour, minute=timestamp.minute, second=timestamp.second)

    bands = []
    if phh: bands.append("phh")
    if lxx: bands.extend(["lhh", "lvv", "lhv", "lvh"])
    if lhh: bands.append("lhh")
    if lvv: bands.append("lvv")
    if lhv: bands.append("lhv")
    if lvh: bands.append("lvh")
    if cvv: bands.append("cvv")

    filter = ImageInfo(
        folder=folder, filename=None, date=date_obj, spiral=spiral, band=bands,
        width=width, res=res, smo=smo, ham=ham, refr=refr, lat=lat, lon=lon,
        hoff=None, depth=None, DC=DC, DL=DL, HC=HC, HV=HV, thresh=thresh,
        squint=squint, text=text
    )

    # Dispatch processing
    scenes = tomoforge(
        paths=paths, filter=filter, single=single, nopair=nopair, RR=RR,
        fused=fused, sub=sub, sup=sup, canopy=canopy,
        masks=masks, npar=npar, out=out
    )

    print(f"Processing completed in {Time.time() - time_start:.2f} seconds.")
    if load:
        interactive_console({"scenes": scenes})