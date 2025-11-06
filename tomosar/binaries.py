import shutil
import re
from collections import defaultdict
import json
from importlib.resources import files
import subprocess
from pathlib import Path
import json
from importlib.resources import path as importpath
from contextlib import contextmanager
from typing import Iterator
from contextlib import ExitStack
import rasterio
from rasterio.io import DatasetReader
from xml.etree import ElementTree as ET
import numpy as np
from pyproj import Transformer
from datetime import datetime, timezone, timedelta

from .utils import changed, warn, local, generate_mocoref
from .config import Settings, LOCAL

# Catch-all run command for binariy executables
def run(cmd: str | list, capture: bool = True):
    
    if not isinstance(cmd, list):
        cmd = [cmd]
    binary_name = cmd[0]
    require_binary(binary_name)
    cmd = [local(part) if isinstance(part, Path) else str(part) for part in cmd]
    if Settings().VERBOSE:
            print(' '.join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=capture, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed with exit code {e.returncode}.\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        ) from e

# Check if required binary is installed
def require_binary(name: str, hint: str|None = None) -> str:
    try:
        dep = load_dependencies()[name]
    except:
        raise RuntimeError(f"Failed to find required binary '{name}' in tomosar.setup.dependencies.json")
    path = shutil.which(name)
    if path is None:
        if not hint:
            raise RuntimeError(f"Required binary '{name}' not found in PATH.\n\033[1mSource\033[22m: {dep['Source']}")
        else:
            raise RuntimeError(f"Required binary '{name}' not found in PATH.\n{hint}")
    return path

def load_dependencies() -> dict[str,dict]:
    path = files("tomosar.setup").joinpath("dependencies.json")
    with open(path) as f:
        return json.load(f)

def check_required_binaries() -> None:
    deps = load_dependencies()
    missing = []
    for name, dep in deps.items():
        try:
            hint = json.dumps(dep, indent=4)
            require_binary(name, hint=hint)
        except RuntimeError as e:
            print(f"[Missing] {e}")
            missing.append(name)
    if missing:
        print(f"{len(missing)} missing binaries: {missing}")
        return
    else:
        print("All required binaries are available.")

def substitute_flags(path: Path, resolved: dict):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(r"\{\{(\w+)\}\}")

    def replacer(match):
        return resolved.get(match.group(1), match.group(0))

    substituted = pattern.sub(replacer, content)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(substituted)

# Get resource file
@contextmanager
def _resolve_resources(keys, resource_func, **kwargs) -> Iterator[dict]:
    """Context manager that resolves multiple resource keys and keeps them alive."""
    with ExitStack() as stack:
        resolved = {}
        for key in keys:
            try:
                ctx = resource_func(None, key, **kwargs)
                path = stack.enter_context(ctx)
                resolved[key] = str(path) if path and path.is_file() else ""
            except Exception as e:
                warn(f"Could not resolve resource for key '{key}': {e}")
                resolved[key] = ""
        yield resolved

@contextmanager
def resource(path: str | Path | None, key: str = None, **kwargs) -> Iterator[Path]:
    """Provides a temporary local copy of the file specified in path. If path is empty or None,
    provides a temporary local copy of a file specified in settings or fetched internally based on key.

    Will resolve patterns matching {{KEY}} inside files to point to matching files, which are also provided as temporary copies.

    If no file is provided None is yielded.

    Valid keys: 'SATELLITES', 'SWEPOS_COORDINATES', 'RTKP_CONFIG', 'RECEIVER' 'DEM'', 'CANOPY', 'TEST_FILE_SVB', 'TEST_FILE_SAVAR'
    NOTE: RTKP_CONFIG can be called with standard=True to disable Explorer specific options in internal files
    NOTE: RECEIVER takes antenna (required) and radome (optional) keywords
    NOTE: DEM and CANOPY take bounds (required) and res (required) keywords"""
    # Generate a local copy of a file
    def local_copy(file: Path) -> Path:
        if file.is_file():
            tmp_path = Path.cwd() / file.with_suffix(".tmp").name
            shutil.copy2(file, tmp_path)
            return tmp_path
        raise FileNotFoundError(f"{file} not found")
    
    # Prepare tmp_path
    tmp_path = None
    
    # Check if path was specified
    if path:
        path = Path(path)
        if path.exists() and path.is_file():
            tmp_path = local_copy(path)
        else:
            if key:
                warn(f"Path {path} does not point to a file. Ignoring.")
            else:
                raise RuntimeError(f"Path {path} does not point to a file.")
    
    if key:
        key = key.upper()
    # Match against file keys for Settings reference or internal pointers    
    match key:
        case "SATELLITES":
            filename = Settings().SATELLITES
            if filename:
                path = Path(filename)
                if path.exists():
                    tmp_path = local_copy(path)
                else:
                    raise ValueError(f"Path {path} read from settings {key} does not point to a file.")
        case "SWEPOS_COORDINATES":
            filename = Settings().SWEPOS_COORDINATES
            if filename:
                path = Path(filename)
                if path.is_file():
                    tmp_path = local_copy(path)
                else:
                   raise ValueError(f"Path {path} read from settings {key} does not point to a file.") 
        case "RTKP_CONFIG":
            filename = Settings().RTKP_CONFIG
            standard = kwargs.get("standard",False)
            if filename and not standard:
                path = Path(filename)
                if path.exists():
                    tmp_path = local_copy(path)
                else:
                    raise ValueError(f"Path {path} read from settings {key} does not point to a file.")
            if standard:
                with importpath('tomosar.data', "m8t_5hz_standard.conf") as path:
                    tmp_path = local_copy(path)
        case "RECEIVER":
            antenna = kwargs.get("antenna", None)
            if antenna is None:
                raise KeyError("To get a RECEIVER resource, antenna must be specified (antenna=)")
            radome = kwargs.get("radome", "NONE")
            filename = Settings().RECEIVER(receiver_id=antenna, radome=radome)
            if filename:
                path = Path(filename)
                if path.exists():
                    tmp_path = local_copy(path)
                else:
                    warn(f"Path {path} read from settings does not point to a file. Ignoring")
        case "DEM":
            bounds = kwargs.get("bounds", None)
            if bounds is None:
                raise KeyError("To get a DEM resource, bounds must be specified (bounds=)")
            res = kwargs.get("resolution", None)
            if res is None:
                raise KeyError("To get a DEM resource, resolution must be specified (resolution=)")
            vrt_path = LOCAL / "DEM.vrt"
            vrt_path = build_vrt(vrt_path, Settings().DEMS)
            tmp_path = generate_raster(vrt_path, bounds, res)   
        case "CANOPY":
            bounds = kwargs.get("bounds", None)
            if bounds is None:
                raise KeyError("To get a CANOPY resource, bounds must be specified (bounds=)")
            res = kwargs.get("resolution", None)
            if res is None:
                raise KeyError("To get a CANOPY resource, resolution must be specified (resolution=)")
            vrt_path = LOCAL / "CANOPY.vrt"
            vrt_path = build_vrt(Settings().CANOPIES)
            tmp_path = generate_raster(vrt_path)
        case "TEST_FILE_SVB":
            with importpath('tomosar.test_files',"minimal_svb.ubx") as test_file:
                tmp_path = local_copy(test_file)
        case "TEST_FILE_SAVAR":
            with importpath('tomosar.test_files',"minimal_savar.ubx") as test_file:
                tmp_path = local_copy(test_file)
    
    # Find all {{KEY}} in the file
    try:
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        keys = set(re.findall(r"\{\{(\w+)\}\}", content))
    except (TypeError, UnicodeDecodeError):
        keys = {}

    # Resolve and substitute
    if keys:
        with _resolve_resources(keys, resource, **kwargs) as resolved:
            substitute_flags(tmp_path, resolved)
            # Yield temporary local path
            try:
                yield tmp_path
            finally:
                if isinstance(tmp_path, Path): # Not None
                    tmp_path.unlink(missing_ok=True)  # Clean up after use
    else:
        # Yield temporary local path
        try:
            yield tmp_path
        finally:
            if isinstance(tmp_path, Path): # Not None
                tmp_path.unlink(missing_ok=True)  # Clean up after use

@contextmanager
def tmp(*args, temporary: bool = True, allow_dir: bool = False) -> Iterator[Path]:
    """Makes a path or several paths temporary within the context. If allow_dir is False, the path must point
    to a file or not exist; if the path does not exist, it will still be unlinked after the context ends (allowing
    file generation within the context). If allow_dir is True and path does not exist when tmp is called,
    a temporary directory matching the path will be generated. If the path points to a directory, ALL content
    is also made temporary.
    
    If a falsy value is passed tmp will yield None."""
    def unlink(paths: tuple[Path, ...]) -> None:
        """Helper function to unlink a tuple of paths"""
        for path in paths:
            if path.is_file():
                path.unlink()
            if path.is_dir():
                shutil.rmtree(path)

    if not args:
        yield None
    paths = tuple(Path(path) for path in args)
    for path in paths:
        if not allow_dir and path.is_dir():
            raise ValueError("To allow directories to be made temporary, specify allow_dir=True.")
        if allow_dir and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    try:
        yield paths[0] if len(paths) == 1 else paths
    finally:
        if temporary:
            unlink(paths)
              
# Get elevation from DEMs for a point
def elevation(lat: float, lon: float, dem_path: Path|str = None) -> float | None:
    """Returns the elevation for a specific point from the highest resolved DEM at that point,
    or from the user specified DEM."""
    def  get_dem() -> tuple[np.ndarray, DatasetReader]:
        def _check_file_type(filename: Path) -> str:
            ext = filename.suffix.lower()
            if ext in [".tif", ".tiff"]:
                return "TIFF"
            elif ext == ".vrt":
                return "VRT"
            else:
                return "Unknown"
        file_type = _check_file_type(dem_path)
        def _point_in_bounds(src: DatasetReader) -> bool:
            bounds = src.bounds
            try:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                return bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top
            except Exception:
                return False

        def _find_raster_in_vrt(vrt_path: Path) -> str | None:
            tree = ET.parse(vrt_path)
            root = tree.getroot()
            for source in root.iter("SourceFilename"):
                raster_name = source.text
                raster_path = vrt_path.parent / raster_name
                try:
                    with rasterio.open(raster_path) as src:
                        if _point_in_bounds(src, lat, lon):
                            return str(raster_path)
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
          
# Named functions for binary executables
def crx2rnx(crx_file: str|Path) -> Path:
    rnx_path = crx_file.with_suffix('.rnx')
    run(['crx2rnx', crx_file])
    crx_file.unlink(missing_ok=True)
    return rnx_path

def _generate_merged_filenames(files: list[Path]) -> Path | None:
    pattern = re.compile(
        r"^(?:(?P<station>[A-Z0-9]{9})_)?(?P<source>[A-Z0-9]{1,10})_(?P<datetime>\d{11})_(?P<duration>\d{2}[SMHD])(?:_(?P<freq>\d{2}[SMHD]))?_(?P<type>[A-Z]+)\.(?P<ext>[A-Za-z0-9]{3})$"
    )
    units = re.compile(r"(?P<number>\d{2})(?P<unit>[SMHD])")
    grouped = defaultdict(set)

    for f in files:
        if (match := pattern.match(f.name)):
            type_raw = match.group("type")
            if type_raw.endswith("N"):
                type = "NAV"
            elif type_raw.endswith("O"):
                type = "OBS"
            else:
                type = type_raw

            stat = match.group("station") or ""
            freq = match.group("freq") or ""
            key = (
                stat,
                match.group("source"),
                freq,
                type,
                match.group("duration"),
                match.group("ext")
            )
            grouped[key].add(int(match.group("datetime")))

    descriptive_filenames = []
    for (station, source, frequency, type, duration, ext), datetimes in grouped.items():
        datetime = min(datetimes)
        if (match := units.match(duration)):
            dur = int(match.group("number"))
            dur_unit = match.group("unit")
        else:
            raise RuntimeError(f"Failed to parse filenames: {files}")
        no_files = len(datetimes)
        dur *= no_files

        if type == "OBS":
            out_type = "MO"
            out_ext = "obs"
        elif type == "NAV":
            out_type = "MN"
            out_ext = "nav"
        elif type in {"ORB", "CLK"}:
            out_type = type
            out_ext = ext
        else:
            out_type = type
        if station:
            merged_filename = f"{station}_{source}_{datetime}_{dur:02}{dur_unit}"
        else:
            merged_filename = f"{source}_{datetime}_{dur:02}{dur_unit}"
        if frequency:
            merged_filename += f"_{freq}"
        merged_filename += f"_{out_type}.{out_ext}"
        merged_filename = files[0].parent / merged_filename
        descriptive_filenames.append(merged_filename)

    if len(descriptive_filenames) > 1:
        raise RuntimeError(f"Unable to merge files due to conflicting sources, frequencies, durations, types or extensions. Merging aborted. Files: {files}")
    return descriptive_filenames[0] if descriptive_filenames else None

def merge_rnx(rnx_files: list[str|Path], force: bool = False) -> Path|None:
    merged_file = _generate_merged_filenames(rnx_files)
    if merged_file and merged_file.exists() and not force:
        print(f"Discovered merged file {local (merged_file)}. Aborting merge of RNX files.")
        return merged_file
    print(f"Merging rinex files > {local(merged_file)} ...", end=" ", flush=True)
    run(["gfzrnx", "-f", "-q", "-finp"] + [f for f in rnx_files] + ["-fout", merged_file])
    print("done.")
    return merged_file

def merge_eph(eph_files: list[str|Path], force: bool = False) -> tuple[Path|None, Path|None]:
    eph_files = [Path(f) for f in eph_files]
    # Get .SP3 files
    sp3_files = [f for f in eph_files if f.suffix == ".SP3"]
    # Merge
    merged_sp3 = _generate_merged_filenames(sp3_files)
    if merged_sp3 and  merged_sp3.exists() and not force:
        print(f"Discovered merged file {local(merged_sp3)}. Aborting merge of .SP3 files.")
    else:
        print(f"Merging .SP3 files > {local(merged_sp3)} ...", end=" ", flush=True)
        with merged_sp3.open("w", encoding="utf-8") as out_file:
            for file_path in sp3_files:
                with Path(file_path).open("r", encoding="utf-8") as in_file:
                    for line in in_file:
                        out_file.write(line)
        print("done.")

    # Get .CLK files
    clk_files = [f for f in eph_files if f.suffix == ".CLK"]
    # Merge
    merged_clk = _generate_merged_filenames(clk_files)
    if merged_clk and  merged_clk.exists() and not force:
        print(f"Discovered merged file {local(merged_clk)}. Aborting merge of .CLK files.")
    else:
        print(f"Merging .CLK files > {local(merged_clk)} ...", end=" ", flush=True)
        with merged_clk.open("w", encoding="utf-8") as out_file:
            for file_path in clk_files:
                with Path(file_path).open("r", encoding="utf-8") as in_file:
                    for line in in_file:
                        out_file.write(line)
        print("done.")

    return merged_sp3, merged_clk

def ubx2rnx(ubx_file: str|Path, nav: bool = True, sbs: bool = True) -> tuple[Path, Path|None, Path|None]:
    """Convert a UBX file (drone gnss_logger_dat-[...].bin file) to RINEX."""
    obs_path = ubx_file.with_suffix(".obs")
    cmd = ["convbin", "-r", "ubx", "-od", "-os", "-o", obs_path]
    if nav:
        nav_path = ubx_file.with_suffix(".nav")
        cmd.extend(["-n", nav_path])
    else:
        nav_path = None
    if sbs:
        sbs_path = ubx_file.with_suffix(".sbs")
        cmd.extend(["-s", sbs_path])
    else:
        sbs_path = None
    cmd.append(ubx_file)
    result = run(cmd)
    print(result.stdout)

    return obs_path, nav_path, sbs_path

def _split_by_site_occupation(
        rnx_file: Path|str,
        output_path: Path|str|None = None,
        single: bool = True,
        tstart: datetime = None,
        tend: datetime = None,
        verbose: bool = False
    ) -> dict[Path, dict]:
    """Splits a RINEX file by blocks with 'EVENT: NEW SITE OCCUPATION' lines, using the main header
    but the APPROX POSITION XYZ line updated from the new block along with FIRST OBS TIME and LAST OBS TIME.
    Returns a dict indexed by output path with the following nested keys:
    - "APPROX POSITION XYZ": a 3-tuple of floats with the header ECEF coordinates,
    - "TIME OF FIRST OBS": a datetime object representing the first obs, and
    - "TIME OF LAST OBS": a datetime object representing the last obs."""

    def get_metadata() -> tuple[tuple[float,float,float], datetime, datetime]:
        """Helper function to get metadata from current_position, first_observation, first_constellation, 
        current_observation and current_constellation"""
        # Position in ECEF coordinates
        parts = current_position.split()
        try:
            x, y, z = map(float, parts[0:3])
            position = (x, y, z)
        except:
            raise RuntimeError(f"Failed to parse APPROX POSITION XYZ line: {current_position}")
        
        # First observation epoch
        parts = first_observation.split()
        try:
            year, month, day, hour, minute, second = map(float, parts[1:7])
            first_obs = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=timezone.utc)
        except:
            raise RuntimeError(f"Failed to parse first observation line: {first_observation}")
        if len(first_constellation) > 1:
            first_obs = (first_obs, "MIX")
        elif len(first_constellation) == 1:
            first_obs = (first_obs, first_constellation.pop())
        else:
            raise RuntimeError(f"No constellation found for first observation: {first_observation}")
        
        # Last observation epoch
        parts = current_observation.split()
        try:
            year, month, day, hour, minute, second = map(float, parts[1:7])
            last_obs = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=timezone.utc)
        except:
            raise RuntimeError(f"Failed to parse last observation line: {current_observation}")
        if len(current_constellation) > 1:
            last_obs = (last_obs, "MIX")
        elif len(current_constellation) == 1:
            last_obs = (last_obs, current_constellation.pop())
        else:
            raise RuntimeError(f"No constellation found for last observation: {current_observation}")

        return position, first_obs, last_obs
                
    with open(rnx_file, 'r') as f:
        lines = f.readlines()

    # Extract header
    header = []
    i = 0
    while not lines[i].strip().endswith("END OF HEADER"):
        if "APPROX POSITION XYZ" in lines[i]:
            header_position = lines[i]
        header.append(lines[i])
        i += 1
    header.append(lines[i])  # Add END OF HEADER
    i += 1
    
    # Set verbose
    verbose = verbose or Settings.VERBOSE
    if verbose:
        print(f"Working on file: {rnx_file} to find SITES ...", flush=True)

    # Prepare for splitting
    blocks = []
    current_block = []
    current_position = header_position
    current_observation = None
    first_observation = None

    constellation_map = {
        'G': 'GPS',
        'R': 'GLONASS',
        'E': 'GALILEO',
        'C': 'BEIDOU',
        'J': 'QZSS',
        'I': 'IRNSS',
        'S': 'SBAS'
    }

    valid_prefixes = ('>', 'G', 'R', 'E', 'C', 'J', 'I', 'S')
    clean_lines = [line for line in lines if line.strip() and line.strip()[0] in valid_prefixes or "EVENT:" in line or "COMMENT" in line]
    
    while i < len(lines):
        line = lines[i]
        if not line.strip() or (line.strip()[0] not in valid_prefixes and "EVENT:" not in line and "COMMENT" not in line):
            # Skip malformed lines
            i += 1
            continue
        if "EVENT: NEW SITE OCCUPATION" in line:
            # Save current block if it exists
            if current_block:
                blocks.append((current_block, *get_metadata()))
                current_block = []

            # Clear observations
            first_observation = None
            first_constellation = set()
            current_observation = None
            current_constellation = set()

            # Look ahead for APPROX POSITION XYZ
            j = i
            while j < len(lines):
                # Position of new block
                if "APPROX POSITION XYZ" in lines[j]:
                    current_position = lines[j]
                # Observation data starts in new block
                if lines[j].startswith(">"):
                    i = j
                    break
                j += 1
        else:
            if line.startswith(">"):
                if current_observation and not first_observation:
                    first_observation = current_observation
                    first_constellation = current_constellation
                current_observation = line
                current_constellation = set()
            if line.strip()[0] in constellation_map:
                current_constellation.add(constellation_map[line.strip()[0]])
            current_block.append(line)
            i += 1

    # Add last block
    if current_block:
        blocks.append((current_block, *get_metadata()))

    # Write out each block
    output = {}
    max_dur = (timedelta(), None)
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = rnx_file.with_suffix(".obs")
    for idx, (block_lines, position, first_obs, last_obs) in enumerate(blocks):
        # Update header with new position and first and last obs times
        position_line = f"{position[0]:14.4f}{position[1]:14.4f}{position[2]:14.4f}{" " * 18}APPROX POSITION XYZ\n"
        first_obs_line = f"{f'{first_obs[0].year:04}':>6}{f'{first_obs[0].month:02}':>6}{f'{first_obs[0].day:02}':>6}{f'{first_obs[0].hour:02}':>6}{f'{first_obs[0].minute:02}':>6}{f'{first_obs[0].second + first_obs[0].microsecond * 1E-6:02.7f}':>13}{first_obs[1]:>8}{" "*9}TIME OF FIRST OBS\n"
        last_obs_line = f"{f'{last_obs[0].year:04}':>6}{f'{last_obs[0].month:02}':>6}{f'{last_obs[0].day:02}':>6}{f'{last_obs[0].hour:02}':>6}{f'{last_obs[0].minute:02}':>6}{f'{last_obs[0].second + last_obs[0].microsecond * 1E-6:02.7f}':>13}{last_obs[1]:>8}{" "*9}TIME OF LAST OBS\n"
        for i, line in enumerate(header):
            if "APPROX POSITION XYZ" in line:
                header[i] = position_line
                header_position = position_line
            if "TIME OF FIRST OBS" in line:
                header[i] = first_obs_line
            if "TIME OF LAST OBS" in line:
                header[i] = last_obs_line
        lines = header + block_lines
        if verbose:
            print(f"SITE PARSED:\n{position_line}{first_obs_line}{last_obs_line}", end="")
        if single:
            t2 = min(last_obs[0], tend) if tend else last_obs[0]
            t1 = max(first_obs[0], tstart) if tstart else first_obs[0]
            if t2 - t1 > max_dur[0]:
                max_dur = (t2 - t1, idx)
                output[output_path] = {
                    "RINEX": ''.join(lines),
                    "APPROX POSITION XYZ": position,
                    "TIME OF FIRST OBS": first_obs[0],
                    "TIME OF LAST OBS": last_obs[0]
                }
                if verbose:
                    print("... NEW MAXIMUM OBS: True", flush=True)
            elif verbose:
                print("... NEW MAXIMUM OBS: False", flush=True)
        else:
            site_out_path = output_path.with_suffix(f".{idx:02}.obs")
            output[site_out_path] = {
                "RINEX": ''.join(lines),
                "APPROX POSITION XYZ": position,
                "TIME OF FIRST OBS": first_obs[0],
                "TIME OF LAST OBS": last_obs[0]
            }
    
    for path, value in output.items():
        path.write_text(value.pop("RINEX", None))

    return output

def reach2rnx(rtcm_file: str|Path, reference_date: datetime|None = None, obs_file: str|Path|None = None, single: bool = True, tstart: datetime = None, tend: datetime = None, nav: bool = False, sbs: bool = False, verbose: bool = False) -> tuple[dict|None, Path|None, Path|None]:
    """Convert a Reach RTCM3 file to RINEX with correct header(s). If single is False, produces a separate file for each site,
    otherwise extracts only the site with the longest observation. If tstart and tend are given, only observations
    within the specified interval count against observation length (but the entire observation time is still recorded)."""
    rtcm_file = Path(rtcm_file)
    if not obs_file:
        obs_file = rtcm_file.with_suffix(".obs")

    if not reference_date:
        if dt := re.search(r'\d{14}', rtcm_file.with_suffix("").name):
            reference_date = datetime.strptime(dt.group(), '%Y%m%d%H%M%S')
        else:
            raise ValueError(f"Could not determine a reference timestamp from the file name ({rtcm_file.name}). Please provide it explicitly.")

    with tmp(rtcm_file.with_name(rtcm_file.stem + "_OBS.tmp")) as obs_path:
        cmd = ["convbin", "-r", "rtcm3", "-od", "-os", '-tr', reference_date.strftime('%Y/%m/%d'), reference_date.strftime('%H:%M:%S'), "-o", obs_path]
        if nav:
            nav_path = obs_path.with_suffix(".nav")
            cmd.extend(["-n", nav_path])
        else:
            nav_path = None
        if sbs:
            sbs_path = obs_path.with_suffix(".sbs")
            cmd.extend(["-s", sbs_path])
        else:
            sbs_path = None
        cmd.append(rtcm_file)
        result = run(cmd)
        print(result.stdout)
        if obs_path.exists():
            _update_antenna(obs_path, antenna="EML_REACH_RS3", radome="NONE", verbose=verbose)
            obs_files = _split_by_site_occupation(obs_path, output_path=obs_file, single=single, tstart=tstart, tend=tend, verbose=verbose)
        else:
            obs_files = None

    return obs_files, nav_path, sbs_path

def chc2rnx(hcn_file: str|Path, nav: bool = False, sbs: bool = False) -> tuple[Path, Path|None, Path|None]:
    """Convert a CHCI83 HCN file to RINEX with correct header."""
    obs_path = hcn_file.with_suffix(".obs")
    cmd = ["convbin", "-r", "nov", "-od", "-os", "-o", obs_path]
    if nav:
        nav_path = hcn_file.with_suffix(".nav")
        cmd.extend(["-n", nav_path])
    else:
        nav_path = None
    if sbs:
        sbs_path = hcn_file.with_suffix(".sbs")
        cmd.extend(["-s", sbs_path])
    else:
        sbs_path = None
    cmd.append(hcn_file)
    result = run(cmd)
    print(result.stdout)
    if obs_path.exists():
        _update_antenna(obs_path, antenna="CHCI83", radome="NONE")

    return obs_path, nav_path, sbs_path

def _update_antenna(file_path, antenna: str, radome: str, verbose: bool = False) -> None:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    new_line = f"{' ' * 20}{antenna:<16}{radome:<24}ANT # / TYPE\n"

    updated = False
    for i, line in enumerate(lines):
        if "ANT # / TYPE" in line:
            lines[i] = new_line
            updated = True
            break
    
    if not updated:
        raise RuntimeError("Failed to parse RINEX header.")

    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    if verbose or Settings().VERBOSE:
        print(f"Updated RINEX header in {file_path} to set antenna={antenna} and radome={radome}")

def rnx2rtkp(
        rover_obs: str|Path,
        base_obs: str|Path,
        nav_file: str|Path,
        out_path: str|Path,
        config_file: str|Path = None,
        sbs_file: str|Path = None,
        elevation_mask: float|None = None,
        mocoref_file: str|Path = None,
        mocoref_type: str|None = None,
        mocoref_line: int = 1
) -> None:
    """Runs RTKLIB's rnx2rtkp with dynamic command construction based on available resources."""
    antenna_type, radome = _ant_type(base_obs)
    print(f"Detected base antenna type: {antenna_type} {radome}")
    with resource(None, "SATELLITES") as atx:
        with resource(None, "RECEIVER", antenna=antenna_type, radome=radome) as receiver:
            if receiver is None:
                receiver_file = atx
            else:
                receiver_file = receiver
            
            constellations, freqs, fallback = _parse_atx(receiver_file, antenna_type=antenna_type, radome=radome, mode="rnx2rtkp")
            if fallback:
                print("Defaulted to NONE radome")
            if constellations:
                print(f"Avaialable constellations: {','.join(constellations)}")
                match freqs:
                    case '1':
                        print("Available frequencies: L1")
                    case '2':
                        print("Available frequencies: L1+L2")
                    case '3':
                        print("Available frequencies: L1+L2+L5")
            else:
                print("No callibration data available. Using all constellations and frequencies.")

    with resource(config_file, "RTKP_CONFIG", antenna=antenna_type, radome=radome) as config:
        cmd = ['rnx2rtkp', '-k', str(config), '-o', out_path]
        if elevation_mask:
            cmd.extend(['-m', elevation_mask])
        if constellations:
            cmd.extend(["-sys", ",".join(constellations), "-f", freqs])
        if mocoref_file:
            (mocoref_latitude, mocoref_longitude, mocoref_height), _ = generate_mocoref(mocoref_file, type=mocoref_type, line=mocoref_line, generate=False)
            cmd.extend(["-l", mocoref_latitude, mocoref_longitude, mocoref_height])
        if fallback:
            with resource(base_obs) as tmp_obs:
                _update_antenna(tmp_obs, antenna=antenna_type, radome='NONE')
                cmd.extend([rover_obs, tmp_obs, nav_file])
                if sbs_file:
                    cmd.append(sbs_file)
                print(f"Running RTKP post processing ...\n\tRover: {local(rover_obs)}\n\tBase: {local(tmp_obs)}\n\tNav: {local(nav_file)}\n", flush=True)
                try: 
                    run(cmd, capture=False)
                except RuntimeError:
                    if config_file is None:
                        with resource(None, "RTKP_CONFIG", standard=True) as new_config:
                            cmd[2] = new_config
                            run(cmd)
        else:        
            cmd.extend([rover_obs, base_obs, nav_file])
            if sbs_file:
                cmd.append(sbs_file)
            print(f"Running RTKP post processing ...\n\tRover: {local(rover_obs)}\n\tBase: {local(base_obs)}\n\tNav:{local(nav_file)}\n", flush=True)
            try: 
                run(cmd, capture=False)
            except RuntimeError:
                if config_file is None:
                    with resource(None, "RTKP_CONFIG", standard=True) as new_config:
                        cmd[2] = new_config
                        run(cmd)
    print("Done.")

def _ant_type(rinex_path) -> tuple[None|list[str], None|str]:
    with open(rinex_path, 'r') as f:
        for line in f:
            if 'ANT # / TYPE' in line:
                # ANT TYPE and RADOME is typically in columns 21–60
                try:
                    spl = line[20:60].split()
                    if len(spl) == 2:
                        ant_type, radome = spl
                    if len(spl) == 1:
                        ant_type = spl[0]
                        radome = "NONE"
                    return ant_type, radome
                except ValueError:
                    raise ValueError("Failed to parse ANTENNA TYPE from RINEX header")
    warn("ANTENNA TYPE not specified in RINEX header")
    return None, None

def _parse_atx(file_path, antenna_type, radome, mode: str = "glab") -> tuple[list[str], str, bool]:
    """
    Parses an ATX file to find the entry for a given antenna type and radome. If the radome is not found, defaults to 'NONE'.
    Runs in two modes: 'glab' or 'rnx2rtkp'. In gLAB-mode:
        Returns a list of strings representing available constellations and frequencies, e.g., ['G12', 'R1'],
        and string with unavailable constellations, e.g. '-CJSI0'.
    In rnx2rtkp-mode:
        Returns a list of strings representing available constellations, e.g. ['G', 'R'],
        and a string representing available frequencies across all constellations, e.g. '2' ('1': L1, '2': L1+L2, '3': L1+L2+L5)
    In both modes it also returns a boolean:
        fallback = True if radome defaulted to 'NONE' else False
    """
    mode = mode.casefold()

    if not mode in ('glab', 'rnx2rtkp'):
        raise ValueError("The following modes are available: 'glab' and 'rnx2rtkp'.")
    # Read the ATX file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Normalize inputs
    antenna_type = antenna_type.strip().upper()
    radome = radome.strip().upper()

    # Find the matching entry
    entry_lines = {}
    for i, l in enumerate(lines):
        line = l.rstrip()
        if line.endswith("TYPE / SERIAL NO"):
            parts = line[0:60].split()
            if not antenna_type == parts[0]:
                continue
            if radome == parts[1]:
                entry_lines['target'] = i
                break
            elif "NONE" == parts[1]:
                entry_lines['fallback'] = i

    # Check if target or else falback was found
    if 'target' in entry_lines:
        target_line = entry_lines['target']
        fallback = False
    elif 'fallback' in entry_lines:
        target_line = entry_lines['fallback']
        fallback = True
    else:
        return [], "", False
    
    freq_pattern = re.compile(r"^(?P<const>[A-Z])(?P<freq>\d{2})")
    freq_map = {}
    for l in lines[target_line:]:
        line = l.rstrip()
        if line.endswith("END OF ANTENNA"):
            break
        if line.endswith("START OF FREQUENCY"):
            parts = line[0:60].split()
            match = freq_pattern.match(parts[0])
            if match:
                constellation = match.group(1)
                frequency = int(match.group(2))
                if constellation not in freq_map:
                    freq_map[constellation] = set()
                freq_map[constellation].add(frequency)

    # Format output strings
    match mode:
        case "glab":
            frequencies = []
            for constellation, freqs in freq_map.items():
                sorted_freqs = sorted(freqs)
                frequencies.append(f"{constellation}{''.join(map(str, sorted_freqs))}")
            
            unavailable = []
            # G = GPS
            # R = GLONASS
            # E = Galileo
            # C = BeiDou
            # J = QSS
            # S = SBAS
            # I = IRNS/NAVIC
            for constellation in ['G', 'R', 'E', 'C', 'J', 'S', 'I']:
                if constellation not in freq_map:
                    unavailable.append(constellation)
            unavailable = "-" + "".join(unavailable) + '0'

            return frequencies, unavailable, fallback
        
        case "rnx2rtkp":
            constellations = []
            frequencies = None
            for constellation, freqs in freq_map.items():
                if constellation == 'S':
                    break # SBAS satellites are not used by rnx2rtkp
                if 1 in freqs:
                    constellations.append(constellation)
                    if 2 in freqs:
                        if 5 in freqs:
                            frequencies = min(3, frequencies) if frequencies else 3
                        else:
                            frequencies = min(2, frequencies) if frequencies else 2
                    else:
                        frequencies = 1
            
            return constellations, str(frequencies), fallback

def ppp(
        obs_file: str|Path,
        sp3_file: str|Path,
        clk_file: str|Path,
        out_path: str|Path,
        navglo_file: str|Path = None,
        atx_file: str|Path = None,
        antrec_file: str|Path = None,
        elevation_mask: float|None = None
    ) -> str:
    """Runs gLAB with static PPP mode to determine the position of a GNSS base from its RINEX observation file.
    Returns the content of the out file."""
    with resource(atx_file, "SATELLITES") as atx:
        antenna_type, radome = _ant_type(obs_file)
        print(f"Detected antenna type: {antenna_type} {radome}")
        with resource(antrec_file, "RECEIVER", antenna=antenna_type, radome=radome) as receiver:
            if receiver is None:
                receiver_file = atx
            else:
                receiver_file = receiver
            cmd = [
                'glab',
                    '-input:obs', obs_file,
                    '-input:ant', atx,
                    '-input:orb', sp3_file,
                    '-input:clk', clk_file,
                    '--summary:waitfordaystart',
                    '-summary:formalerrorver', '0.0013',
                    '-summary:formalerror3d', '0.002',
                    '-summary:formalerrorhor', '0.0013'
            ]
            if elevation_mask:
                cmd.extend(['-pre:elevation', elevation_mask])
            freqs, unavailable, fallback = _parse_atx(receiver_file, antenna_type=antenna_type, radome=radome, mode="glab")
            if fallback:
                print("Defaulted to NONE radome ...")
            if freqs:
                print(f"Available frequencies: {freqs}")
                for f in freqs:
                    cmd.extend(['-pre:availf', f])
                if unavailable:
                    cmd.extend(['-pre:sat', unavailable])
            else:
                print("No callibration data available. Using all frequencies.")
                cmd.append("--model:recphasecenter")
            if navglo_file:
                cmd.extend(["-input:navglo", navglo_file])
            if receiver:
                cmd.extend(["-input:antrec", receiver])
            print(f"Running station PPP on {local(obs_file)}{f' > {local(out_path)}' if out_path else ''} ...", end=" ", flush=True)
            result = run(cmd)
            if out_path:
                out_path.write_text(result.stdout)
            print("done. ")
    
    return result.stdout

# VRT
def build_vrt(vrt_path: Path|str, paths: list[Path|str]) -> Path:
    """
    Build a .vrt mosaic from all .tif files, returning the file path if a single GeoTIFF file is passed
    """
    tif_files = []
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix in (".tif", ".tiff"):
            tif_files.append(path)
        else:
            new_files = list(path.rglob("*.tif"))
            new_files.extend(list(path.rglob(".tiff")))
            tif_files.extend(new_files)
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFF files found in {paths}")
    if len(tif_files) == 1:
        return tif_files[0]
    
    # Check if the current VRT matches input files and build new VRT otherwise
    vrt_path = Path(vrt_path)
    hash_path = vrt_path.with_suffix(".hash")
    if changed(hash_path, tif_files):
        print("Building new VRT ...", end=" ", flush=True)
        cmd = ["gdalbuildvrt", "-resolution", "highest", vrt_path] + [f for f in tif_files]
        run(cmd)

def generate_raster(
        vrt_path: Path|str,
        bounds: tuple[float, float, float, float],
        res: tuple[float, float]
    ) -> Path:

    out_path = Path.cwd() / Path(vrt_path).with_suffix(".tmp").name
    cmd = ["gdalwarp", "-te", *bounds, "-tr", *res, "-r", "average", "-of", "GTiff", vrt_path, out_path]
    run(cmd)
    return out_path