from tqdm import tqdm
from datetime import datetime, timedelta, date, timezone
import pytz
import pandas as pd
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ftplib import FTP
import time
import numpy as np
import matplotlib.pyplot as plt
import re
import zipfile
import shutil
from collections import defaultdict
import json
from sklearn.cluster import DBSCAN

from .version import version
from .config import Settings
from .utils import prompt_ftp_login, gunzip, warn, local, string_sub, date_to_gps_week, gps_week_to_date, leap_seconds, parse_datetime_string
from .manager import run, tmp, resource, modify_config
from .transformers import geo_to_ecef, ecef_to_geo, ecef_to_enu, change_rf, geo_to_map

# Named functions for binary executables
def crx2rnx(crx_file: str|Path) -> Path:
    rnx_path = crx_file.with_suffix('.rnx')
    run(['crx2rnx', crx_file])
    crx_file.unlink(missing_ok=True)
    return rnx_path

def _generate_merged_filenames(files: list[Path], output_dir: Path|str|None = None) -> tuple[Path|None, datetime|None, datetime|None]:
    pattern = re.compile(
        r"^(?:(?P<station>[A-Z0-9]{9})_)?(?P<source>[A-Z0-9]{1,10})_(?P<datetime>\d{11})_(?P<duration>\d{2}[SMHD])(?:_(?P<freq>\d{2}[SMHD]))?_(?P<type>[A-Z]+)\.(?P<ext>[A-Za-z0-9]{3})$"
    )
    units = re.compile(r"(?P<number>\d{2})(?P<unit>[SMHD])")
    grouped = defaultdict(set)
    if not files:
        return None, None, None
    
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
        dt_min = min(datetimes)
        if (match := units.match(duration)):
            base_dur = int(match.group("number"))
            dur_unit = match.group("unit")
        else:
            raise RuntimeError(f"Failed to parse filenames: {files}")
        no_files = len(datetimes)
        dur = base_dur * no_files

        if type == "OBS":
            out_type = "MO"
            out_ext = "obs"
        elif type == "NAV":
            out_type = "MN"
            out_ext = "nav"
        else:
            out_type = type
            out_ext = ext
        if station:
            merged_filename = f"{station}_{source}_{dt_min}_{dur:02}{dur_unit}"
        else:
            merged_filename = f"{source}_{dt_min}_{dur:02}{dur_unit}"
        if frequency:
            merged_filename += f"_{frequency}"
        merged_filename += f"_{out_type}.{out_ext}"
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = files[0].resolve().parent
        merged_filename = output_dir / merged_filename
        # Verify that the included files form a sequence
        pattern = re.compile(r'(?P<year>\d{4})(?P<doy>\d{3})(?P<hour>\d{2})(?P<minute>\d{2})')
        match dur_unit:
            case "S":
                step = timedelta(seconds=base_dur)
            case "M":
                step = timedelta(minutes=base_dur)
            case "H":
                step = timedelta(hours=base_dur)
            case "D":
                step = timedelta(days=base_dur)
        start = None
        current = None
        previous = None
        for dt in sorted(datetimes):
            match = pattern.match(str(dt))
            if current:
                previous = current
            current = datetime(year=int(match.group("year")), month=1, day=1, hour=int(match.group("hour")), minute=int(match.group("minute"))) + timedelta(days=int(match.group("doy"))-1)
            if previous and not current == previous + step:
                missing_dt = previous + step
                missing_dt = f"{missing_dt.year:04}{missing_dt.timetuple().tm_yday:03}{missing_dt.hour:02}{missing_dt.minute:02}"
                if station:
                    missing_file = f"{station}_{source}"
                else:
                    missing_file = f"{source}"
                missing_file += f"_{missing_dt}_{duration}"
                if frequency:
                    missing_file += f"_{frequency}"
                missing_file += f"_{out_type}.{out_ext}"
                raise FileNotFoundError(f"Unable to merge files due to missing file: {missing_file}")
            if not start:
                start = current
        # If successful append merged filename
        descriptive_filenames.append((merged_filename, start, current + step))

    # Verfiy that only files from a single sequence were passed
    if len(descriptive_filenames) > 1:
        raise RuntimeError(f"Unable to merge files due to conflicting sources, frequencies, durations, types or extensions. Merging aborted. Files: {files}")
    return descriptive_filenames[0] if descriptive_filenames else (None, None, None)

def merge_rnx(rnx_files: list[str|Path], force: bool = False, output_dir: Path|str|None = None) -> Path|None:
    if len(rnx_files) == 1:
        if rnx_files[0] == output_dir / rnx_files[0]:
            return rnx_files[0]
        return shutil.copy2(rnx_files[0], output_dir)
    
    merged_file, t_start, t_end = _generate_merged_filenames(rnx_files, output_dir=output_dir)
    if not merged_file:
        return None
    if merged_file.exists() and not force:
        print(f"Discovered merged file {local (merged_file)}. Aborting merge of RNX {merged_file.suffix[1:].upper()} files.")
        return merged_file
    print(f"Merging RINEX {merged_file.suffix[1:].upper()} files > {local(merged_file)} ...", end=" ", flush=True)
    run(["gfzrnx", "-f", "-q", "-finp"] + [f for f in rnx_files] + ["-fout", merged_file])
    print("done.")
    return merged_file

def ubx2rnx(ubx_file: str|Path, nav: bool = True, sbs: bool = True, obs_file: str|Path|None = None) -> tuple[Path, Path|None, Path|None]:
    """Convert a UBX file (drone gnss_logger_dat-[...].bin file) to RINEX."""
    if obs_file:
        obs_path = Path(obs_file)
    else:
        obs_path = ubx_file.with_suffix(".obs")
        
    cmd = ["convbin", "-r", "ubx", "-od", "-os", "-o", obs_path]
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
    cmd.append(ubx_file)
    print(f"Converting {local(ubx_file)} ...\n-->{local(obs_path)}\n{f'-->{local(nav_path)}\n' if nav else ''}{f'-->{local(sbs_path)}\n' if sbs else ''}", end="")
    run(cmd)

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
        position_line = f"{position[0]:14.4f}{position[1]:14.4f}{position[2]:14.4f}{' ' * 18}APPROX POSITION XYZ\n"
        first_obs_line = f"{f'{first_obs[0].year:04}':>6}{f'{first_obs[0].month:02}':>6}{f'{first_obs[0].day:02}':>6}{f'{first_obs[0].hour:02}':>6}{f'{first_obs[0].minute:02}':>6}{f'{first_obs[0].second + first_obs[0].microsecond * 1E-6:02.7f}':>13}{first_obs[1]:>8}{' '*9}TIME OF FIRST OBS\n"
        last_obs_line = f"{f'{last_obs[0].year:04}':>6}{f'{last_obs[0].month:02}':>6}{f'{last_obs[0].day:02}':>6}{f'{last_obs[0].hour:02}':>6}{f'{last_obs[0].minute:02}':>6}{f'{last_obs[0].second + last_obs[0].microsecond * 1E-6:02.7f}':>13}{last_obs[1]:>8}{' '*9}TIME OF LAST OBS\n"
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
    within the specified interval count against observation length (but the entire observation time is still recorded).
    
    The reference_date parameter is used to get the correct GPS week, and if not provided will be inferred from the filename if
    possible. Otherwise this will raise a ValueError."""
    rtcm_file = Path(rtcm_file)
    if obs_file:
        obs_file = Path(obs_file)
    else:
        obs_file = rtcm_file.with_suffix(".obs")

    if not reference_date:
        if dt := re.search(r'\d{14}', rtcm_file.with_suffix("").name):
            reference_date = datetime.strptime(dt.group(), '%Y%m%d%H%M%S')
        else:
            raise ValueError(f"Could not determine a reference timestamp from the file name ({rtcm_file.name}). Please provide it explicitly.")

    print(f'Converting {local(rtcm_file)} with reference date {reference_date.strftime('%Y/%m/%d')} {reference_date.strftime('%H:%M:%S')}...\n-->{local(obs_path)}\n{f'-->{local(nav_path)}\n' if nav else ''}{f'-->{local(sbs_path)}\n' if sbs else ''}')
    with tmp(rtcm_file.with_name(rtcm_file.stem + "_OBS.tmp")) as obs_path:
        cmd = ["convbin", "-r", "rtcm3", "-od", "-os", '-tr', reference_date.strftime('%Y/%m/%d'), reference_date.strftime('%H:%M:%S'), "-o", obs_path]
        if nav:
            nav_path = obs_file.with_suffix(".nav")
            cmd.extend(["-n", nav_path])
        else:
            nav_path = None
        if sbs:
            sbs_path = obs_file.with_suffix(".sbs")
            cmd.extend(["-s", sbs_path])
        else:
            sbs_path = None
        cmd.append(rtcm_file)
        run(cmd)
        if obs_path.exists():
            update_antenna(obs_path, antenna="EML_REACH_RS3", radome="NONE", verbose=verbose)
            obs_files = _split_by_site_occupation(obs_path, output_path=obs_file, single=single, tstart=tstart, tend=tend, verbose=verbose)
            if verbose or Settings().VERBOSE:
                print(f"{obs_path} generated the following corrected RINEX OBS file(s):")
                for file in obs_files:
                    print(f"{' '*2}{file}")
        else:
            obs_files = None

    return obs_files, nav_path, sbs_path

def chc2rnx(hcn_file: str|Path, nav: bool = False, sbs: bool = False, obs_file: str|Path|None = None) -> tuple[Path, Path|None, Path|None]:
    """Convert a CHCI83 HCN file to RINEX with correct header."""
    if obs_file:
        obs_path = Path(obs_file)
    else:
        obs_path = hcn_file.with_suffix(".obs")
    print(f"Converting {local(hcn_file)} ...\n-->{local(obs_path)}\n{f'-->{local(nav_path)}\n' if nav else ''}{f'-->{local(sbs_path)}\n' if sbs else ''}", end="")
    cmd = ["convbin", "-r", "nov", "-od", "-os", "-o", obs_path]
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
    cmd.append(hcn_file)
    run(cmd)
    if obs_path.exists():
        update_antenna(obs_path, antenna="CHCI83", radome="NONE")

    return obs_path, nav_path, sbs_path

def rnx2rtkp(
        rover_obs: str|Path,
        base_obs: str|Path,
        nav_file: str|Path,
        out_path: str|Path,
        config_file: str|Path|None = None,
        sbs_file: str|Path|None = None,
        sp3_file: str|Path|None = None,
        clk_file: str|Path|None = None,
        elevation_mask: float|None = None,
        mocoref_file: str|Path|None = None,
        mocoref_type: str|None = None,
        mocoref_line: int = 1,
        antenna_type: str = None,
        radome: str = "NONE",
        constellations: list[str] = [],
        freqs: str = "",
        processing_frame: str = "ITRF"
) -> str:
    """Runs RTKLIB's rnx2rtkp with dynamic command construction based on available resources."""
    cmd = ['rnx2rtkp']
    if out_path:
        capture = False
        cmd.extend(['-o', out_path])
    else:
        capture = True
    if config_file:
        cmd.extend(['-k', config_file])
    if elevation_mask:
        cmd.extend(['-m', elevation_mask])
    if constellations:
        cmd.extend(["-sys", ",".join(constellations), "-f", freqs])
    if mocoref_file:
        mocoref_pos, _ = generate_mocoref(mocoref_file, type=mocoref_type, line=mocoref_line, generate=False, output_frame=processing_frame)
        cmd.extend(["-r", *mocoref_pos])
    with resource(base_obs) as tmp_obs:
        if antenna_type:
            update_antenna(tmp_obs, antenna=antenna_type, radome=radome)
        cmd.extend([rover_obs, tmp_obs, nav_file])
        if sbs_file:
            cmd.append(sbs_file)
        if sp3_file:
            cmd.append(sp3_file)
        if clk_file:
            cmd.append(clk_file)
        result = run(cmd, capture=capture)
        return result.stdout

def glab_ppp(
        obs_file: str|Path,
        sp3_file: str|Path,
        clk_file: str|Path|None = None,
        inx_file: str|Path|None = None,
        out_path: str|Path|None = None,
        navglo_file: str|Path = None,
        atx_file: str|Path = None,
        antrec_file: str|Path = None,
        elevation_mask: float|None = None
    ) -> str:
    """Runs gLAB with static PPP mode to determine the position of a GNSS base from its RINEX observation file.
    Returns the content of the out log of gLAB run, and writes it to out_path if provided."""
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
                    '--summary:waitfordaystart',
                    '-summary:formalerrorver', '0.0013',
                    '-summary:formalerror3d', '0.002',
                    '-summary:formalerrorhor', '0.0013'
            ]
            if sp3_file and clk_file:
                cmd.extend([
                    '-input:orb', sp3_file,
                    '-input:clk', clk_file,
                ])
            elif sp3_file:
                cmd.extend(['-input:sp3', sp3_file])
            else:
                raise FileNotFoundError("No SP3 file was provided")
            if elevation_mask:
                cmd.extend(['-pre:elevation', elevation_mask])
            freqs, unavailable, fallback = parse_atx(receiver_file, antenna_type=antenna_type, radome=radome, mode="glab")
            if fallback:
                print("Defaulted to NONE radome")
            if freqs:
                print(f"Using callibrated frequencies: {freqs}")
                for f in freqs:
                    cmd.extend(['-pre:availf', f])
                if unavailable:
                    cmd.extend(['-pre:sat', unavailable])
            else:
                print("No callibration data available. Using all available frequencies.")
                cmd.append("--model:recphasecenter")
            if navglo_file:
                cmd.extend(["-input:navglo", navglo_file])
            if receiver:
                cmd.extend(["-input:antrec", receiver])
            if inx_file:
                cmd.extend(["-input:inx", inx_file])
            result = run(cmd)
            if out_path:
                out_path.write_text(result.stdout)
    
    return result.stdout

# RINEX helpers
def extract_rnx_info(file_path: str|Path) -> tuple[datetime|None,
                                                   datetime|None,
                                                   tuple[float, float, float],
                                                   tuple[float,float,float]]:
    """
    Extract start/end times and approximate position from RINEX header. Output:
    - TIME OF FIRST OBS (datetime or None): if not present in header reads from RINEX epochs
    - TIME OF LAST OBS (datetime or None): if not present in header reads from RINEX epochs
    - APPROX POS XYZ (tuple of floats)
    - ANTENNA: DELTA H/E/N (tuple of floats)
    """
    earliest_date = datetime(year=2020, month=1, day=1,tzinfo=timezone.utc)
    start_time = None
    end_time = None
    approx_position = None
    antenna_delta = (0., 0., 0.)
    with open(file_path, 'r', errors='ignore') as f:
        for line in f:
            if 'TIME OF FIRST OBS' in line:
                parts = line.strip().split()
                try:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    hour, minute, second = int(parts[3]), int(parts[4]), float(parts[5])
                    start_time = datetime(year, month, day, hour, minute, int(second), tzinfo=timezone.utc)
                except:
                    pass
            elif 'TIME OF LAST OBS' in line:
                parts = line.strip().split()
                try:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    hour, minute, second = int(parts[3]), int(parts[4]), float(parts[5])
                    end_time = datetime(year, month, day, hour, minute, int(second), tzinfo=timezone.utc)
                except:
                    pass
            elif 'APPROX POSITION XYZ' in line:
                parts = line.strip().split()
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    approx_position = (x, y, z)
                except:
                    pass
            elif 'ANTENNA: DELTA H/E/N' in line:
                parts = line.strip().split()
                antenna_delta = (float(parts[1]), float(parts[2]), float(parts[0]))
    if start_time is None or start_time < earliest_date:
        start_time = get_first_rinex_timestamp(file_path)
    if end_time is None or end_time < earliest_date:
        end_time = get_last_rinex_timestamp(file_path)
    return start_time, end_time, approx_position, antenna_delta

def get_first_rinex_timestamp(file_path: str|Path) -> datetime | None:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header_ended = False
        for line in f:
            line = line.strip()
            if not header_ended:
                if "END OF HEADER" in line:
                    header_ended = True
                continue

            if line.startswith('>'):
                parts = line.split()
                year, month, day, hour, minute, second = map(float, parts[1:7])
                return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=timezone.utc)
    return None

def get_last_rinex_timestamp(file_path: str|Path) -> datetime | None:
    with open(file_path, 'rb') as f:
        # Read file in reverse
        for line in reversed(list(f.readlines())):
            try:
                line = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue  # Skip malformed lines

            if line.startswith('>'):
                parts = line.split()
                # Format: > YYYY MM DD HH MM SS.SSS...
                year, month, day, hour, minute, second = map(float, parts[1:7])
                return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=timezone.utc)
    return None

def update_rinex_position(file_path, new_coords) -> None:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    if np.isnan(new_coords).any():
        warn("NaN coordinate detected. Header not updated.")
        return
    new_line = f"{new_coords[0]:14.4f}{new_coords[1]:14.4f}{new_coords[2]:14.4f}{" " * 18}APPROX POSITION XYZ\n"

    updated = False
    for i, line in enumerate(lines):
        if "APPROX POSITION XYZ" in line:
            lines[i] = new_line
            updated = True
            break
    
    if not updated:
        for i, line in enumerate(lines):
            if "END OF HEADER" in line:
                lines.insert(i, new_line)
                break
    
    lines.insert(2,"THE COORDINATES HAVE BEEN UPDATED IN WGS84                  COMMENT\n")

    with open(file_path, 'w') as file:
        file.writelines(lines)
    print(f"{local(file_path)} header position updated.")

def update_antenna(file_path, antenna: str, radome: str, verbose: bool = False) -> None:
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

# Ephemeris helpers
def splice_sp3(eph_files: list[Path], force: bool = False, output_dir: Path|str|None = None) -> tuple[Path|None, Path|None]:
    """Splice multiple SP3 files into one, preserving header and data integrity"""
    if len(eph_files) == 1:
        if eph_files[0] == output_dir / eph_files[0].name:
            return eph_files[0]
    
    eph_files.sort()
    merged_file, t_start, t_end = _generate_merged_filenames(eph_files, output_dir=output_dir)
    if not merged_file:
        return None
    if merged_file.exists() and not force:
        print(f"Discovered merged file {local(merged_file)}. Aborting merge of SP3 files.")
        return merged_file
    print(f"Merging SP3 files > {local(merged_file)} ...", end=" ", flush=True)
    # Write spliced file
    with merged_file.open("w", encoding="utf-8") as out_file:
        first_epoch = False
        for i, file_path in enumerate(eph_files):
            with file_path.open("r", encoding="utf-8") as in_file:
                for line in in_file:
                    if "EOF" in line:
                        continue # Skip EOF line
                    if i != 0 and line.strip()[0] not in ("*", "P", "V"):
                        continue # Add only Epochs and Position or Velocity records from subsequent files
                    if i==0 and not first_epoch and line.strip()[0] == "*":
                        first_epoch = True
                        # Add line noting splice
                        if len(eph_files) > 1:
                            out_file.write(f"/* Spliced by TomoSAR v{version[:5]} from {len(eph_files)} files\n")
                    out_file.write(line)
        out_file.write("EOF\n")
    print("done.")

    return merged_file

def splice_clk(clk_files: list[Path], output_dir: Path, force: bool = False) -> Path:
    """
    Splice multiple RINEX CLK files into one, preserving the header and data integrity.
    """
    if len(clk_files) == 1:
        if clk_files[0] == output_dir / clk_files[0].name:
            return clk_files[0]
   
    clk_files.sort()
    merged_file, t_start, t_end = _generate_merged_filenames(clk_files, output_dir=output_dir)
    if not merged_file:
        return None
    if merged_file.exists() and not force:
        print(f"Discovered merged file {local(merged_file)}. Aborting merge of CLK files.")
        return merged_file
    print(f"Merging CLK files > {local(merged_file)} ...", end=" ", flush=True)
    # Write spliced file
    with merged_file.open("w", encoding="utf-8") as out_file:
        for i, file_path in enumerate(clk_files):
            with file_path.open("r", encoding="utf-8") as in_file:
                end_of_header = False
                for line in in_file:
                    if i==0 and "PGM / RUN BY / DATE" in line:
                        # Add line noting splice
                        out_file.write(line)
                        if len(clk_files) > 1:
                            out_file.write(f"Spliced by TomoSAR v{version[:5]} from {len(clk_files)} files".ljust(60) + "COMMENT".ljust(20) + "\n")
                        continue
                    if "END OF HEADER" in line:
                        end_of_header = True
                        if i != 0:
                            continue
                    if i != 0 and not end_of_header:
                        continue # Skip headers for subsequent files
                    out_file.write(line)
    print("done.")
    
    return merged_file

def splice_dcb(dcb_files: list[Path], output_dir: Path, force: bool = False) -> Path:
    """
    Splice multiple Bias-SINEX files into one, preserving the header and data integrity.
    """
    if len(dcb_files) == 1:
        if dcb_files[0] == output_dir / dcb_files[0].name:
            return dcb_files[0]
    
    dcb_files.sort()
    merged_file, t_start, t_end = _generate_merged_filenames(dcb_files, output_dir=output_dir)
    if not merged_file:
        return None
    if merged_file.exists() and not force:
        print(f"Discovered merged file {local(merged_file)}. Aborting merge of Bias-SINEX files.")
        return merged_file
    print(f"Merging Bias-SINEX files > {local(merged_file)} ...", end=" ", flush=True)
    # Write spliced file
    with merged_file.open("w", encoding="utf-8") as out_file:
        for i, file_path in enumerate(dcb_files):
            with file_path.open("r", encoding="utf-8") as in_file:
                bias_solution = False
                for line in in_file:
                    if i==0 and "%=BIA" in line:
                        # Subsititute end timestamp
                        if i==0:
                            out_file.write(string_sub(line, r'\d{4}:\d{3}:\d{5}', f"{t_end.year:04}:{t_end.timetuple().tm_yday:03}:{3600 * t_end.hour + 60 * t_end.minute + t_end.second:05}", 3))
                            continue
                    if i==0 and "CODE'S rapid IAR phase/code OSB results for day" in line:
                        #  Add line noting splice
                        out_file.write(line)
                        if len(dcb_files) > 1:
                            out_file.write(f"* Spliced by TomoSAR v{version[:5]} from {len(dcb_files)} files\n")
                        continue
                    if "-BIAS/SOLUTION" or "%=ENDBIA" in line:
                        continue
                    if "+BIAS/SOLUTION" in line:
                        bias_solution = True
                        if i != 0:
                            continue
                    if i != 0 and not bias_solution:
                        continue # Skip other blocks for subsequent files
                    out_file.write(line)
        out_file.write("-BIAS/SOLUTION\n")
        out_file.write("%=ENDBIA\n")
    print("done.")
    
    return merged_file

def splice_inx(inx_files: list[Path], output_dir: Path, force: bool = False) -> Path:
    """
    Splice IONEX files into one, preserving the header and data integrity.
    """
    if len(inx_files) == 1:
        if inx_files[0] == output_dir / inx_files[0].name:
            return inx_files[0]
    
    inx_files.sort()
    merged_file, t_start, t_end = _generate_merged_filenames(inx_files, output_dir=output_dir)
    if not merged_file:
        return None
    if merged_file.exists() and not force:
        print(f"Discovered merged file {local(merged_file)}. Aborting merge of INX files.")
        return merged_file
    print(f"Merging INX files > {local(merged_file)} ...", end=" ", flush=True)
    # Write spliced file
    tec_lines = []
    rms_lines = []
    with merged_file.open("w", encoding="utf-8") as out_file:
        for i, file_path in enumerate(inx_files):
            with file_path.open("r", encoding="utf-8") as in_file:
                end_of_header = False
                tec_data = False
                rms_data = False
                for line in in_file:
                    if i==0 and "PGM / RUN BY / DATE" in line:
                        # Add line noting splice
                        out_file.write(line)
                        if len(inx_files) > 1:
                            out_file.write(f"Spliced by TomoSAR v{version[:5]} from {len(inx_files)} files".ljust(60) + "COMMENT".ljust(20) + "\n")
                        continue
                    if i==0 and "EPOCH OF LAST MAP" in line:
                        # Modify last epoch
                        out_file.write(f"{t_end.year:>6}{t_end.month:>6}{t_end.day:>6}{t_end.hour:>6}{t_end.minute:>6}{t_end.second:>6}{' '*24}EPOCH OF LAST MAP\n")
                        continue
                    if not end_of_header:
                        if i == 0:
                            out_file.write(line)
                        else:
                            continue # Skip headers for subsequent files
                    if "END OF HEADER" in line:
                        end_of_header = True
                        if i != 0:
                            continue
                    if "START OF TEC MAP" in line:
                        tec_data = True
                    if "END OF TEC MAP" in line:
                        tec_lines.append(line)
                        tec_data = False
                    if "START OF RMS MAP" in line:
                        rms_data = True
                    if "END OF RMS MAP" in line:
                        rms_data = False
                        rms_lines.append(line)
                    if tec_data:
                        tec_lines.append(line)
                    if rms_data:
                        rms_lines.append(line)
        # Write data blocks
        out_file.writelines(tec_lines)
        out_file.writelines(rms_lines)
        out_file.write(f"{' '*60}END OF FILE\n")
    print("done.")
    
    return merged_file

def merge_ephemeris(files: list[Path], output_dir: Path|str|None = None, force: bool = False) -> tuple[Path|None, Path|None]:
    # Divide files after type
    sp3_files = []
    clk_files = []
    inx_files = []
    for f in files:
        match f.suffix.upper():
            case ".SP3":
                sp3_files.append(f)
            case ".CLK":
                clk_files.append(f)
            case ".INX":
                inx_files.append(f)
    
    merged_sp3 = splice_sp3(sp3_files, output_dir=output_dir, force=force)
    merged_clk = splice_clk(clk_files, output_dir=output_dir, force=force)
    merged_inx = splice_inx(inx_files, output_dir=output_dir, force=force)

    return merged_sp3, merged_clk, merged_inx

def fetch_cod_files(
    start_time: datetime,
    end_time: datetime,
    output_dir: str|Path,
    ignore_files: str|set = set(),
    dry: bool = False,
    max_workers: int = 10,
    max_retries: int = 3
) -> tuple[list[Path], int]:
    """Accesses the GSSC lake at gssc.esa.int to find CODE SP3 ORB files, CLK files, and IONEX files
    covering the time from start_time to end_time. The parameter ignore_files can contain the following strings:
    - "SP3": do not download SP3 ORB files
    - "CLK": do not download CLK files
    - "INX": do not download IONEX files
    
    :return downloaded_files: list of pathlib.Path objects pointing to downloaded files
    :return failed: number of failed downloads"""
    ftp, ftp_user, ftp_pass = prompt_ftp_login('gssc.esa.int', max_attempts=max_retries, anonymous=True)

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure start time is at least 6 hours before start
    # start_time = start_time - timedelta(hours=6)
    start_time = start_time.date()
    # Ensure end time is at least 6 hours after end
    # end_time = end_time + timedelta(hours=6)
    end_time = end_time.date()

    # Collect files to download
    files_to_download = []
    current = start_time
    while current <= end_time:
        gps_week = date_to_gps_week(current)
        doy = current.timetuple().tm_yday
        ftp_path = f"/gnss/products/{gps_week}/"

        # Look for Ephemeris files
        try:
            ftp.cwd(ftp_path)
            files = ftp.nlst()
            if "SP3" not in ignore_files:
                # Find COD MGEX .SP3 file
                target_name = f"COD0MGXFIN_{current.year}{doy}0000_01D_05M_ORB.SP3.gz"
                match = next((f for f in files if f == target_name), None)
                if match:
                    local_path = output_dir / match
                    files_to_download.append((ftp_path, match, local_path))
                else:
                    warn(f"Could not find SP3 orbit file for {current}.")
            
            if "CLK" not in ignore_files:
                # Find COD MGEX .CLK
                target_name = f"COD0MGXFIN_{current.year}{doy}0000_01D_30S_CLK.CLK.gz"
                match = next((f for f in files if f == target_name), None)
                if match:
                    local_path = output_dir / match
                    files_to_download.append((ftp_path, match, local_path))
                else:
                    warn(f"Could not find CLK file for {current} –– {ftp_path}: {current.year}{doy}")

        except Exception as e:
            warn(f"Could not access {ftp_path}: {e}")
        
        # Look for IONEX file
        if "INX" not in ignore_files:
            ftp_path = f"/gnss/products/ionex/{current.year}/{doy}"
            try:
                ftp.cwd(ftp_path)
                files = ftp.nlst()

                # Find COD INX file
                target_name = f"COD0OPSRAP_{current.year}{doy}0000_01D_01H_GIM.INX.gz"
                match = next((f for f in files if f == target_name), None)
                
                if match:
                    local_path = output_dir / match
                    files_to_download.append((ftp_path, match, local_path))
                else:
                    warn(f"Could not find IONEX file for {current} –– {ftp_path}: {current.year}{doy}")
            except Exception as e:
                warn(f"Could not access {ftp_path}: {e}")
            
        current += timedelta(days=1)

    ftp.quit()

    if dry:
        print("Dry-run mode: the following files would be downloaded:")
        for _, filename, _ in files_to_download:
            print(f"  {filename}")
        return


    def download_file(file_info: tuple[str, str, Path]) -> tuple[str, bool]:
        ftp_path, filename, local_path = file_info
        for attempt in range(1, max_retries + 1):
            try:
                ftp = FTP("gssc.esa.int")
                ftp.login(user=ftp_user, passwd=ftp_pass)
                ftp.cwd(ftp_path)
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f"RETR " + filename, f.write)
                ftp.quit()
                return filename, True
            except Exception as e:
                print(f"Attempt {attempt} failed for {filename}: {e}")
                time.sleep(1)
        return filename, False
    
    # Download with progress bar
    downloaded_files = []
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, fi): fi[1] for fi in files_to_download}
        with tqdm(total=len(futures), desc="Downloading files") as pbar:
            for future in as_completed(futures):
                filename, success = future.result()
                if success:
                    tqdm.write(f"Downloaded: {filename}")
                    downloaded_files.append(gunzip(output_dir / filename))
                else:
                    tqdm.write(f"Failed: {filename}")
                    failed += 1
                pbar.update(1)
    
    return downloaded_files, failed

def parse_atx(file_path, antenna_type, radome, mode: str = "glab") -> tuple[list[str], str, bool]:
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

# SWEPOS helpers
def find_station(rover_pos, stations_path: str|Path = None):
    """
    Finds the nearest SWEPOS station to a given lat/lon coordinate with altitude (alt).
    If no path is provided, defaults to 'SWEPOS_koordinatlista.csv' in the project root/config_files.
    """

    # Load the station coordinates
    with resource(stations_path, 'SWEPOS_COORDINATES') as f:
        df = pd.read_csv(f, encoding='utf-8-sig')

    # Define Euclidean distance
    def euclidean_distance(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

    flight_x, flight_y, flight_z = rover_pos

    # Compute distances
    df['Distance'] = df.apply(lambda row: euclidean_distance(
        flight_x, flight_y, flight_z,
        row['SW99_X'], row['SW99_Y'], row['SW99_Z']
    ), axis=1)

    # Find nearest station
    nearest_station = df.loc[df['Distance'].idxmin()]

    return nearest_station

def fetch_swepos_files(
    station_code: str,
    start_time: datetime,
    end_time: datetime,
    output_dir: str|Path,
    min_dur: int = None,
    dry_run: bool = False,
    fetch_nav: bool = False,
    max_workers: int = 10,
    max_retries: int = 3
) -> tuple[list[Path], int]:

    # Log into FTP server
    settings = Settings()
    ftp, ftp_user, ftp_pass = prompt_ftp_login(server="ftp-sweposdata.lm.se", max_attempts=3, user=settings.SWEPOS_USERNAME, pw=settings.SWEPOS_PASSWORD)

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files to download
    files_to_download = []
    current = start_time.replace(minute=0, second=0, microsecond=0)

    file_count = 0
    while current <= end_time or file_count < min_dur:
        file_count += 1
        year = current.strftime("%Y")
        day_of_year = current.strftime("%j")
        hour = current.strftime("%H")
        ftp_path = f"/Rinex3/se_swepos_hourly/{year}/{day_of_year}/{hour}/"

        try:
            ftp.cwd(ftp_path)
            files = ftp.nlst()
            # Find all obs files for this station and hour
            station_files = [f for f in files if f.startswith(station_code) and (f.endswith("MO.crx.gz") or f.endswith("O.rnx.gz"))]
            if fetch_nav:
                station_files.extend([f for f in files if f.startswith(station_code) and (f.endswith("MN.crx.gz") or f.endswith("N.rnx.gz"))])
            # Group by file type (everything after station code and before .gz)
            file_types = set()
            for f in station_files:
                # Example: 0VIN00SWE_S_20252390300_01H_01S_MO.crx.gz
                # Split by '_' and get the part that indicates R or S
                parts = f.split('_')
                if len(parts) < 3:
                    continue
                # The third part is usually 'R' or 'S'
                rs_flag = parts[1]
                # The rest is the file type
                file_type = '_'.join(parts[2:])
                file_types.add((file_type, rs_flag))

            # For each file type, prefer R over S
            for file_type, _ in file_types:
                r_file = f"{station_code}_R_{file_type}"
                s_file = f"{station_code}_S_{file_type}"
                if r_file in station_files:
                    filename = r_file
                elif s_file in station_files:
                    filename = s_file
                else:
                    continue  # Neither available
                local_path = output_dir / filename
                if local_path.exists():
                    continue  # Skip already downloaded
                files_to_download.append((ftp_path, filename, local_path))
        except Exception as e:
            print(f"Could not access {ftp_path}: {e}")

        current += timedelta(hours=1)


    ftp.quit()

    if dry_run:
        print("Dry-run mode: the following files would be downloaded:")
        for _, filename, _ in files_to_download:
            print(f"  {filename}")
        return

    def download_file(file_info: tuple[str, str, Path]) -> tuple[str, bool]:
        ftp_path, filename, local_path = file_info
        for attempt in range(1, max_retries + 1):
            try:
                ftp = FTP("ftp-sweposdata.lm.se")
                ftp.login(user=ftp_user, passwd=ftp_pass)
                ftp.cwd(ftp_path)
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f"RETR " + filename, f.write)
                ftp.quit()
                return filename, True
            except Exception as e:
                print(f"Attempt {attempt} failed for {filename}: {e}")
                time.sleep(1)
        return filename, False

    # Download with progress bar
    failed = 0
    downloaded_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, fi): fi[1] for fi in files_to_download}
        with tqdm(total=len(futures), desc="Downloading SWEPOS files") as pbar:
            for future in as_completed(futures):
                filename, success = future.result()
                if success:
                    tqdm.write(f"Downloaded: {filename}")
                    decompressed_path = gunzip(output_dir / filename)
                    decompressed_path = (output_dir / filename).with_suffix('')  # removes .gz
                    if decompressed_path.suffix == '.crx':
                        rnx_path = crx2rnx(decompressed_path)
                        if not rnx_path.is_file():
                            raise FileNotFoundError(f"Unpacking of Hatanaka compressed .crx file {decompressed_path} unsuccessful")
                        downloaded_files.append(rnx_path)
                    else:
                        downloaded_files.append(decompressed_path)
                else:
                    tqdm.write(f"Failed: {filename}")
                    failed += 1
                pbar.update(1)

    return downloaded_files, failed

def merge_swepos_rinex(files: list[str|Path], output_dir: Path) -> tuple[Path|None, Path|None]:
    obs_files = []
    nav_files = []
    for f in files:
        match f.with_suffix("").name[-1].upper():
            case "O":
                obs_files.append(f)
            case "N":
                nav_files.append(f)

    # Merge rinex files    
    merged_obs = merge_rnx(obs_files, force=True, output_dir=output_dir)
    merged_nav = merge_rnx(nav_files, force=True, output_dir=output_dir)

    return merged_obs, merged_nav

# Read output
def read_rnx2rtkp_out(input: str|Path, processing_frame: str = "ITRF") -> tuple[np.ndarray, float, np.ndarray]:
    """Parses a rnx2rtkp .pos file.
    Returns:
        array with the coordinates,
        array with the GPST corresponding to the coordinates,
        array with the quality conversion corresponding to the coordinates (Q),
        """
    
    if isinstance(input, str):
        lines = input.splitlines()
    elif isinstance(input, Path):
        with open(input, 'r') as f:
            lines = f.readlines()

    # Skip header lines
    data_lines = [line for line in lines if not line.startswith('%')]

    # Parse numeric data
    try:
        data = [list(map(float, line.split())) for line in data_lines]
        data = np.array(data)
    except ValueError:
        raise ValueError(f"Could not parse numeric data from {input}")

    # Constants

    def get_epoch(gps_array: np.ndarray) -> np.ndarray:
        """
        Convert GPS week and seconds-of-week to decimal years (UTC).
        gps_array: np.ndarray of shape (n, 2) -> [gps_week, seconds_of_week]
        Returns: np.ndarray of decimal years
        """
        gps_array = np.asarray(gps_array, dtype=float)
        weeks = gps_array[:, 0]
        seconds = gps_array[:, 1]

        week_dates = gps_week_to_date(weeks)
        utc_datetimes = week_dates + seconds.astype("timedelta64[ms]") - leap_seconds(week_dates)

        # Extract year and compute decimal year
        years = utc_datetimes.astype('datetime64[Y]').astype(int) + 1970
        year_start = np.array([np.datetime64(f"{y}-01-01T00:00:00") for y in years])
        next_year_start = np.array([np.datetime64(f"{y+1}-01-01T00:00:00") for y in years])

        seconds_into_year = (utc_datetimes - year_start).astype('timedelta64[s]').astype(float)
        year_length = (next_year_start - year_start).astype('timedelta64[s]').astype(float)

        decimal_years = years + seconds_into_year / year_length
        return decimal_years
    
    # Auto-detect coordinate columns
    results = {}
    if data.shape[1] >= 5:
        lat = data[:, 2].T                        
        lon = data[:, 3].T
        h = data[:, 4].T
        coords = np.vstack((lon, lat, h))       # LLH coordinates ITRF frame (lon, lat, h)
        results["SD"] = data[:, 7:10].T         # NEU SD
        results["ratio"] = data[:, 14]          # AR ratio
        results["gps_week"] = data[:, 0]        # GPS week
        results["gpst"] = data[:, 1]            # GPST (s) of week
        results["quality"] = data[:, 5]         # Q number
    else:
        raise ValueError(f"Unexpected format in {input}: not enough columns")
    
    # Change to TARGET_FRAME, passing epoch array as 4th coordianate
    st = Settings()
    processing_frame = st.resolve_frame(processing_frame)
    if processing_frame != st.TARGET_FRAME:
        coords = geo_to_ecef(*coords, rf=processing_frame)
        coords = change_rf(processing_frame, st.TARGET_FRAME, *coords, get_epoch(data[:, 0:2]))

        # Convert back to geodetic in TARGET_FRAME
        coords = ecef_to_geo(*coords)

    # Change to projected coordinates
    h = coords[2]
    coords = geo_to_map(lon=coords[0], lat=coords[1])

    results["coordinates"] = np.asarray((*coords, h)) # shape (3, n): Easting, Northing, Height
    
    return results

def read_glab_out(input: str|Path, verbose: bool = False) -> dict:
    x, y, z = [], [], []
    err, epoch = [], []
    x_err, y_err, z_err = [], [], []
    def get_epoch(year, doy, sod) -> float:
        """Returns timestamp as a fractional year epoch"""
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        year_length = (end_of_year - start_of_year).total_seconds()
        seconds_into_year = year + timedelta(days=doy - 1, seconds=sod).total_seconds()
        return year + seconds_into_year / year_length

    if isinstance(input, Path):
        with open(input, 'r') as f:
            for line in f:
                if line.startswith("OUTPUT"):
                    parts = line.split()
                    try:
                        x.append(float(parts[11]))
                        y.append(float(parts[12]))
                        z.append(float(parts[13]))
                        x_err.append(float(parts[17]))
                        y_err.append(float(parts[18]))
                        z_err.append(float(parts[19]))
                        err.append(float(parts[10]))
                        epoch.append(get_epoch(year=int(parts[1]), doy=int(parts[2]), sod=float(parts[3])))
                    except (IndexError, ValueError):
                        continue  # Skip malformed lines
    else:
        lines = input.splitlines()
        for line in lines:
            if line.startswith("OUTPUT"):
                parts = line.split()
                try:
                    x.append(float(parts[11]))
                    y.append(float(parts[12]))
                    z.append(float(parts[13]))
                    x_err.append(float(parts[17]))
                    y_err.append(float(parts[18]))
                    z_err.append(float(parts[19]))
                    err.append(float(parts[10]))
                    epoch.append(get_epoch(year=int(parts[1]), doy=int(parts[2]), sod=float(parts[3])))
                except (IndexError, ValueError):
                    continue  # Skip malformed lines
    
    # Output position values
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Output epochs
    epoch = np.array(epoch)

    # Output formal errors
    err = np.array(err)
    x_err = np.array(x_err)
    y_err = np.array(y_err)
    z_err = np.array(z_err)

    # Convergennce
    conv = (err < 0.002) & (x_err < 0.0013) & (y_err < 0.0013) & (z_err < 0.0013)
    idx = np.argmax(conv)
    # Convergence time and total time
    year = int(epoch[0])
    year_length = (datetime(year+1, 1, 1) - datetime(year, 1, 1)).total_seconds()
    conv_time = timedelta(seconds=round((epoch[idx] - epoch[0])*year_length))
    total_time = timedelta(seconds=round((epoch[-1] - epoch[0])*year_length))
    if not conv.any():
        raise RuntimeError(f"Station PPP failed to converge: {input if isinstance(input, Path) else 'gLAB OUTPUT'} (total runtime: {total_time})")

    st = Settings()
    # Convert to MOCOREF_FRAME
    x_itrf, y_itrf, z_itrf = x, y, z
    x, y, z = change_rf("ITRF2020", st.MOCOREF_FRAME, x, y, z, epoch)

    # Mean position after convergence
    x_mean = np.nanmean(x[conv])
    y_mean = np.nanmean(y[conv])
    z_mean = np.nanmean(z[conv])
    x_itrf = np.nanmean(x_itrf[conv])
    y_itrf = np.nanmean(y_itrf[conv])
    z_itrf = np.nanmean(z_itrf[conv])

    # Residuals
    x_res = x - x_mean
    y_res = y - y_mean
    z_res = z - z_mean

    # Geodetic coordinates
    if verbose or st.VERBOSE:
        print(f"PPP solution converged after {conv_time}, average taken over {total_time - conv_time}")
    
    mean = np.asarray((x_mean, y_mean, z_mean))
    mean_itrf = np.asarray((x_itrf, y_itrf, z_itrf))

    lon, lat, h = ecef_to_geo(*mean, rf=st.MOCOREF_FRAME)
    results = {
        "position": mean,
        "epochs": epoch,
        "epoch": np.nanmedian(epoch),
        "residuals": np.asarray([x_res, y_res, z_res]),
        "convergence_idx": idx,
        "convergence_time": conv_time,
        "convergence_duration": total_time - conv_time,
        "total_duration": total_time,
        "lon": lon,
        "lat": lat,
        "h": h,
        "rotation": ecef_to_enu(lon, lat),
        "itrf_position": mean_itrf,
    }

    return results
  
def detect_convergence_and_mean(x_vals, y_vals, z_vals, x_err, y_err, z_err, err, window_size=100, threshold_percentile=10, verbose: bool = False):
    # Residuals from full-series mean
    x_res = x_vals - np.mean(x_vals)
    y_res = y_vals - np.mean(y_vals)
    z_res = z_vals - np.mean(z_vals)

    # Rolling standard deviation
    x_std = np.array([np.std(x_res[i:i+window_size]) for i in range(len(x_res)-window_size)])
    y_std = np.array([np.std(y_res[i:i+window_size]) for i in range(len(y_res)-window_size)])
    z_std = np.array([np.std(z_res[i:i+window_size]) for i in range(len(z_res)-window_size)])

    combined_std = (x_std + y_std + z_std) / 3
    threshold = np.percentile(combined_std, threshold_percentile)

    # Find first index where std drops below threshold
    convergence_index = np.argmax(combined_std < threshold)

    # Compute mean from stable data
    x_mean = np.mean(x_vals[convergence_index:])
    y_mean = np.mean(y_vals[convergence_index:])
    z_mean = np.mean(z_vals[convergence_index:])

    # New residuals
    x_res = x_vals - x_mean
    y_res = y_vals - y_mean
    z_res = z_vals - z_mean

    # STD
    x_std = np.std(x_res[convergence_index:])
    y_std = np.std(y_res[convergence_index:])
    z_std = np.std(z_res[convergence_index:])

    th = (err < 0.002) & (x_err < 0.0013) & (y_err < 0.0013) & (z_err < 0.0013)
    test_idx = np.argmax(th)
    x_tmean = np.mean(x_vals[convergence_index:])
    y_tmean = np.mean(y_vals[convergence_index:])
    z_tmean = np.mean(z_vals[convergence_index:])


    if verbose:
        # Plot residuals and convergence point
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(x_res, label='X residual')
        axes[0].plot(y_res, label='Y residual')
        axes[0].plot(z_res, label='Z residual')
        axes[0].axvline(convergence_index, color='red', linestyle='--', label='Convergence Point')
        axes[0].axvline(test_idx, color='green', linestyle='--', label='Threshold Idx')
        axes[0].legend()
        axes[0].set_title('ECEF Residuals and Convergence Detection')
        axes[0].set_ylabel('Residual (m)')
        axes[0].grid(True)
        axes[1].plot(x_err, label='X error')
        axes[1].plot(y_err, label='Y error')
        axes[1].plot(z_err, label='Z error')
        axes[1].axvline(convergence_index, color='red', linestyle='--', label='Convergence Point')
        axes[1].axvline(test_idx, color='green', linestyle='--', label='Threshold Idx')
        axes[1].plot(err, label='Formal error')
        axes[1].set_title('Formal Errors')
        axes[1].set_ylabel('Error')
        axes[1].grid(True)
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

        print(f"Mean ECEF position after convergence:\nX: {x_mean:.4f} m\nY: {y_mean:.4f} m\nZ: {z_mean:.4f} m")
        distance = math.sqrt((x_mean - x_tmean)**2 + (y_mean - y_tmean)**2 + (z_mean - z_tmean)**2)
        print(f"Distance between threshold means: {distance} m")

    return (x_mean, y_mean, z_mean), (x_std, y_std, z_std), convergence_index

# Function to read mocoref data from a data file
def generate_mocoref(
        data: str|Path|dict|pd.DataFrame,
        timestamp: datetime|float|str,
        type: str = None,
        output_dir: Path|str|None = None,
        line: int = 1,
        pco_diff: float = -0.079,
        tstart: datetime|None = None,
        tend: datetime|None = None,
        tolerance: float = 0.2,
        generate: bool = True,
        verbose: bool = False,
        output_frame: str = "ITRF"
    ) -> tuple[tuple[float, float, float], Path|None]:
    """Reads mocoref data from a data file. Valid types: CSV, JSON, LLH and mocoref. If not specified attempts to determine file type from file extension.
    The line parameter specifies which line in a CSV file the mocoref data is read from. Optionally generates a mocoref.moco file.
    
    If data is dict or DataFrame instead of Path or string, mocoref data will be read from the dict or DataFrame instead.
    A DataFrame will be interpreted as having the mocoref data in a single line if the line parameter is positive, and as being an LLH log if it is zero.
    
    LLH data read from a file is assumed to lack a header and have identical columns to Reach RS3 output:
    date, time, latitude, longitude, height, Q, satellites, sdn, sde, sdu, sdne, sdeu, sdun, age, AR_ratio.

    If tstart and/or tend are specified the LLH data will be limited to matching timestamps, otherwise the entire log is used. Note however that from within the timestamps used,
    there is extracted the position which gives the LONGEST TOTAL FIX with a tolerance of position variation specified by the tolerance parameter.
    
    The pco_offset parameter allows the user to specify a vertical PCO offset between the receiver used to record mocoref data and the receiver used in drone processing.
    Note: this applies to CSV files ONLY.
    
    Returns:
    - pos: tuple with ECEF coordinates in ITRF2020
    - mocoref_path: path to generated file or None"""

    # Check if data or data file was passed
    data_file = None
    if isinstance(data, dict):
        type = "JSON"
    elif isinstance(data, pd.DataFrame):
        if line == 0:
            type = "LLH"
        else:
            type = "CSV"
    else:
        data_file = Path(data)
        if not data_file.is_file():
            raise FileNotFoundError(f"File {data_file} cannot be found.")
        # Automatically interpret type unless specified
        if type is None:
            if data_file.suffix in (".CSV", ".csv"):
                type = "CSV"
            elif data_file.suffix in (".JSON", ".json"):
                type = "JSON"
            elif data_file.suffix in (".LLH", ".llh"):
                type = "LLH"
            elif data_file.name == "mocoref.moco":
                type = "mocoref"
            else:
                raise RuntimeError("Failed to interpret file type.")
    
    if output_dir:
        output_dir = Path(output_dir)
    elif data_file:
        output_dir = data_file.resolve().parent
    else:
        output_dir = Path.cwd()
    
    type = type.casefold()
    # Validate type
    valid_types = ["csv", "json", "llh", "mocoref"]
    if type not in valid_types:
        raise TypeError(f"Invalid type {type}. Valid types: {valid_types}")
    
    # Get mocoref data 
    settings = Settings()
    match type:
        case "csv":
            # Read data from file
            if data_file:
                data = pd.read_csv(data_file)

            # Validate the presence of mocoref data
            if settings.MOCOREF_LATITUDE not in data.columns:
                raise KeyError(f"{settings.MOCOREF_LATITUDE} key not found in {data_file}")
            if settings.MOCOREF_LONGITUDE not in data.columns:
                raise KeyError(f"{settings.MOCOREF_LONGITUDE} key not found in {data_file}")
            if settings.MOCOREF_HEIGHT not in data.columns:
                raise KeyError(f"{settings.MOCOREF_HEIGHT} key not found in {data_file}")
            if settings.MOCOREF_ANTENNA not in data.columns:
                raise KeyError(f"{settings.MOCOREF_ANTENNA} key not found in {data_file}")
            
            # Validate line parameter
            if not isinstance(line, int) or not line > 0:
                raise TypeError(f"The line parameter must specify a valid positive integer line for this data. Current: {line}")
            if line > len(data):
                raise IndexError(f"The line {line} does not exist in data.")
            
            mocoref_latitude = data[settings.MOCOREF_LATITUDE].iloc[line - 1]
            mocoref_longitude = data[settings.MOCOREF_LONGITUDE].iloc[line - 1]
            mocoref_height = data[settings.MOCOREF_HEIGHT].iloc[line - 1]
            mocoref_antenna = data[settings.MOCOREF_ANTENNA].iloc[line - 1]

            # Modify antenna offset to account for difference between RS3 and CHCI83 PCO:
            mocoref_antenna = mocoref_antenna + pco_diff

        case "json":
            # Read data from file
            if data_file:
                data = json.load(data_file)

            # Validate the presence of mocoref data
            if settings.MOCOREF_LATITUDE not in data:
                raise KeyError(f"{settings.MOCOREF_LATITUDE} key not found in {data_file}")
            if settings.MOCOREF_LONGITUDE not in data:
                raise KeyError(f"{settings.MOCOREF_LONGITUDE} key not found in {data_file}")
            if settings.MOCOREF_HEIGHT not in data:
                raise KeyError(f"{settings.MOCOREF_HEIGHT} key not found in {data_file}")
            if settings.MOCOREF_ANTENNA not in data:
                raise KeyError(f"{settings.MOCOREF_ANTENNA} key not found in {data_file}")
            
            mocoref_latitude = data[settings.MOCOREF_LATITUDE]
            mocoref_longitude = data[settings.MOCOREF_LONGITUDE]
            mocoref_height = data[settings.MOCOREF_HEIGHT]
            mocoref_antenna = data[settings.MOCOREF_ANTENNA]
        case "llh":
            # Read data from file
            if data_file:
                data = pd.read_csv(data_file, sep=r'\s+',  names=['date','time', settings.MOCOREF_LATITUDE, settings.MOCOREF_LONGITUDE,
                                                     settings.MOCOREF_HEIGHT, 'quality', 'satellites', 'sdn', 'sde', 'sdu',
                                                     'sdne', 'sdeu', 'sdun', 'age', 'ar_ratio'])

            # Validate the presence of mocoref data
            if settings.MOCOREF_LATITUDE not in data:
                raise KeyError(f"{settings.MOCOREF_LATITUDE} key not found in {data_file}")
            if settings.MOCOREF_LONGITUDE not in data:
                raise KeyError(f"{settings.MOCOREF_LONGITUDE} key not found in {data_file}")
            if settings.MOCOREF_HEIGHT not in data:
                raise KeyError(f"{settings.MOCOREF_HEIGHT} key not found in {data_file}")
            
            # Extract matching timestamps if a RINEX file is provided:
            if tstart or tend:
                data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%Y/%m/%d %H:%M:%S.%f', utc=True)
                if tstart:
                    data = data[data['timestamp'] >= tstart]
                if tend:
                    data = data[data['timestamp'] <= tend]
            # LLH logs record phase center position
            mocoref_antenna = 0
            
            # If quality is specified limit to Q=1 if present and find most dense cluster of continuous segments
            if 'quality' in data and np.where(data['quality'] == 1):
                data = data.iloc[np.where(data['quality'] == 1)]
                
                # Weight and mean for each continuous segment in ECEF
                idiff = np.diff(data.index)
                split_points = np.where(idiff > 1)[0]
                start = 0
                weights = []
                points = []
                for point in split_points:
                    end = point + 1
                    segment = data.iloc[start:end]
                    weights.append(len(segment))
                    points.append(geo_to_ecef(
                        segment[settings.MOCOREF_LONGITUDE].mean(),
                        segment[settings.MOCOREF_LATITUDE].mean(),
                        segment[settings.MOCOREF_HEIGHT].mean()
                    ))
                    start = end
                segment = data.iloc[start:]
                weights.append(len(segment))
                points.append(geo_to_ecef(
                    segment[settings.MOCOREF_LONGITUDE].mean(),
                    segment[settings.MOCOREF_LATITUDE].mean(),
                    segment[settings.MOCOREF_HEIGHT].mean()
                ))

                # Convert to numpy arrays
                weights = np.array(weights)
                points = np.array(points)

                # Use DBSCAN clustering to find dense regions within the threshold
                db = DBSCAN(eps=tolerance, min_samples=1, metric='euclidean')
                labels = db.fit_predict(points)

                # Find the cluster with maximum total weight
                best_cluster_label = None
                max_weight = 0
                for label in set(labels):
                    cluster_mask = labels == label
                    cluster_weight = weights[cluster_mask].sum()
                    if cluster_weight > max_weight:
                        max_weight = cluster_weight
                        best_cluster_label = label

                # Extract best cluster
                best_mask = labels == best_cluster_label
                points = points[best_mask]
                weights = weights[best_mask]

                # Perform weighted mean
                pos = np.sum(weights[:, np.newaxis] * points, axis=0) / np.sum(weights)
                mocoref_longitude, mocoref_latitude, mocoref_height = ecef_to_geo(*pos)
            else:
                mocoref_latitude = data[settings.MOCOREF_LATITUDE].mean()
                mocoref_longitude = data[settings.MOCOREF_LONGITUDE].mean()
                mocoref_height = data[settings.MOCOREF_HEIGHT].mean()
        case "mocoref":
            generate = False
            if not data_file:
                raise RuntimeError("No mocoref file was specified")
            mocoref_path = data_file
            with open(data_file, 'r') as file:
                lines = file.readlines()
            value = re.compile(r"\d+(?:[\.\,]\d*)?")
            mocoref_antenna = float(value.search(lines[3]).group())
            mocoref_latitude = float(value.search(lines[4]).group())
            mocoref_longitude = float(value.search(lines[5]).group())
            mocoref_height = float(value.search(lines[6]).group())
    
    # Convert to ECEF coordinates
    st = Settings()
    mocoref_pos = geo_to_ecef(mocoref_longitude, mocoref_latitude, mocoref_height + mocoref_antenna, rf=st.MOCOREF_FRAME)

    # Resolve and unify frames
    if isinstance(timestamp, str):
        timestamp = parse_datetime_string(timestamp)
    mocoref_pos = change_rf(st.MOCOREF_FRAME, output_frame, *mocoref_pos, epoch=timestamp)

    if generate or verbose:
        lines = []
        lines.append("Moco reference {CH}\n")
        lines.append("===================\n")
        lines.append("\n")
        lines.append(f"Antenna height [m] {{d}}: {mocoref_antenna}\n")
        lines.append(f"Latitude [deg]     {{d}}: {mocoref_latitude}\n")
        lines.append(f"Longitude [deg]    {{d}}: {mocoref_longitude}\n")
        lines.append(f"Ground [m]         {{d}}: {mocoref_height}\n")
    if generate:
        mocoref_path = output_dir / "mocoref.moco"
        with open(mocoref_path, 'w') as file:
            file.writelines(lines)
    else:
        mocoref_path = None
    if verbose or settings.VERBOSE:
        print(''.join(lines))

    return mocoref_pos, mocoref_path

# Orchestrating functions
def fetch_swepos(
        drone_gnss: str|Path,
        stations_path: str|Path = None,
        max_downloads: int = 10,
        max_retries: int = 3,
        output_dir: str|Path = None,
        min_dur: int  = None,
        dry: bool = False,
        fetch_nav: bool = False,
    ) -> tuple[Path, Path|None]:
    drone_gnss = Path(drone_gnss)
    if output_dir is None:
        output_dir = drone_gnss.resolve().parent
    else:
        output_dir = Path(output_dir)
    
    if drone_gnss.suffix == '.obs':
        temp = False
        obs_file = drone_gnss
    else:
        obs_file, _, _ = ubx2rnx(drone_gnss, nav=False, sbs=False)
        temp = True
    
    with tmp(obs_file, temporary=temp) as obs_path:
        start_utc, end_utc, pos, _ = extract_rnx_info(obs_path)
        stockholm_tz = pytz.timezone('Europe/Stockholm')

        if start_utc and end_utc:
            start_local = start_utc.astimezone(stockholm_tz)
            end_local = end_utc.astimezone(stockholm_tz)
            print(f"Start time: {start_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} / {start_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"End time: {end_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} / {end_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            print("No valid timestamps found in the file.")

        lon, lat, h = ecef_to_geo(*pos)
        if pos:
            print(f"Approximate location: (lat: {lat}, lon: {lon}, h: {h:.3f})")
        else:
            print("No valid position could be extracted from the file.")

        print("", flush=True) # Print empty line and flush buffer

        try:
            nearest_station = find_station(pos, stations_path)
        except FileNotFoundError as e:
            print(f"{e}. The path to the coordinate list can be specified with --stations")
            return
        
        station_code = nearest_station['FilID'] + "00SWE"
        if not nearest_station.empty:
            print("Nearest SWEPOS Station:")
            print(f"Name: {nearest_station['Stationsnamn']} ({station_code})")
            print(f"Distance: {nearest_station['Distance']/1000:.2f} km")
        else:
            raise RuntimeError("Failed to locate nearest station")

        print("Logging into Swepos network ...", flush=True)
        with tmp(output_dir / "tmp", allow_dir=True) as tmp_dir:
            downloaded_files, failed = fetch_swepos_files(
                station_code=station_code,
                start_time=start_utc,
                end_time=end_utc,
                output_dir=tmp_dir,
                max_workers=max_downloads,
                max_retries=max_retries,
                min_dur=min_dur,
                dry_run=dry,
                fetch_nav=fetch_nav
            )
            if failed:
                raise FileNotFoundError("Download from Swepos failed (see above).")

            merged_obs, merged_nav = merge_swepos_rinex(downloaded_files, output_dir=output_dir )

        if not merged_obs.is_file():
            raise FileNotFoundError(f"Generated OBS file: {merged_obs} not found")
        if not merged_nav is None and not merged_nav.is_file():
            raise FileNotFoundError(f"Generated NAV file: {merged_nav} not found")
        
        start_utc, _, etrs89_pos, _ = extract_rnx_info(merged_obs)
        update_rinex_position(merged_obs, etrs89_pos)
        return merged_obs, merged_nav

def rtkp(
        rover_obs: str|Path,
        base_obs: str|Path,
        nav_file: str|Path,
        out_path: str|Path|None = None,
        config_file: str|Path|None = None,
        sbs_file: str|Path|None = None,
        sp3_file: str|Path|None = None,
        precise: bool = False,
        clk_file: str|Path|None = None,
        inx_file: str|Path|None = None,
        atx_file: str|Path|None = None,
        receiver_file: str|Path|None = None,
        elevation_mask: float|None = None,
        mocoref_file: str|Path = None,
        mocoref_type: str|None = None,
        mocoref_line: int = 1,
        download_dir: str|Path|None = None,
        max_downloads: int = 10,
        max_retries: int = 3,
        dry: bool = False,
        retain: bool = False,
        raw: bool = False,
        force_splice: bool = False,
        processing_frame: str = "ITRF"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs RTKP processing on the ROVER OBS relative BASE OBS, and stores position in out_path if not pointing to a folder.
    If if sp3_file (SP3 file) is provided, runs in precise mode (CLK file must be provided if the SP3 file is a pure orbit file).
    If precise is True and no SP3 file is provided matching precise ephemeris data will be downloaded from ESA (number of
    parallel downloads specified by max_downloads and each file is attempted up to max_retries times).
    
    If a mocoref_file is provided the position of the BASE will be read from there (mocoref data can be read from CSV files, JSON
    files, LLH logs or mocoref.moco logs as in tomosar.utils.generate_mocoref). Otherwise the BASE position will be read from the
    BASE OBS header.
    
    If a config file is not provided, the Tomosar internal config will be used.
    
    If dry is True, the files needed to be downloaded will be displayed but no processing will be run (this will have no effect
    if not run in precise mode). If retain is True the downloaded ephmeris data will be placed in the output directory, otherwise
    they will be stored in temporary files.

    The raw parameter can be set to True in order NOT to use internal Tomosar resources.
    
    The output directory is the folder containing the out_path, or the folder pointed to by the out_path.
    
    Returns:
    - A Nx3 array with the coordinates (X, Y, Z)
    - A Nx1 array with GPS time (seconds) for the coordinates
    - A Nx1 array with the quality conversion (Q) for the coordinates"""

    rover_obs = Path(rover_obs)
    if not rover_obs.is_file():
        raise FileNotFoundError(f"Rover OBS file not found: {rover_obs}")

    base_obs = Path(base_obs)
    if not base_obs.is_file():
        raise FileNotFoundError(f"Base OBS file not found: {base_obs}")
    
    nav_file = Path(nav_file)
    if not nav_file.is_file():
        raise FileNotFoundError(f"NAV file not found: {nav_file}")

    # Check which ephemeris files were provided
    eph_files = set()
    if sp3_file:
        sp3_file = Path(sp3_file)
        if sp3_file.is_file():
            eph_files.add("SP3")
            # If no CLK file is provided: assume SP3 file contains clock corrections as well
            if not clk_file:
                eph_files.add("CLK") 
        else:
            warn(f"User provided SP3 file {sp3_file} not found, ignoring")
            sp3_file = None
    if clk_file:
        clk_file = Path(clk_file)
        if clk_file.is_file():
            eph_files.add("CLK")
        else:
            warn(f"User provided CLK file {clk_file} not found, ignoring")
            clk_file = None
    if inx_file:
        inx_file = Path(inx_file)
        if inx_file.is_file():
            eph_files.add("INX")
        else:
            warn(f"User provided IONEX file {inx_file} not found, ignoring")
            inx_file = None

    if out_path:
        out_path = Path(out_path)
        if out_path.is_dir():
            output_dir = out_path
            out_path = None
        else:
            output_dir = out_path.resolve().parent
            output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = base_obs.resolve().parent
    if download_dir:
        output_dir = Path(download_dir)

    antenna_type, radome = _ant_type(base_obs)
    if not raw:    
        print(f"Detected base antenna type: {antenna_type} {radome}")
        with resource(None, "SATELLITES") as atx:
            with resource(None, "RECEIVER", antenna=antenna_type, radome=radome) as receiver:
                if receiver is None:
                    receiver_file = atx
                else:
                    receiver_file = receiver
                
                constellations, freqs, fallback = parse_atx(receiver_file, antenna_type=antenna_type, radome=radome, mode="rnx2rtkp")
                if fallback:
                    print("Defaulted to NONE radome")
                if constellations:
                    print(f"Using callibrated constellations: {','.join(constellations)}")
                    match freqs:
                        case '1':
                            print("With frequencies: L1")
                        case '2':
                            print("With frequencies: L1+L2")
                        case '3':
                            print("With frequencies: L1+L2+L5")
                else:
                    warn("No callibration data available. Using all available constellations and frequencies.")
    
    print(f"Running {'raw ' if raw else ''}RTKP post processing {'in precise mode ' if precise else 'with broadcast data '}...\n   Rover: {local(rover_obs)}\n   Base: {local(base_obs)}\n   Nav: {local(nav_file)}\n{f'   SP3: {local(sp3_file)}\n' if sp3_file else ''}{f'   CLK: {local(clk_file)}\n' if clk_file else ''}{f'   INX: {local(inx_file)}\n' if inx_file else ''}{f'-->Out: {local(out_path)}' if out_path else ''}", flush=True)
    with resource(config_file, "RTKP_CONFIG", antenna=antenna_type, radome=radome, satellites=atx_file, receiver_file=receiver_file) as config:
        with tmp(output_dir / "tmp", allow_dir=True) as tmp_dir:
            if raw:
                modify_config(config, raw=True)
                constellations = []
                freqs = ""
            if sp3_file or precise:
                # Modify to precise pos1-eph mode
                if not eph_files == {"SP3", "CLK", "INX"}:
                    start_utc, end_utc, _, _ = extract_rnx_info(base_obs)
                    downloaded_files, failed = fetch_cod_files(start_time=start_utc, end_time=end_utc, output_dir=tmp_dir, ignore_files=eph_files, max_workers=max_downloads, max_retries=max_retries, dry=dry)
                    if failed:
                        warn(f"Failed to download {failed} files from GSSC lake (see above).")
                    
                    merged_sp3_file, merged_clk_file, merged_inx_file = merge_ephemeris(downloaded_files, output_dir=output_dir if retain else tmp_dir, force=force_splice)
                    if not sp3_file:
                        sp3_file = merged_sp3_file
                    if not clk_file:
                        clk_file = merged_clk_file
                    if not inx_file:
                        inx_file = merged_inx_file
                modify_config(config, precise=True, ionofile=inx_file)
            if dry:
                return
            try:
                out = rnx2rtkp(
                    rover_obs=rover_obs,
                    base_obs=base_obs,
                    nav_file=nav_file,
                    out_path=out_path,
                    config_file=config,
                    sbs_file=sbs_file,
                    sp3_file=sp3_file,
                    clk_file=clk_file,
                    elevation_mask=elevation_mask,
                    mocoref_file=mocoref_file,
                    mocoref_type=mocoref_type,
                    mocoref_line=mocoref_line,
                    constellations=constellations,
                    freqs=freqs,
                    antenna_type=antenna_type,
                    radome=radome,
                    processing_frame=processing_frame
                )
            except RuntimeError:
                # Try without Explorer specific options
                modify_config(config, standard=True)
                out = rnx2rtkp(
                    rover_obs=rover_obs,
                    base_obs=base_obs,
                    nav_file=nav_file,
                    out_path=out_path,
                    config_file=config,
                    sbs_file=sbs_file,
                    sp3_file=sp3_file,
                    clk_file=clk_file,
                    elevation_mask=elevation_mask,
                    mocoref_file=mocoref_file,
                    mocoref_type=mocoref_type,
                    mocoref_line=mocoref_line,
                    constellations=constellations,
                    freqs=freqs,
                    antenna_type=antenna_type,
                    radome=radome,
                    processing_frame=processing_frame
                )

    if out_path:
        if out_path.is_file():
            out = out_path
        else:
            raise FileNotFoundError(f"Could not find generated .pos file: {out_path}")

    results = read_rnx2rtkp_out(out, processing_frame=processing_frame)
    quality_conversion = np.sum(results["quality"] == 1) / len(results["quality"]) * 100
    print(f"Quality conversion: Q1 = {quality_conversion:.2f} %")
    return results
   
def station_ppp(
        obs_path: str|Path,
        navglo_path: str|Path|None = None,
        atx_path: str|Path|None = None,
        antrec_path: str|Path|None = None,
        sp3_file: str|Path|None = None,
        clk_file: str|Path|None = None,
        inx_file: str|Path|None = None,
        download_dir: str|Path|None = None,
        max_downloads: int = 10,
        max_retries: int = 3,
        out_path: str|Path|None = None,
        elevation_mask: float|None = None,
        header: bool = True,
        dry: bool = False,
        retain: bool = False,
        make_mocoref: bool = True,
        force_splice: bool = False,
) -> dict:
    """Runs static PPP on a base observation file by first downloading matching precise ephemeris files from gssc.esa.int.
    Input parameters:
    - navglo_path: file containing navigation data for GLONASS (can be a merged/general navigation file)
    - atx_path: file containing absolute calibration data for antennas
    - antrec_path: file containing absolute calibration data for the base antenna (overrides atx_path for base)
    - sp3_file: full or orbit SP3 file
    - clk_file: clock file complementing orbit SP3 file
    - max_downloads: number of parallel downloads that will be attempted
    - max_retries: number of times a file download is attempted before failing
    - out_path: file where output is stored (if None it is not stored)
    - header: modify the rinex header with the new position
    - dry: only show which files would be downloaded
    - retain: retains downloaded ephemeris data after finishing
    - mocoref: generate mocoref.moco file with position
    
    Returns dict with keys:
    - 'position': PPP processing determined position (ECEF in MOCOREF_FRAME)
    - 'itrf_position': PPP processing determined position (ECEF in ITRF2020)
    - 'epochs': epochs as decimal year
    - 'epoch': median of epochs
    - 'residuals': residuals from determined positon for all epochs with a solution
    - 'convergence_idx': the index for which the solution converged
    - 'convergence_time': timedelta object with time passed before convergence
    - 'convergence_duration': timedelta object with time passed after convergence
    - 'total_duration': timedelta object with total time passed with a solution
    - 'lon': longitudinal coordinate of position (MOCOREF_FRAME)
    - 'lat': latitudinal coordiante of position (MOCOREF_FRAME)
    - 'h': ellipsoidal height of position (MOCOREF_FRAME)
    - 'rotation': ecef_to_enu rotation matrix for position
    - 'header_position': the approximate position specified in the RINEX OBS header
    - 'sp3': path to .SP3 file (if retained, else None)
    - 'clk': path to .CLK file (if retained, else None)
    - 'inx': path to .INX file (if retained, else None)
    - 'mocoref_file': path pointing to mocoref.moco file (if make_mocoref, else None)"""

    obs_path = Path(obs_path)
    if not obs_path.is_file():
        raise FileNotFoundError(f"Rinex observation file not found: {obs_path}")

    # Check which ephemeris files were provided
    eph_files = set()
    if sp3_file:
        sp3_file = Path(sp3_file)
        if sp3_file.is_file():
            eph_files.add("SP3")
            # If no CLK file is provided: assume SP3 file contains clock corrections as well
            if not clk_file:
                eph_files.add("CLK") 
        else:
            warn(f"User provided SP3 file {sp3_file} not found, ignoring")
            sp3_file = None
    if clk_file:
        clk_file = Path(clk_file)
        if clk_file.is_file():
            eph_files.add("CLK")
        else:
            warn(f"User provided CLK file {clk_file} not found, ignoring")
            clk_file = None
    if inx_file:
        inx_file = Path(inx_file)
        if inx_file.is_file():
            eph_files.add("INX")
        else:
            warn(f"User provided IONEX file {inx_file} not found, ignoring")
            inx_file = None
    
    if out_path:
        out_path = Path(out_path)
        if out_path.is_dir():
            output_dir = out_path
            out_path = None
        else:
            output_dir = out_path.resolve().parent
            output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = obs_path.resolve().parent
    if download_dir:
        output_dir = Path(download_dir)

    print(f"Running station PPP ...\n{f'   Station: {local(obs_path)}\n'}{f'   Navglo: {local(navglo_path)}\n' if navglo_path else ''}{f'   SP3: {local(sp3_file)}\n' if sp3_file else ''}{f'   CLK: {local(clk_file)}\n' if clk_file else ''}{f'   INX: {local(inx_file)}\n' if inx_file else ''}{f'   ATX: {local(atx_path)}\n' if atx_path else ''}{f'   Receiver: {local(antrec_path)}\n' if antrec_path else ''}{f'-->Out: {local(out_path)}' if out_path else ''}", flush=True)
    start_utc, end_utc, approx_pos, antenna_delta = extract_rnx_info(obs_path)
    with tmp(output_dir / "tmp", allow_dir=True) as tmp_dir:
        if not eph_files == {"SP3", "CLK", "INX"}:
            downloaded_files, failed = fetch_cod_files(start_time=start_utc, end_time=end_utc, output_dir=tmp_dir, ignore_files=eph_files, max_workers=max_downloads, max_retries=max_retries, dry=dry)
            if failed:
                warn("Download of precise ephemeris and clock data from ESA failed.")
            
            merged_sp3_file, merged_clk_file, merged_inx_file = merge_ephemeris(downloaded_files, output_dir=output_dir if retain else tmp_dir, force=force_splice)
            if not sp3_file:
                sp3_file = merged_sp3_file
            if not clk_file:
                clk_file = merged_clk_file
            if not inx_file:
                inx_file = merged_inx_file

        # Run PPP command
        out = ""
        print()
        out = glab_ppp(
            obs_file=obs_path,
            sp3_file=sp3_file,
            clk_file=clk_file,
            inx_file=inx_file,
            out_path=out_path,
            navglo_file=navglo_path,
            atx_file=atx_path,
            antrec_file=antrec_path,
            elevation_mask=elevation_mask
        )
   
    if out_path and not out_path.is_file():
        raise FileNotFoundError(f"Cannot find generated out file: {out_path}")

    # Extract position
    results = read_glab_out(out, verbose=True)
    results["header_position"] = np.asarray(approx_pos)
    results["sp3"] = sp3_file if sp3_file.is_file() else None
    results["clk"] = clk_file if clk_file.is_file() else None
    results["inx"] = inx_file if inx_file.is_file() else None

    if header:
        # Compare against header
        diff = results["rotation"] @ (results["postion"] - approx_pos)
        distance = math.sqrt((diff**2).sum())
        print(f"Header position shifted: {distance:.3} m (E: {diff[0]:.3} m, N: {diff[1]:.3} m, U: {diff[2]:.3} m)")
        update_rinex_position(obs_path, results["position"])
        
    if make_mocoref:
        settings = Settings()
        mocoref = {
            settings.MOCOREF_LATITUDE: results["lat"],
            settings.MOCOREF_LONGITUDE: results["lon"],
            settings.MOCOREF_HEIGHT: results["h"],
            settings.MOCOREF_ANTENNA: 0.
        }
        _, results["mocoref_file"] = generate_mocoref(mocoref, timestamp=results['epoch'], generate=True)
    else:
        results["mocoref_file"] = None

    return results
    
def reachz2rnx(archive: Path|str, reference_date: datetime|None = None, output_dir: str|Path|None = None, rnx_file: Path|str|None = None, nav: bool = False, verbose: bool = False) -> tuple[dict[Path, dict], tuple[Path|None, Path|None, Path|None]]:
    """Extracts a Reach .zip archive to produce:
    - A RINEX OBS file for a single site
    - A mocoref.moco for the OBS file
    - A RINEX NAV file (optional)
    
    Optionally takes a RINEX OBS file as input to extract from the archive the OBS file which has the greatest overlap with the
    input RINEX file. Otherwise extracts the longest segment. The reference_date parameter is used to get the correct GPS week for
    the RTCM3 file if this cannot be parsed from the archive name (if a RINEX OBS is provided as reference, the reference_date
    is set from this OBS if parsing from the filename fails and it is not specified by user).
    
    Returns:
    - Dict indexed by paths to files produced:
        - OBS PATH: dict with keys:
            - APPROX POS XYZ
            - TIME OF FIRST OBS
            - TIME OF LAST OBS
        - MOCOREF PATH: mocoref position (lat, lon, h)
        - NAV PATH: True (exists only if nav file was extracted)
    - Tuple of paths: (OBS PATH, MOCOREF PATH, NAV PATH or None)"""

    archive = Path(archive)

    # Determine base name for extracted OBS and NAV files
    base_name = archive.with_suffix("").name

    # Determine reference timestamp from basename if not provided
    if not reference_date:
        if dt := re.search(r'\d{14}', base_name):
            reference_date = datetime.strptime(dt.group(), '%Y%m%d%H%M%S')
        elif rnx_file:
            reference_date, _, _ = extract_rnx_info(rnx_file)
        else:
            raise ValueError(f"Could not determine a reference timestamp from the archive name ({archive.name}). Please provide it explicitly.")

    # Determine output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = archive.resolve().parent

    obs_file = output_dir / (base_name + ".obs")
    print(f"Converting {local(archive)} ...\n-->{local(obs_file)}\n{f'-->{local(obs_file.with_suffix(".nav"))}\n' if nav else ''}-->{local(output_dir / "mocoref.moco")}", flush=True)
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        def extract_to(source: Path, destination: Path, final_destination: Path|None = None) -> None:
            if not final_destination:
                final_destination = destination
            if verbose or Settings().VERBOSE:
                print(f"Extracting {archive}/{source} to {final_destination}")
            with zip_ref.open(source) as source_file:
                with open(destination, 'wb') as target_file:
                    target_file.write(source_file.read())
        
        all_files = zip_ref.namelist()
        rtcm3_file = False
        llh_file = False
        nav_file = False
        for file in all_files:
            # Look for files
            if re.search(r'\.(rtcm3|RTCM3)$', file):
                rtcm3_file = file
            if re.search(r'\.(llh|LLH)$', file):
                llh_file = file
            if re.search(r'\.\d{2}(P|p)$', file) and nav:
                nav_file = file

        # Extract RINEX OBS
        if rtcm3_file:
            with tmp(output_dir / "rtcm3.tmp") as rtcm3_tmp:
                extract_to(rtcm3_file, rtcm3_tmp, final_destination=obs_file)
                if rnx_file:
                    start, end, _, _ = extract_rnx_info(rnx_file)
                else:
                    start, end = None, None
                obs_data, _, _ = reach2rnx(rtcm_file=rtcm3_tmp, reference_date=reference_date, tstart=start, tend=end, obs_file=obs_file, verbose=verbose)
            # Verify success
            if not obs_file.is_file():
                raise FileNotFoundError(f"No RINEX OBS could be extracted from archive RTCM3 file: {rtcm3_file}")
        else:
            raise FileNotFoundError(f"No RTCM3 file found in archive: {archive}")
        
        # Extract mocoref.moco
        if llh_file:
            # Update start and end timestamps to match the OBS file
            start = obs_data[obs_file]["TIME OF FIRST OBS"]
            end = obs_data[obs_file]["TIME OF LAST OBS"]
            epoch = start + (end - start)/2

            # Extract mocoref.moco file
            with tmp(output_dir / "llh.tmp") as llh_tmp:
                extract_to(llh_file, llh_tmp, final_destination=llh_tmp.with_name("mocoref.moco"))
                mocoref_data, mocoref_file = generate_mocoref(llh_tmp, timestamp=epoch, type="LLH", generate=True, tstart=start, tend=end)
            
            # Verify success and add to obs_data
            if mocoref_file.is_file():
                obs_data[mocoref_file] = mocoref_data
            else:
                warn(f"No mocoref.moco file could be extracted from archive LLH log: {llh_file}")
        else:
            warn(f"No LLH log found in archive: {archive}")

        # Verify nav data
        if nav:
            if nav_file:
                extract_to(nav_file, obs_file.with_suffix(".nav"))
                nav_file = obs_file.with_suffix(".nav")
                
                # Add to obs_data
                obs_data[nav_file] = True
            else:
                warn(f"No NAV data found in archive: {archive}")
                nav_file = None
        else:
            nav_file = None
        
    return obs_data, (obs_file, mocoref_file, nav_file)
        