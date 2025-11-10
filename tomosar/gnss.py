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

from .utils import prompt_ftp_login, gunzip, ecef2enu, warn, generate_mocoref
from .binaries import crx2rnx, merge_rnx, merge_eph, ubx2rnx, reach2rnx, rnx2rtkp, ppp, resource, local, tmp, _ant_type, _parse_atx
from .config import Settings
from .transformers import ecef_to_geo

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

def find_station(rover_pos, stations_path: str|Path = None):
    """
    Finds the nearest SWEPOS station to a given lat/lon coordinate with altitude (alt).
    If no path is provided, defaults to 'SWEPOS_koordinatlista.csv' in the project root/config_files.
    """

    # Load the station coordinates
    with resource(stations_path, 'SWEPOS_COORDINATES') as f:
        df = pd.read_csv(f, encoding='utf-8-sig')

    # Define Euclidean distance
    def euclidean_distance(x1, y1, z1, x2, y2, z2):
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
):

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

    def download_file(file_info):
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
                else:
                    tqdm.write(f"Failed: {filename}")
                    failed += 1
                pbar.update(1)

    return failed

def merge_swepos_rinex(data_dir: str|Path) -> tuple[Path|None, Path|None]:
    data_path = Path(data_dir)
    obs_files = sorted(data_path.glob("*O.rnx"))
    nav_files = sorted(data_path.glob("*N.rnx"))

    # Merge rinex files    
    if obs_files:
        merged_obs = merge_rnx(obs_files, force=True)
        # Move outside of tmp dir
        merged_obs = shutil.move(merged_obs, merged_obs.parent.parent)
    else:
        merged_obs = None

    if nav_files:
        merged_nav = merge_rnx(nav_files, force=True)
        # Move outside of tmp dir
        merged_nav = shutil.move(merged_nav, merged_nav.parent.parent)
    else:
        merged_nav = None

    return merged_obs, merged_nav

def merge_ephemeris(data_dir: str|Path) -> tuple[Path|None, Path|None]:
    data_path = Path(data_dir)
    eph_files = sorted(data_path.glob("*.SP3"))
    eph_files.extend(sorted(data_path.glob("*.CLK")))

    if eph_files:
        merged_sp3, merged_clk = merge_eph(eph_files, force=True)

    return merged_sp3, merged_clk

def fetch_sp3_clk(
    start_time: datetime,
    end_time: datetime,
    output_dir: str|Path,
    dry: bool = False,
    max_workers: int = 10,
    max_retries: int = 3
) -> int:
    ftp, ftp_user, ftp_pass = prompt_ftp_login('gssc.esa.int', max_attempts=max_retries, anonymous=True)

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure start time is at least 6 hours before start
    start_time = start_time #- timedelta(hours=6)
    start_time = start_time.date()
    # Ensure end time is at least 6 hours after end
    end_time = end_time #+ timedelta(hours=6)
    end_time = end_time.date()

    # Collect files to download
    files_to_download = []
    current = start_time
    while current <= end_time:
        gps_week = date_to_gps_week(current)
        doy = current.timetuple().tm_yday
        ftp_path = f"/gnss/products/{gps_week}/"

        try:
            ftp.cwd(ftp_path)
            files = ftp.nlst()
            # Find MGEX .sp3 file for the correct day
            target_name = f"COD0MGXFIN_{current.year}{doy}0000_01D_05M_ORB.SP3.gz"
            match = next((f for f in files if f == target_name), None)

            if not match:
                target_name = f"GFZ0MGXRAP_{current.year}{doy}0000_01D_05M_ORB.SP3.gz"
                match = next((f for f in files if f == target_name), None)
                if not match:
                    target_name = f"IAC0MGXFIN_{current.year}{doy}0000_01D_05M_ORB.SP3.gz"
                    match = next((f for f in files if f == target_name), None)

            if match:
                local_path = output_dir / match
                files_to_download.append((ftp_path, match, local_path))
            else:
                print(f"Could not find .sp3 file for {current}.")
            
            # Find MGEX .clk file for the correct day
            target_name = f"COD0MGXFIN_{current.year}{doy}0000_01D_30S_CLK.CLK.gz"
            match = next((f for f in files if f == target_name), None)

            if not match:
                target_name = f"GFZ0MGXRAP_{current.year}{doy}0000_01D_30S_CLK.CLK.gz"
                match = next((f for f in files if f == target_name), None)
                if not match:
                    target_name = f"IAC0MGXFIN_{current.year}{doy}0000_01D_30S_CLK.CLK.gz"
                    match = next((f for f in files if f == target_name), None)
                    
            if match:
                local_path = output_dir / match
                files_to_download.append((ftp_path, match, local_path))
            else:
                print(f"Could not find .clk file for {current} –– {ftp_path}: {current.year}{doy}")

            current += timedelta(days=1)
        except Exception as e:
            print(f"Could not access {ftp_path}: {e}")

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
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, fi): fi[1] for fi in files_to_download}
        with tqdm(total=len(futures), desc="Downloading files") as pbar:
            for future in as_completed(futures):
                filename, success = future.result()
                if success:
                    tqdm.write(f"Downloaded: {filename}")
                    gunzip(output_dir / filename)
                else:
                    tqdm.write(f"Failed: {filename}")
                    failed += 1
                pbar.update(1)
    
    return failed

def date_to_gps_week(input_date: datetime) -> int:
    gps_start = date(1980, 1, 6)  # GPS epoch
    delta = input_date - gps_start
    gps_week = delta.days // 7
    return gps_week

def read_rnx2rtkp_out(input: str|Path) -> tuple[np.ndarray, float, np.ndarray]:
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
    

    # Auto-detect coordinate columns (assumes columns 3–5 are E/N/U or X/Y/Z)
    if data.shape[1] >= 5:
        coords = data[:, 2:5]
        gpst = data[:, 1]
        q = data[:, 5]
    else:
        raise ValueError(f"Unexpected format in {input}: not enough columns")
    
    return coords, gpst, q

def read_glab_out(input: str|Path, verbose: bool = False) -> tuple[np.ndarray,
                                                                       np.ndarray,
                                                                       int,
                                                                       np.ndarray,
                                                                       np.ndarray
                                                                    ]:
    x, y, z = [], [], []
    err, ts = [], []
    x_err, y_err, z_err = [], [], []
    def get_timestamp(year, doy, sod) -> datetime:
        # Start from Jan 1 of the given year
        base_date = datetime(year, 1, 1)
        # Add (DoY - 1) days and seconds of day
        full_datetime = base_date + timedelta(days=doy - 1, seconds=sod)
        return full_datetime

    diff_n, diff_e, diff_u = [], [], []
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
                        diff_n.append(float(parts[23]))
                        diff_e.append(float(parts[24]))
                        diff_u.append(float(parts[25]))
                        err.append(float(parts[10]))
                        ts.append(get_timestamp(year=int(parts[1]), doy=int(parts[2]), sod=float(parts[3])))
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
                    diff_n.append(float(parts[23]))
                    diff_e.append(float(parts[24]))
                    diff_u.append(float(parts[25]))
                    err.append(float(parts[10]))
                    ts.append(get_timestamp(year=int(parts[1]), doy=int(parts[2]), sod=float(parts[3])))
                except (IndexError, ValueError):
                    continue  # Skip malformed lines
    
# Output position values
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Output formal errors
    err = np.array(err)
    x_err = np.array(x_err)
    y_err = np.array(y_err)
    z_err = np.array(z_err)

    # Convergennce
    conv = (err < 0.002) & (x_err < 0.0013) & (y_err < 0.0013) & (z_err < 0.0013)
    idx = np.argmax(conv)
    # Convergence time and total time
    conv_time = ts[idx] - ts[0]
    total_time = ts[-1] - ts[0]
    if not conv.any():
        raise RuntimeError(f"Station PPP failed to converge: {input if isinstance(input, Path) else 'gLAB OUTPUT'} (total runtime: {total_time})")

    # Mean position after convergence
    x_mean = np.nanmean(x[conv])
    y_mean = np.nanmean(y[conv])
    z_mean = np.nanmean(z[conv])

    # STD after convergence
    # x_std = x[conv].std()
    # y_std = y[conv].std()
    # z_std = z[conv].std()

    # Residuals
    x_res = x - x_mean
    y_res = y - y_mean
    z_res = z - z_mean

    # Geodetic coordinates
    lon, lat, h = ecef_to_geo.transform(x_mean, y_mean, z_mean)
    if verbose or Settings().VERBOSE:
        print(f"PPP solution converged after {conv_time}, average taken over {total_time - conv_time}")
        print(f"Position: lat={lat}, lon={lon}, h={h:.3f}")

    rotation = ecef2enu(lat, lon)
    
    mean = (x_mean, y_mean, z_mean)
    diff = (diff_e, diff_n, diff_u)

    return mean, rotation,  idx, diff, np.asarray((x_res, y_res, z_res))

def modify_config(config_path: Path, standard: bool = False, precise: bool = False) -> None:
    """
    Modifies an existing RTKLIB config file to enable precise ephemeris mode
    and sets the paths to SP3 and CLK files.

    Parameters:
    - config_path: Path to the existing config file
    - sp3_path: Path to the SP3 file
    - clk_path: Path to the CLK file
    - output_path: Path to save the modified config file
    """

    if not (standard or precise):
        warn("Running modify_config with standard=False and precise=False has no effect.")
        return
    
    # Read the original config file
    with open(config_path, 'r') as file:
        lines = file.readlines()

    if precise:
        # Flag to check if the required field is found
        sateph_found = False

    # Modify the relevant lines
    for i, line in enumerate(lines):
        if precise:
            # Modify required field if found
            if line.strip().startswith('pos1-sateph'):
                lines[i] = 'pos1-sateph        =1\n'
                sateph_found = True
        if standard:
            # Remove explorer specific fields
            if "# This requires the Explorer version of RTKLIB" in line:
                lines[i] = ''

    if precise:
        # Append missing fields if not found
        if not sateph_found:
            lines.append('pos1-sateph =1\n')

    # Write the modified config to the output file
    with open(config_path, 'w') as file:
        file.writelines(lines)

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
        elevation_mask: float|None = None,
        mocoref_file: str|Path = None,
        mocoref_type: str|None = None,
        mocoref_line: int = 1,
        max_downloads: int = 10,
        max_retries: int = 3,
        dry: bool = False,
        retain: bool = False
) -> None:
    """Calls rnx2rtkp and reads the .pos file"""

    rover_obs = Path(rover_obs)
    if not rover_obs.is_file():
        raise FileNotFoundError(f"Rover OBS file not found: {rover_obs}")

    base_obs = Path(base_obs)
    if not base_obs.is_file():
        raise FileNotFoundError(f"Base OBS file not found: {base_obs}")
    
    nav_file = Path(nav_file)
    if not nav_file.is_file():
        raise FileNotFoundError(f"NAV file not found: {nav_file}")


    if out_path:
        out_path = Path(out_path)
        if out_path.is_dir():
            output_dir = out_path
            out_path = None
        else:
            output_dir = out_path.parent
            output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = rover_obs.parent

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
    
    print(f"Running RTKP post processing ...\n  Rover: {local(rover_obs)}\n  Base: {local(base_obs)}\n  Nav: {local(nav_file)}\n-->Out: {local(out_path)}", flush=True)
    with resource(config_file, "RTKP_CONFIG", antenna=antenna_type, radome=radome) as config:
        with tmp(output_dir / "tmp", allow_dir=True) as tmp_dir:
            if sp3_file or precise:
                # Modify to precise pos1-eph mode
                modify_config(config, precise=True)
                if not sp3_file:
                    start_utc, end_utc, _, _ = extract_rnx_info(base_obs)
                    if not sp3_file:
                        failed = fetch_sp3_clk(start_time=start_utc, end_time=end_utc, output_dir=tmp_dir, max_workers=max_downloads, max_retries=max_retries, dry=dry)
                        if failed:
                            raise FileNotFoundError("Download of precise ephemeris and clock data from ESA failed.")
                        
                        sp3_file, clk_file = merge_ephemeris(tmp_dir)
                        if retain:
                            # Move files out of temporary directory
                            sp3_file = shutil.move(sp3_file, output_dir)
                            clk_file = shutil.move(clk_file, output_dir)
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
                    radome=radome
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
                    mocoref_line=mocoref_line
                )

    if out_path:
        if out_path.is_file():
            out = out_path
        else:
            raise FileNotFoundError(f"Could not find generated .pos file: {out_path}")

    coords, gpst, q = read_rnx2rtkp_out(out)
    quality_conversion = np.sum(q == 1) / len(q) * 100
    print(f"Quality conversion: Q1 = {quality_conversion:.2f} %")
    return coords, gpst, q
    




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

def etrs89_to_wgs84_ecef(X_etrs89, Y_etrs89, Z_etrs89, utc_datetime) -> np.ndarray:
    """Converts ETRS89 ECEF (assumed ETRF2000) coordinates to ITRF208 using time-dependent Helmert transformation."""

    # Convert datetime to fractional year
    year = utc_datetime.year
    start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_of_year = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    year_length = (end_of_year - start_of_year).total_seconds()
    seconds_into_year = (utc_datetime - start_of_year).total_seconds()
    epoch_target = year + seconds_into_year / year_length

    # Reference epoch for ETRF2000 to ITRF2008
    t_ref = 2000.0
    delta_t = epoch_target - t_ref

    # Helmert parameters: ETRF2000 → ITRF2008
    dX = -(0.0521 + 0.0001 * delta_t)
    dY = -(0.0493 + 0.0001 * delta_t)
    dZ = 0.0585 + 0.0018 * delta_t
    rX_arcsec = -(0.000891 + 0.000081 * delta_t)
    rY_arcsec = -(0.005390 + 0.000490 * delta_t)
    rZ_arcsec = 0.008712 + 0.000792 * delta_t
    rX = np.deg2rad(rX_arcsec / 3600)
    rY = np.deg2rad(rY_arcsec / 3600)
    rZ = np.deg2rad(rZ_arcsec / 3600)
    s_ppm = 0.00134 + 0.00008 * delta_t
    s = s_ppm * 1e-6

    # Apply transformation: ETRF2000 → ITRF2008
    X = np.array([X_etrs89, Y_etrs89, Z_etrs89])
    R = np.array([
        [1, -rZ, rY],
        [rZ, 1, -rX],
        [-rY, rX, 1]
    ])
    X_itrf2008 = (1 + s) * R @ X + np.array([dX, dY, dZ])

    return X_itrf2008

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
        output_dir = drone_gnss.parent
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

        lon, lat, h = ecef_to_geo.transform(*pos)
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
                print("Failed to locate nearest station.")
                return

        print("Logging into Swepos network ...", flush=True)
        with tmp(output_dir / "tmp", allow_dir=True) as tmp_dir:
            failed = fetch_swepos_files(
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
                    raise FileNotFoundError("Download from Swepos failed.")

            merged_obs, merged_nav = merge_swepos_rinex(tmp_dir)

        if not merged_obs.is_file():
            raise FileNotFoundError(f"Generated OBS file: {merged_obs} not found")
        if not merged_nav is None and not merged_nav.is_file():
            raise FileNotFoundError(f"Generated NAV file: {merged_nav} not found")
        
        start_utc, _, etrs89_pos, _ = extract_rnx_info(merged_obs)
        wgs84_pos = etrs89_to_wgs84_ecef(*etrs89_pos, utc_datetime=start_utc)
        update_rinex_position(merged_obs, wgs84_pos)
        return merged_obs, merged_nav
    
def station_ppp(
        obs_path: str|Path,
        navglo_path: str|Path|None = None,
        atx_path: str|Path|None = None,
        antrec_path: str|Path|None = None,
        sp3_file: str|Path|None = None,
        clk_file: str|Path|None = None,
        max_downloads: int = 10,
        max_retries: int = 3,
        out_path: str|Path|None = None,
        elevation_mask: float|None = None,
        header: bool = True,
        dry: bool = False,
        retain: bool = False,
        make_mocoref: bool = True
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, Path, Path]:
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
    
    Output:
    - Base position
    - Rotation matrix from ECEF to local ENU at base position
    - Distance from header position to base position
    OR if retain:
    - Base postion
    - SP3 file path
    - CLK file path"""

    obs_path = Path(obs_path)
    if not obs_path.is_file():
        raise FileNotFoundError(f"Rinex observation file not found: {obs_path}")

    if out_path:
        out_path = Path(out_path)
        if out_path.is_dir():
            output_dir = out_path
            out_path = None
        else:
            output_dir = out_path.parent
            output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = obs_path.parent

    start_utc, end_utc, approx_pos, antenna_delta = extract_rnx_info(obs_path)
    with tmp(output_dir / "tmp", allow_dir=True) as tmp_dir:
        if not sp3_file:
            failed = fetch_sp3_clk(start_time=start_utc, end_time=end_utc, output_dir=tmp_dir, max_workers=max_downloads, max_retries=max_retries, dry=dry)
            if failed:
                raise FileNotFoundError("Download of precise ephemeris and clock data from ESA failed.")
            
            sp3_file, clk_file = merge_ephemeris(tmp_dir)
            if retain:
                # Move files out of temporary directory
                sp3_file = shutil.move(sp3_file, output_dir)
                clk_file = shutil.move(clk_file, output_dir)

        # Run PPP command
        out = ""
        print()
        out = ppp(
            obs_file=obs_path,
            sp3_file=sp3_file,
            clk_file=clk_file,
            out_path=out_path,
            navglo_file=navglo_path,
            atx_file=atx_path,
            antrec_file=antrec_path,
            elevation_mask=elevation_mask
        )
   
    if out_path and not out_path.is_file():
        raise FileNotFoundError(f"Cannot find generated out file: {out_path}")

    # Extract position
    pos, rotation, _, _, _ = read_glab_out(out, verbose=True)
    # Update position to be to the ARP
    pos = pos - rotation.T @ antenna_delta

    # Compare against header
    diff = rotation @ [pos[0] - approx_pos[0], pos[1] - approx_pos[1], pos[2] - approx_pos[2]]
    distance = math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
    print(f"Distance from header position: {distance:.3} m (E: {diff[0]:.3} m, N: {diff[1]:.3} m, U: {diff[2]:.3} m)")
    
    if header:
        update_rinex_position(obs_path, pos)
    if make_mocoref:
        lon, lat, h = ecef_to_geo.transform(*pos)
        settings = Settings()
        mocoref = {
            settings.MOCOREF_LATITUDE: lat,
            settings.MOCOREF_LONGITUDE: lon,
            settings.MOCOREF_HEIGHT: h,
            settings.MOCOREF_ANTENNA: 0.
        }
        generate_mocoref(mocoref, generate=True)
    
    if retain:
        # Return only pos along with SP3 and CLK file
        return pos, sp3_file, clk_file
    return pos, rotation, distance

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
        output_dir = archive.parent

    obs_file = output_dir / (base_name + ".obs")

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
            # Update stard and end timestamps to match the OBS file
            start = obs_data[obs_file]["TIME OF FIRST OBS"]
            end = obs_data[obs_file]["TIME OF LAST OBS"]

            # Extract mocoref.moco file
            with tmp(output_dir / "llh.tmp") as llh_tmp:
                extract_to(llh_file, llh_tmp, final_destination=llh_tmp.with_name("mocoref.moco"))
                mocoref_data, mocoref_file = generate_mocoref(llh_tmp, type="LLH", generate=True, tstart=start, tend=end)
            
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
        