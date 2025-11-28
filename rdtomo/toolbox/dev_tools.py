import pytz
import click
from pathlib import Path
from pyproj import Transformer
import csv
import matplotlib.pyplot as plt
import matplotlib
import csv
import struct
import numpy as np

from ..gnss import extract_rnx_info, read_glab_out, rtkp as run_rtkp
from ..config import PACKAGE_PATH, Settings
from .setup_tools import install_changed
from ..data import DataDir

matplotlib.use("Qt5Agg")

@click.group(hidden=True)
def dev() -> None:
    """Entry point for dev tools"""
    pass

@dev.command()
def update_install() -> None:
    """Updates the local install.hash to match current pyproject.toml"""
    install_changed(update=True)
   
@dev.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-d", "--duration", type=int, help="Minimized duration (seconds)", default=600)
def sample_ubx(input_file, duration: int = 600) -> None:
    """Extract a segment from a UBX file"""
    from pyubx2 import UBXReader
    with open(input_file, 'rb') as infile:
        ubr = UBXReader(infile, protfilter=2)  # UBX only
        messages = []
        start_itow = None

        for raw_data, parsed_data in ubr:
            if hasattr(parsed_data, 'iTOW'):
                if start_itow is None:
                    start_itow = parsed_data.iTOW
                if parsed_data.iTOW - start_itow <= duration * 1000:
                    messages.append(raw_data)
                else:
                    break
            else:
                if start_itow is not None:
                    messages.append(raw_data)
    output_file = PACKAGE_PATH / "tests" / "minimal.ubx"
    with open(output_file, 'wb') as outfile:
        for msg in messages:
            outfile.write(msg)

    print(f"Saved {len(messages)} messages to {output_file}")

@dev.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def rnx_info(file: Path) -> None:
    """Display timestamp and header position info extracted from a RNX file."""
    start_utc, end_utc, pos = extract_rnx_info(file)
    stockholm_tz = pytz.timezone('Europe/Stockholm')

    if start_utc and end_utc:
        start_local = start_utc.astimezone(stockholm_tz)
        end_local = end_utc.astimezone(stockholm_tz)
        print(f"Start time: {start_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} / {start_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"End time: {end_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} / {end_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        print("No valid timestamps found in the file.")
    if pos:
        lon, lat, h = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True).transform(*pos)
        print(f"Header position: lat={lat}, lon={lon}, height={h}")
    else:
        print("No position given in file header.")

def extract_imu_to_csv(bin_filename: str | Path, csv_filename: str | Path, num_samples: int = None):
    labels = [
        "channel_7 (raw)", "channel_8 (raw)",
        "anglvel_x (deg/s)", "anglvel_y (deg/s)", "anglvel_z (deg/s)",
        "accel_x (g)", "accel_y (g)", "accel_z (g)",
    ]
    rows = []
    with open(bin_filename, "rb") as binfile, open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)

        count = 0
        nan_count = 0
        while True:
            if num_samples is not None and count >= num_samples:
                break
            sample = binfile.read(32)
            if len(sample) < 32:
                break
            # Unpack 8 little-endian floats
            row = list(struct.unpack('<8f', sample))
            nan_count += np.isnan(row).sum()
            writer.writerow([f"{v:.6f}" for v in row])
            rows.append(row)
            count += 1
    
    rows = [ [row[i] for row in rows] for i in range(8) ]
    print(f"\nTotal NaN count: {nan_count/(8*count) * 100} %")
    return rows, labels

def close_micro_gaps(data, max_gap=1):
    data = np.array(data)
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        data[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            data[~nan_mask]
        )
    return data

@dev.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("-n", "--num", type=int, default=None, help="Number of samples to read")
@click.option("-p", "--processed", is_flag=True, help="Look for processed file and plot both")
def read_imu(file: Path, num: int|None, processed: bool = False) -> None:
    """Try to read an IMU log and convert to CSV, and plot the results."""
    # Load and plot
    channels, labels = extract_imu_to_csv(file, file.with_suffix('.csv'), num_samples=num)
    processed_channels = []
    if file.with_suffix(".bin_out").exists() and processed:
        processed_channels, _ = extract_imu_to_csv(file.with_suffix(".bin_out"), file.with_suffix(".csv_out"), num_samples=num)
    plt.figure(figsize=(14, 12))
    for i in range(8):
        plt.subplot(4, 2, i+1)
        data = np.array(channels[i][1:])
        non_nan_data = close_micro_gaps(data, max_gap=1)

        nan_indices = np.where(np.isnan(data))[0]
        if processed_channels:
            processed_data = np.array(processed_channels[i][1:])
            plt.plot(processed_data, 'gx', label='Processed Data')
            factor = processed_data[~np.isnan(data)] / data [~np.isnan(data)]
            factor = np.median(factor)
            print(f"Channel {i}: {factor}")
            if not np.isnan(factor):
                data = factor * data
                non_nan_data = factor * non_nan_data / 2

        plt.plot(data, 'bx', label='Data')
        # Find and mark NaNs with red 'x'
        plt.plot(nan_indices, non_nan_data[nan_indices], 'rx', label='NaN')

        plt.legend()
        plt.title(labels[i])
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

@dev.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def inspect_out(file: Path) -> None:
    """Inspect the .out file produced by glab."""
    _, _, conv_idx, diff, residuals = read_glab_out(
        file_path=file,
        verbose=True
    )

    coords = ["X", "Y", "Z"]
    plt.axvline(x=conv_idx, linestyle="--", color='r', label="Convergence")
    for i, res in enumerate(residuals):
        plt.plot(res, label=f"{coords[i]}")
        plt.xlabel("Epoch number")
        plt.ylabel("Residual")
        plt.title("Residual plot")
        plt.legend()
    plt.show()

    coords = ["E", "N", "U"]
    plt.axvline(x=conv_idx, linestyle="--", color='r', label="Convergence")
    for i, d in enumerate(diff):
        plt.plot(d, label=f"{coords[i]}")
        plt.xlabel("Epoch number")
        plt.ylabel("Divergence")
        plt.title("Divergence plot")
        plt.legend()
    plt.show()

@dev.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=DataDir), default=DataDir.cwd())
@click.option("--swepos", "use_swepos", is_flag=True, help="Substitute for base OBS with files from nearest Swepos station")
@click.option("--ppp", "use_ppp", is_flag=True, help="Subsitute for mocoref data by running static PPP on base OBS")
@click.option("-z", "-zip", "is_zip", is_flag=True, help="Force base OBS and mocoref.moco files to be generated from a Reach ZIP archive")
@click.option("--mocoref", "is_mocoref", is_flag=True, help="Force mocoref data to be read from mocoref.moco file")
@click.option("--csv", "is_csv", is_flag=True, help="Force mocoref data to be read from CSV file")
@click.option("--json", "is_json", is_flag=True, help="Force mocoref data to be read from JSON file")
@click.option("--llh", "is_llh", is_flag=True, help="Force mocoref data to be read from LLH file")
@click.option("--rnx", "is_rnx", is_flag=True, help="Force base OBS to be directly accessible (not extracted)")
@click.option("--hcn", "is_hcn", is_flag=True, help="Force base OBS to be extracted from a .HCN file")
@click.option("--rtcm3", "is_rtcm3", is_flag=True, help="Force base OBS to be extracted from a .RTCM3 file")
@click.option("-h", "--header", "use_header", is_flag=True, help="Read mocoref data from RINEX header (no separate file, use ONLY if RINEX header is known to contain precise position)")
@click.option("--broadcast", "use_broadcast", is_flag=True, help="Use broadcast ephemeris data (NOTE: this may improve Q1 percentage, but risks reducing integrity, run tomosar test precise-rktp to test)")
@click.option("-a", "--atx", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the satellite antenna .atx file")
@click.option("-r", "--receiver", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the .atx file containing receiver antenna info")
@click.option("--downloads", type=int, default=10, help="Max number of parallel downloads (default: 10)")
@click.option("--attempts", type=int, default=3, help="Max number of attempts for each file (default: 3)")
@click.option("-m", "--mask", "elevation_mask", type=float, default=None, help="Elevation mask for satellites")
@click.option("-l", "--line", "csv_line", type=int, default=1, help="Line in CSV file to read data from (default=1)")
@click.option("--offset", type=float, default=-0.079, help="Specify vertical PCO between mocoref data log receiver and drone processing receiver (default=-0.079) for CSV files")
@click.option("--overlap", "minimal_overlap", type=float, default=10, help="Specify minimal overlap between base OBS and drone flight in minutes (default: 10 minutes)")
@click.option("-k", "--config", type=click.Path(exists=True, path_type=Path), default=None, help="Specify external config file for rnx2rtkp")
def rtkp_frames(
    path: DataDir,
    use_swepos: bool,
    use_ppp: bool,
    is_zip: bool,
    is_mocoref: bool,
    is_csv: bool,
    is_json: bool,
    is_llh: bool,
    is_rnx: bool,
    is_hcn: bool,
    is_rtcm3: bool,
    use_header: bool,
    use_broadcast: bool,
    atx: Path,
    receiver: Path,
    downloads: int,
    attempts: int,
    elevation_mask: float,
    csv_line: int,
    offset: float,
    minimal_overlap: float,
    config: Path,
) -> None:
    """Compare solutions from processing with explicit ITRF -> TARGET_FRAME transformation and assuming track is in same frame as
    reference base coordinates. This test opens a Data Directory and runs RTKP post processing."""

    with path.open(
        require_drone=True,
        atx = atx,
        receiver = receiver,
        use_swepos = use_swepos,
        use_ppp = use_ppp,
        use_header=use_header,
        is_zip = is_zip,
        is_mocoref = is_mocoref,
        is_csv = is_csv,
        is_llh = is_llh,
        is_json = is_json,
        is_rnx = is_rnx,
        is_hcn = is_hcn,
        is_rtcm3 = is_rtcm3,
        csv_line = csv_line,
        offset = offset,
        download_attempts = attempts,
        max_downloads = downloads,
        elevation_mask = elevation_mask,
        minimal_overlap = minimal_overlap,
    ) as data:
        if use_swepos and not elevation_mask:
            if not use_broadcast:
                elevation_mask = 20 # Precise 
            else:
                elevation_mask = 5
        results_itrf = run_rtkp(
            rover_obs=data.drone_rnx_obs,
            base_obs=data.base_obs,
            nav_file=data.drone_rnx_nav,
            sbs_file=data.drone_rnx_sbs,
            sp3_file=data.sp3,
            clk_file=data.clk,
            inx_file=data.inx,
            atx_file=atx,
            receiver_file=receiver,
            precise=not use_broadcast,
            out_path=data.drone_rnx_obs.with_suffix(".pos"),
            config_file=config,
            elevation_mask=elevation_mask,
            mocoref_file=data.mocoref,
            retain=False
        )
        print()
        st = Settings()
        results_mf = run_rtkp(
            rover_obs=data.drone_rnx_obs,
            base_obs=data.base_obs,
            nav_file=data.drone_rnx_nav,
            sbs_file=data.drone_rnx_sbs,
            sp3_file=results_itrf["sp3"],
            clk_file=results_itrf["clk"],
            inx_file=results_itrf["inx"],
            precise=not use_broadcast,
            out_path=data.drone_rnx_obs.with_suffix(".pos"),
            config_file=config,
            elevation_mask=elevation_mask,
            mocoref_file=data.mocoref,
            processing_frame=st.MOCOREF_FRAME
        )
    
    coords_itrf, gpst, q_itrf = results_itrf["coordinates"], results_itrf["gpst"], results_itrf["quality"]
    coords_mf, q_mf = results_mf["coordinates"], results_mf["quality"]

    # Index tracking
    int_only = (q_itrf != 1) & (q_mf == 1)
    raw_only = (q_mf != 1) & (q_itrf == 1)
    both = (q_itrf != 1) & (q_mf != 1)
    
    fig, axs = plt.subplots(3, 1, squeeze=False, figsize=(12, 12), sharex=True, tight_layout=True)
    axs = axs.flatten()
    ax = axs[0]
    #ax.plot(gpst, coords_precise[:,2], 'g-', label=f"Precise")
    ax.plot(gpst, coords_itrf[2,:], 'g-', label=f"ITRF2020 track changed to {st.MOCOREF_FRAME}")
    ax.plot(gpst, coords_mf[2,:], 'b:', label=f"Assumed {st.MOCOREF_FRAME} track")
    ax.plot(gpst[int_only], coords_itrf[2, int_only], 'r+', label=f"ITRF2020 only float")
    ax.plot(gpst[raw_only], coords_mf[2, raw_only], 'm+', label=f"{st.MOCOREF_FRAME} only float")
    ax.plot(gpst[both], coords_itrf[2, both], 'y+', label=f"Both float (ITRF2020)")
    ax.plot(gpst[both], coords_mf[2, both], 'c+', label=f"Both float ({st.MOCOREF_FRAME})")
    ax.set_ylabel("Ellipsoidal Height (m)")
    ax.legend()
    
    ax = axs[1]
    diff = coords_mf - coords_itrf
    dist = np.sqrt((diff**2).sum(axis=0)).squeeze()
    ax.plot(gpst, dist, label="Distance (m)")
    ax.set_ylabel("Coordinate difference (m)")

    ax = axs[2]
    ax.plot(gpst, results_itrf["ratio"], 'g-', label="ITRF2020")
    ax.plot(gpst, results_mf["ratio"], 'r-', label=f"{st.MOCOREF_FRAME}")
    ax.set_ylabel("AR Ratio")
    ax.legend()

    fig.supxlabel("GPST (s)")

    plt.show()

