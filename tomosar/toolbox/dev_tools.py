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
from tqdm import tqdm

from ..gnss import extract_rnx_info, read_glab_out, rtkp
from ..utils import ecef2enu
from ..transformers import geo_to_ecef
from ..config import PACKAGE_PATH
from .setup_tools import install_changed

matplotlib.use("Qt5Agg")

@click.group(hidden=True)
def dev() -> None:
    """Entry point for dev tools"""
    pass
   
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
@click.argument("rover_obs", type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.argument("nav_path", type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.argument("base_obs", type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.option("--sp3", "sp3_path", type=click.Path(exists=True, path_type=Path, dir_okay=False), help="Path to SP3 file (if not specifed: download)", default=None)
@click.option("--clk", "clk_path", type=click.Path(exists=True, path_type=Path, dir_okay=False), help="Path to CLK file to complement SP3 file if needed", default=None)
@click.option("-k", "--config", 'conf_path', help="Path to config file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("-s", "--sbas", 'sbs_path', help="Path to SBAS corrections file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--mocoref", 'mocoref_path', help="Path to mocoref file for first base", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("-m", "--mask", "elevation_mask", type=float, help="Specify elevation mask")
def compare_rtkp(rover_obs, base_obs, nav_path, sp3_path, clk_path, conf_path, sbs_path, mocoref_path, elevation_mask) -> None:
    """Compare solutions from precise and broadcast ephemeris solutions"""

    # Run base 1
    print("PRECISE:")
    coords_precise, gpst, q_precise= rtkp(rover_obs, base_obs, nav_path, config_file=conf_path, sbs_file=sbs_path, mocoref_file=mocoref_path, elevation_mask=elevation_mask, precise=True, sp3_file=sp3_path, clk_file=clk_path)

    # Run base 2
    print("BROADCAST:")
    coords_bc, _, q_bc = rtkp(rover_obs, base_obs, nav_path, config_file=conf_path, sbs_file=sbs_path, mocoref_file=mocoref_path, elevation_mask=elevation_mask)

    fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(8, 8), sharex=True, tight_layout=True)
    axs = axs.flatten()
    ax = axs[0]
    precise_only = (q_precise != 1) & (q_bc == 1)
    bc_only = (q_bc != 1) & (q_precise == 1)
    both =( q_precise != 1) &( q_bc != 1)
    #ax.plot(gpst, coords_precise[:,2], 'g-', label=f"Precise")
    ax.plot(gpst, coords_bc[:,2], 'g-', label=f"Broadcast track")
    ax.plot(gpst[precise_only], coords_precise[:,2][precise_only], 'r+', label=f"Precise only float")
    ax.plot(gpst[bc_only], coords_bc[:,2][bc_only], 'm+', label=f"Broadcast only float")
    ax.plot(gpst[both], coords_bc[:,2][both], 'y+', label=f"Both float")
    ax.set_ylabel("Ellipsoidal Height (m)")
    ax.legend()
    ax = axs[1]

    diff = np.vstack([
        ecef2enu(coords_precise[n, 0], coords_precise[n, 1]) @ 
            (np.asarray(geo_to_ecef.transform(*coords_bc[n, :])) - np.asarray(geo_to_ecef.transform(*coords_precise[n, :])))
            for n in range(len(gpst))
    ])

    dist = np.sqrt((diff**2).sum(axis=1))

    ax.plot(gpst, dist, label="Distance (m)")
    #ax.plot(gpst, diff[:,0], label="Latitude (deg)")
    #ax.plot(gpst, diff[:,1], label="Longitude (deg)")
    #ax.plot(gpst, diff[:,2], label="Height (m)")
    ax.set_ylabel("Coordinate difference (m)")

    fig.supxlabel("GPST (s)")

    plt.show()


@dev.command()
def update_install() -> None:
    """Updates the local install.hash to match current pyproject.toml"""
    install_changed(update=True)