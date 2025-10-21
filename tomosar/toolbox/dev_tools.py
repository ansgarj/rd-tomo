import pytz
import click
from pathlib import Path
from pyproj import Transformer
import csv
import matplotlib.pyplot as plt
import csv
import struct
import numpy as np
import subprocess

from ..gnss import extract_rnx_info, read_out_file, read_pos_file
from ..binaries import rnx2rtkp, resource
from ..config import PACKAGE_PATH

from pathlib import Path
   
@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-d", "--duration", type=int, help="Minimized duration (seconds)", default=600)
def minimize_ubx(input_file, duration: int = 600) -> None:
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

@click.command()
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
        "anglvel_x (deg/s)", "anglvel_y (deg/s)", "anglvel_z (deg/s)",
        "accel_x (g)", "accel_y (g)", "accel_z (g)",
        "channel_7 (raw)", "channel_8 (raw)"
    ]
    rows = []
    with open(bin_filename, "rb") as binfile, open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)

        count = 0
        while True:
            if num_samples is not None and count >= num_samples:
                break
            sample = binfile.read(32)
            if len(sample) < 32:
                break
            # Unpack 8 little-endian floats
            row = list(struct.unpack('<8f', sample))
            writer.writerow([f"{v:.6f}" for v in row])
            rows.append(row)
            count += 1
    return rows, labels

@click.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("-n", "--num", type=int, default=None, help="Number of samples to read")
def read_imu(file: Path, num: int|None) -> None:
    """Try to read an IMU log and convert to CSV, and plot the results."""
    # Load and plot
    channels, labels = extract_imu_to_csv(file, file.with_suffix('.csv'), num_samples=num)
    channels = [ [row[i] for row in channels] for i in range(8) ]
    plt.figure(figsize=(14, 12))
    for i in range(8):
        plt.subplot(4, 2, i+1)
        plt.plot(channels[i][1:], '-x')
        plt.title(labels[i])
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

@click.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def inspect_out(file: Path) -> None:
    """Inspect the .out file produced by glab."""
    _, _, conv_idx, _, residuals = read_out_file(
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



def plot_coordinates(coords1, coords2, labels, title1, title2):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):
        axes[i].plot(coords1[:, i], label=title1, alpha=0.7)
        axes[i].plot(coords2[:, i], label=title2, alpha=0.7)
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel('Epoch Index')
    plt.tight_layout()
    plt.show()

def plot_difference(t1, coords1, t2, coords2, labels):
    common_times = np.intersect1d(t1, t2)

    # Get indices in t1 and t2 where these common times occur
    indices_t1 = np.nonzero(np.isin(t1, common_times))[0]
    indices_t2 = np.nonzero(np.isin(t2, common_times))[0]
    diff = coords1[indices_t1] - coords2[indices_t2]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axes[i].plot(diff[:, i], color='purple', label=f'Difference ({labels[i]})')
        axes[i].set_ylabel(f'Δ {labels[i]}')
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel('Epoch Index')
    diff_mean = diff.mean(axis=0)
    diff_std = diff.std(axis=0)
    print(f"Mean difference: X={diff_mean[0]:.2g} m ({diff_std[0]:.2g} m), Y={diff_mean[1]:.2g} m ({diff_std[1]:.2g} m), Z={diff_mean[2]:.2g} m ({diff_std[2]:.2g} m)")
    plt.tight_layout()
    plt.show()

@click.command()
@click.argument("rover_obs", type=click.Path(exists=True, path_type=Path))
@click.argument("base1_obs", type=click.Path(exists=True, path_type=Path))
@click.argument("base2_obs", type=click.Path(exists=True, path_type=Path))
@click.argument("nav_path", type=click.Path(exists=True, path_type=Path))
@click.option("-k", "--config", 'conf_path', help="Path to config file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("-s", "--sbas", 'sbs_path', help="Path to SBAS corrections file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("-m", "--mocoref", 'mocoref_path', help="Path to mocoref file for first base", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("-f", "--force", is_flag=True, help="Force reprocessing")
def compare_rtkp(rover_obs, base1_obs, base2_obs, nav_path, conf_path, sbs_path, mocoref_path, force) -> None:
    """Compare .pos files from two rnx2rtkp calls"""

    pos1_path =  base1_obs.with_suffix(".pos")
    pos2_path = base2_obs.with_suffix(".pos")
    # Run base 1
    with resource(conf_path, "RTKP_CONFIG") as config:
        rnx2rtkp(
            rover_obs=rover_obs,
            base_obs=base1_obs,
            nav_file=nav_path,
            config=conf_path,
            out_path=pos1_path,
            mocoref_file=mocoref_path,
            sbs_file=sbs_path
        )
    # Read results
    coords1, q1, t1 = read_pos_file(pos1_path)
    # Display Q=1 percentage
    print(f"{pos1_path} Q1: {q1} %")

    # Run base 2
    with resource(conf_path, "RTKP_CONFIG") as config:
        rnx2rtkp(
            rover_obs=rover_obs,
            base_obs=base2_obs,
            nav_file=nav_path,
            config=config,
            out_path=pos1_path,
            mocoref_file=None,
            sbs_file=sbs_path
        )
    # Read results
    coords2, q2, t2 = read_pos_file(pos2_path)
    # Display Q=1 percentage
    print(f"{pos2_path} Q1: {q2} %")

    labels = ['East/X', 'North/Y', 'Up/Z']

    #plot_coordinates(coords1, coords2, labels, pos1, pos2)

    plot_difference(t1, coords1, t2, coords2, labels)
