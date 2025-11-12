import click
import os
from datetime import timedelta
from pathlib import Path
import math
import numpy as np
from matplotlib import pyplot as plt

from .. import ubx2rnx, rnx2rtkp
from ..gnss import fetch_swepos, station_ppp as run_ppp, read_rnx2rtkp_out, reachz2rnx, rtkp
from ..utils import generate_mocoref, ecef2enu
from ..transformers import geo_to_ecef
from ..config import LOCAL
from ..binaries import resource, tmp

@click.group()
def test() -> None:
    """Entry point for tomotest utilities."""
    pass

@test.command()
@click.option("--savar", is_flag=True, help="Process Savar test file instead of default (SVB)")
def gnss(savar) -> None:
    """Test GNSS processing capabilities."""
    test_dir = LOCAL / "gnss_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(test_dir)
    print(f"Running test in {test_dir} ...")
    if savar:
        test_file = "TEST_FILE_SAVAR"
    else:
        test_file = "TEST_FILE_SVB"
    with resource(None, test_file) as ubx:
        rover_obs, rover_nav, _ = ubx2rnx(ubx, sbs=False)
        if rover_obs.is_file() and rover_nav.is_file():
            print("TEST: Drone GNSS test sample succesfully converted to RINEX")
            print()
        else:
            raise RuntimeError("rtklib binary convbin failed to generate RINEX files from test UBX file.")
        
        print("Reading RINEX data ...", flush=True)
        swepos_obs, swepos_nav = fetch_swepos(rover_obs, fetch_nav=True, min_dur=4, output_dir="SWEPOS")
        if swepos_obs.is_file() and swepos_nav.is_file():
            print("TEST: Download, unpacking and merging sucessful")
        else:
            raise RuntimeError("Download, unpacking or merging of RINEX files from Swepos failed. See above.")

        print()
        print("Running station PPP post processing on Swepos files ...")
        _, _, distance = run_ppp(swepos_obs, swepos_nav, header=False)

        if distance < 0.2:
            print("TEST: station PPP sucessful")
            print()
        else:
            raise RuntimeError("Station PPP processing produced poor solution")

        # Run RTKP
        out_path = rover_obs.with_suffix(".pos")
        rnx2rtkp(rover_obs, swepos_obs, rover_nav, out_path)
        try: 
            _, gpst, q = read_rnx2rtkp_out(out_path)
            quality_conversion = np.sum(q == 1) / len(q) * 100
            dur = timedelta(seconds=gpst[-1] - gpst[0])
            q
            print(f"Total {dur} processed, Q1={quality_conversion:.2f} %")
        except:
            raise RuntimeError("rnx2rtkp failed to produce a .pos file with content")

        if quality_conversion > 99 and dur > timedelta(minutes=10):
            print("TEST: RTKP processing sucessful")
            print()
            print("GNSS processing OPERATIONAL!")
        else:
            raise RuntimeError("rnx2rtkp produced poor Q1 quality or lost time, check tomosar settings RTKP_CONFIG")
        
@test.command()
@click.argument("target", type=click.Path(exists=True, path_type=Path))
@click.option("-m", "--mocoref", "mocoref_file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("-n", "--navglo", "navglo_path", type=click.Path(exists=True, path_type=Path), default=None, help="Path to GLONASS navigation data file (can be a general/merged NAV file)")
@click.option("-z", "--zip", "is_zip", is_flag=True, help="TARGET is a Reach ZIP archive (default for .zip files)")
@click.option("--csv", is_flag=True, help="Mocoref file is a CSV file (default for .csv and .CSV files)")
@click.option("--json", is_flag=True, help="Mocoref file is a JSON file (default for .json and .JSON files)")
@click.option("--llh", is_flag=True, help="Mocoref file is an LLH log (default for .llh and .LLH files)")
@click.option("-l", "--line", type=int, default=1, help="Line in CSV file to read data from (default=1)")
@click.option("--offset", type=float, default=-0.079, help="Specify vertical PCO between data log receiver and drone processing receiver (default=-0.079) for CSV files")
@click.option("--ref", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Reference RINEX OBS")
def station_ppp(
    target: Path,
    mocoref_file: Path,
    navglo_path: Path|None,
    is_zip: bool,
    csv: bool, 
    json: bool,
    llh: bool,
    line: int,
    offset: float,
    ref: Path,
) -> None:
    if target.suffix == ".zip" or is_zip:
        # Assume Reach ZIP archive
        with tmp(target.parent / "tmp", allow_dir=True) as tmp_dir:
            obs_data, (obs_file, mocoref_file, nav_file) = reachz2rnx(target, output_dir=tmp_dir, rnx_file=ref, nav=True, verbose=True)
            (mocoref_latitude, mocoref_longitude, mocoref_height) = obs_data[mocoref_file]
            pos, rotation, _ = run_ppp(obs_file, nav_file, header=False, retain=False, make_mocoref=False)
    else:
        if not mocoref_file:
            raise FileNotFoundError(f"To test station-ppp on a RINEX OBS file, specify a mocoref file by --mocoref")
        # Get mocoref position
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
        (mocoref_latitude, mocoref_longitude, mocoref_height), _ = generate_mocoref(
            mocoref_file,
            type=type,
            line=line,
            pco_offset=offset,
            generate=False
        )
        pos, rotation, _ = run_ppp(target, navglo_path=navglo_path, header=False, retain=False, make_mocoref=False)

    mocoref_pos = geo_to_ecef.transform(mocoref_longitude, mocoref_latitude, mocoref_height)

    diff = rotation @ [mocoref_pos[0] - pos[0], mocoref_pos[1] - pos[1], mocoref_pos[2] - pos[2]]
    distance = math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
    print(f"Distance: {distance:.3} m (E: {diff[0]:.3} m, N: {diff[1]:.3} m, U: {diff[2]:.3} m)")

@test.command()
@click.argument("rover_obs", type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.argument("nav_path", type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.argument("base_obs", type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.option("--sp3", "sp3_path", type=click.Path(exists=True, path_type=Path, dir_okay=False), help="Path to SP3 file (if not specifed: download)", default=None)
@click.option("--clk", "clk_path", type=click.Path(exists=True, path_type=Path, dir_okay=False), help="Path to CLK file to complement SP3 file if needed", default=None)
@click.option("-k", "--config", 'conf_path', help="Path to config file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("-s", "--sbas", 'sbs_path', help="Path to SBAS corrections file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--mocoref", 'mocoref_path', help="Path to mocoref file for first base", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("-m", "--mask", "elevation_mask", type=float, help="Specify elevation mask")
def precise_rtkp(rover_obs, base_obs, nav_path, sp3_path, clk_path, conf_path, sbs_path, mocoref_path, elevation_mask) -> None:
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
    ax.set_ylabel("Coordinate difference (m)")

    fig.supxlabel("GPST (s)")

    plt.show()