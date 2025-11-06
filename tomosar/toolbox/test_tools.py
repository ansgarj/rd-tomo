import click
import os
from datetime import timedelta
from pathlib import Path
import math

from .. import ubx2rnx, rnx2rtkp
from ..gnss import fetch_swepos, station_ppp as run_ppp, read_pos_file, reachz2rnx
from ..utils import generate_mocoref
from ..config import LOCAL
from ..binaries import resource, tmp
from ..transformers import geo_to_ecef

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
            _, q, gpst = read_pos_file(out_path)
            dur = timedelta(seconds=gpst[-1] - gpst[0])
            print(f"Total {dur} processed, Q1={q:.2f} %")
        except:
            raise RuntimeError("rnx2rtkp failed to produce a .pos file with content")

        if q > 99 and dur > timedelta(minutes=10):
            print("TEST: RTKP processing sucessful")
            print()
            print("GNSS processing OPERATIONAL!")
        else:
            raise RuntimeError("rnx2rtkp produced poor Q1 quality or lost time, check tomosar settings RTKP_CONFIG")
        
@test.command()
@click.argument("target", type=click.Path(exists=True, path_type=Path))
@click.argument("-m", "--mocoref", "mocoref_file", type=click.Path(exists=True, path_type=Path), default=None)
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
            _, (obs_file, mocoref_file, nav_file) = reachz2rnx(target, output_dir=tmp_dir, )
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
