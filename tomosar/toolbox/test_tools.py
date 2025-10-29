import click
import os
from datetime import timedelta

from .. import ubx2rnx, rnx2rtkp
from ..gnss import fetch_swepos, station_ppp, read_pos_file
from ..config import LOCAL
from ..binaries import resource

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
    if savar:
        test_file = "TEST_FILE_SAVAR"
    else:
        test_file = "TEST_FILE_SVB"
    with resource(None, test_file) as ubx:
        rover_obs, rover_nav, _ = ubx2rnx(ubx)
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
        distance = station_ppp("SWEPOS", header=False)

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

        if q > 95 and dur > timedelta(minutes=10):
            print("TEST: RTKP processing sucessful")
            print()
            print("GNSS processing operational!")
        else:
            raise RuntimeError("rnx2rtkp produced poor Q1 quality or lost time, check tomosar settings RTKP_CONFIG")