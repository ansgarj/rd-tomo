import click
import os

from .. import ubx2rnx, rnx2rtkp
from ..gnss import fetch_swepos, station_ppp
from ..config import LOCAL
from ..binaries import resource

@click.group()
def test() -> None:
    """Entry point for tomotest utilities."""
    pass

@test.command()
def gnss() -> None:
    """Test GNSS processing capabilities."""
    test_dir = LOCAL / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(test_dir)
    with resource(None,"TEST_FILE") as ubx:
        rover_obs, rover_nav, _ = ubx2rnx(ubx)
        if rover_obs.is_file() and rover_nav.is_file():
            print("TEST: Drone GNSS test sample succesfully converted to RINEX")
        else:
            raise RuntimeError("rtklib binary convbin failed to generate RINEX files from test UBX file.")
        
        print("Reading RINEX data ...", flush=True)
        swepos_obs, swepos_nav = fetch_swepos(rover_obs, fetch_nav=True, min_dur=4)
        if swepos_obs.is_file() and swepos_nav.is_file():
            print("TEST: Download, unpacking and merging sucessful")
        else:
            raise RuntimeError("Download, unpacking or merging of RINEX files from Swepos failed. See above.")

        print("Running station PPP post processing on Swepos files ...")
        distance = station_ppp(test_dir, header=False)
    