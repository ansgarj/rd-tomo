import click
from pathlib import Path
import os
import time as Time
import datetime

from ..gnss import fetch_swepos as run_fetch_swepos, station_ppp as run_station_ppp
from ..trackfinding import trackfinder as run_trackfinder
from .. import ImageInfo, TomoScenes
from ..utils import interactive_console
from ..forging import tomoforge

@click.command()
@click.argument("filepath", type=click.Path(exists=True, path_type=Path))
@click.option("--stations", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the SWEPOS coordinate list CSV")
@click.option("--downloads", type=int, default=10, help="Max number of parallel downloads (default: 10)")
@click.option("--attempts", type=int, default=3, help="Max number of attempts for each file (default: 3)")
@click.option("-o", "--output", type=click.Path(path_type=Path), default="SWEPOS", help="Output directory for SWEPOS RINEX files")
@click.option("-d", "--dry", is_flag=True, help="Dry run without downloads")
@click.option("--cont", is_flag=True, help="Continue run after downloads complete")
@click.option("-n","--nav", is_flag=True, help="Also fetch nav files.")
def fetch_swepos(filepath, stations, downloads, attempts, output, dry, cont, nav) -> None:
    """Extract GNSS info and find nearest SWEPOS station.
    Then download files into output directory."""
    run_fetch_swepos(
        filepath=filepath,
        stations_path=stations,
        max_downloads=downloads,
        max_retries=attempts,
        dry=dry,
        output_dir=output,
        fetch_nav=nav,
        cont=cont
    )

@click.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-a", "--atx", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the satellite antenna .atx file")
@click.option("-r", "--receiver", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the .atx file containing receiver antenna info")
@click.option("--downloads", type=int, default=10, help="Max number of parallel downloads (default: 10)")
@click.option("--attempts", type=int, default=3, help="Max number of attempts for each file (default: 3)")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None, help="Output directory for SWEPOS rinex files")
@click.option("-d", "--dry", is_flag=True, help="Dry run without downloads")
@click.option("--cont", is_flag=True, help="Continue run after downloads complete")
@click.option("-x", "--no-header", 'header', is_flag=True, default=True, flag_value=False, help="Do not fodify OBS file header with new position.")
def station_ppp(data_dir, atx, receiver, downloads, attempts, output, dry, cont, header) -> None:
    """Extract GNSS info and find nearest SWEPOS station."""
    run_station_ppp(
        data_dir=data_dir,
        atx_path=atx,
        antrec_path=receiver,
        max_downloads=downloads,
        max_retries=attempts,
        dry=dry,
        output_dir=output,
        cont=cont,
        header=header
    )

@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("-l", "--linear", type=int, default=0, help="Specify linear track index to modify radar-[...].inf (0 for spiral flights)")
@click.option("-v", "--verbose", is_flag=True, help="Print detailed output")
@click.option("-d", "--dry", is_flag=True, help="Don't save or modify files")
@click.option("--dem", type=click.Path(exists=True, path_type=Path), default=None, help="Path to DEM file or folder to combine with DEMS_GROUND")
@click.option("--npar", type=int, default=None, help="Number of parallel processes (default: CPU count)")
def trackfinder(path, linear, verbose, dry, dem, npar) -> None:
    """Run trackfinder on a .moco file."""
    run_trackfinder(
        path=path,
        dem_path=dem,
        linear=linear,
        verbose=verbose,
        dry=dry,
        npar=npar
    )

@click.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--out", type=click.Path(path_type=Path), default=".", help="Output directory (default: '.')")
@click.option("-t", "--tag", type=str, default="", flag_value=None, help="Tag output .tomo directory with an extra string (default: latest slice processing date)")
@click.option("--single", is_flag=True, help="Process 1-look data in interferometric bands")
@click.option("--nopair", is_flag=True, help="Avoid processing 2-look data in interferometric bands")
@click.option("--RR", is_flag=True, help="Estimate RR and SSF in multilooked tomogram")
@click.option("--fused", is_flag=True, help="Process only fused tomograms")
@click.option("--sub", is_flag=True, help="Process only subsurface tomograms")
@click.option("--sup", is_flag=True, help="Process only supersurface tomograms")
@click.option("--canopy", is_flag=True, help="Process only canopy tomograms")
@click.option("--phh", is_flag=True, help="Only process files from P-band")
@click.option("--lxx", is_flag=True, help="Only process files from L-band")
@click.option("--lhh", is_flag=True, help="Only process L-band files with HH-pol")
@click.option("--lvv", is_flag=True, help="Only process L-band files with VV-pol")
@click.option("--lhv", is_flag=True, help="Only process L-band files with HV-pol")
@click.option("--lvh", is_flag=True, help="Only process L-band files with VH-pol")
@click.option("--cvv", is_flag=True, help="Only process files from C-band")
@click.option("--load", is_flag=True, help="Load generated tomogram scenes into an interactive Python console")
@click.option("-m", "--masks", type=str, default="", help="Folder containing shapefile masks (in addition to TOMOMASKS)")
@click.option("-n", "--npar", type=int, default=os.cpu_count(), help="Number of parallel threads")
@click.option("--folder", type=str, default=None, help="Filter all files not in the provided folder")
@click.option("-d", "--date", type=str, default=None, help="Filter all files where the flight date does not match")
@click.option("-t", "--time", type=str, default=None, help="Filter all files where the flight time does not match")
@click.option("-s", "--spiral", type=int, default=None, help="Filter all files where the spiral ID does not match")
@click.option("-w", "--width", type=float, default=None, help="Filter all files where the processed width does not match")
@click.option("-r", "--res", type=float, default=None, help="Filter all files where the processing resolution does not match")
@click.option("-f", "--refr", type=float, default=None, help="Filter all files where the refractive index does not match")
@click.option("--lat", type=float, default=None, help="Filter all files where the central latitude does not match")
@click.option("--lon", type=float, default=None, help="Filter all files where the central longitude does not match")
@click.option("--thresh", type=float, default=None, help="Filter all files where the processing threshold does not match")
@click.option("--smo", type=float, default=None, help="Filter all files where the smoothing parameter does not match")
@click.option("--ham", type=float, default=None, help="Filter all files where the Hamming window parameter does not match")
@click.option("--squint", type=float, default=None, help="Filter all files where the squint parameter does not match")
@click.option("--text", type=str, default=None, help="Filter all files which do not contain a matching text tag")
@click.option("--DC", type=float, default=None, help="Filter all files where DC parameter does not match")
@click.option("--DL", type=float, default=None, help="Filter all files where the DL parameter does not match")
@click.option("--HC", type=float, default=None, help="Filter all files where the HC parameter does not match")
@click.option("--HV", type=float, default=None, help="Filter all files where the HV parameter does not match")
def forge(paths, single, nopair, RR, fused, sub, sup, canopy,
         phh, lxx, lhh, lvv, lhv, lvh, cvv, load,
         out, masks, npar, folder, date, time, spiral, width, res, refr,
         lat, lon, thresh, smo, ham, squint, text, DC, DL, HC, HV) -> TomoScenes:

    time_start = Time.time()

    print("Input paths:", paths)
    print("Output directory:", out)
    print("Mask directory:", masks)
    print("Parallel threads:", npar)

    # Construct filter
    folder = os.path.abspath(folder) if folder else None
    date_obj = datetime.strptime(date, "%Y-%m-%d") if date else datetime.strptime("1900-01-01", "%Y-%m-%d")
    if time:
        timestamp = datetime.strptime(time, "%H:%M:%S")
        date_obj = date_obj.replace(hour=timestamp.hour, minute=timestamp.minute, second=timestamp.second)

    bands = []
    if phh: bands.append("phh")
    if lxx: bands.extend(["lhh", "lvv", "lhv", "lvh"])
    if lhh: bands.append("lhh")
    if lvv: bands.append("lvv")
    if lhv: bands.append("lhv")
    if lvh: bands.append("lvh")
    if cvv: bands.append("cvv")

    filter = ImageInfo(
        folder=folder, filename=None, date=date_obj, spiral=spiral, band=bands,
        width=width, res=res, smo=smo, ham=ham, refr=refr, lat=lat, lon=lon,
        hoff=None, depth=None, DC=DC, DL=DL, HC=HC, HV=HV, thresh=thresh,
        squint=squint, text=text
    )

    # Dispatch processing
    scenes = tomoforge(
        paths=paths, filter=filter, single=single, nopair=nopair, RR=RR,
        fused=fused, sub=sub, sup=sup, canopy=canopy,
        masks=masks, npar=npar, out=out
    )

    print(f"Processing completed in {Time.time() - time_start:.2f} seconds.")
    if load:
        interactive_console({"scenes": scenes})