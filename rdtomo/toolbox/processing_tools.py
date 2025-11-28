import click
from pathlib import Path
import os
import time as Time
from datetime import datetime, date

from ..gnss import fetch_swepos as run_fetch_swepos, station_ppp as run_station_ppp, reachz2rnx, generate_mocoref
from ..trackfinding import trackfinder as run_trackfinder
from .. import ImageInfo, TomoScenes, Settings
from ..utils import interactive_console, local
from ..forging import tomoforge
from ..data import LoadDir, DataDir, ProcessingDir

@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=LoadDir), default=LoadDir.cwd())
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
@click.option("-f", "--force", is_flag=True, help="Force generation of processing directory (this may overwrite existing directories, but has no effect if PATH is a processing directory)")
@click.option("--dry", is_flag=True, help="Force the specified PATH to be interpreted as a processing directory")
@click.option("-t", "--tag", default = "", flag_value=date.today().strftime('%Y%m%d'), help="Tag processing directory with specified string (default: the date of today)")
def init(
    path: LoadDir,
    force: bool,
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
    dry: bool,
    use_broadcast: bool,
    tag: str,
    config: Path | None,
    atx: Path | None,
    receiver: Path | None,
    downloads: int,
    attempts: int,
    elevation_mask: float|None,
    csv_line: int,
    offset: float,
    minimal_overlap: float,
) -> None:
    """Searches recursively from the PATH (default: CWD) to find matching files:
    (1) Drone GNSS .bin and .log;
    (2) Drone IMU .bin and .log;
    (3) Drone Radar .bin, .log and .cfg;
    (4) GNSS base station;
    (5) Mocoref data or precise position of GNSS base station; and
    (6) Data files for PPP and precise mode RTKP post processing.
    
    If the GNSS base station file is missing, tomosar init can fetch files from the nearest Swepos station,
    or supplement Mocoref data by performing static PPP on the base station.
    
    Note that the path must point to a directory which contains exactly one set of drone data.
    For other files, tomosar init will use the first matching file it finds. For the GNSS base station a RINEX OBS file is
    prioritized over other files: HCN files and RTCM3 files are also accepted, as well as Reach ZIP archives. For mocoref data
    a mocoref.moco file is prioritized followed by a JSON file, with the underlying assumption that these
    have been generated from raw mocoref data; then a LLH log is prioritized over a CSV file. If a Reach ZIP archive is
    used as the source of the GNSS base station file, the mocoref file will also be generated from there. 
    
    The files are converted where applicable and copied/moved into a processing directory, in such a way that the content
    of the data directory where tomosar init was initiated is left unaltered. Then preprocessing is initiated [ONLY GNSS IMPLEMENTED].
    By default tomosar init will use precise ephemeris data for the RTKP post processing, and will download this data if not
    available (disable by running with --broadcast)
    
    Note that tomosar init can also be run inside a processing directory, in which case it simply initiates preprocessing [ONLY GNSS IMPLEMENTED].
    Any directory inside the settings specified PROCESSING_DIRS is assumed to be a processing directory, and any directory
    outside is by default assumed to be a data directory (this behaviour can be overridden by the --processing option).
    
    Use --tag to append the date the processing folder was initiated to the folder name (otherwise copied from the data directory), or --tag=STRING
    to append some other tag."""

    print(f"Type of {path}: {type(path)}")
    if not isinstance(path, (DataDir, ProcessingDir)):
        raise TypeError(f"You can only run rdtomo init on DataDir and ProcessingDir folders: {path} is a {type(path)}")
    
    
    if isinstance(path, DataDir):
        processing_dir = Settings().PROCESSING_DIRS / (path.name + "_" + tag)
        processing_dir.mkdir(exist_ok=force)

        path = path.init(
                processing_dir=processing_dir,
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
                dry=dry,
                rtkp_config=config,
            )
    # Verify that loading as Processing Directory was sucessful (failes on dry)
    if not isinstance(path, ProcessingDir):
        return
    
    path.init(
            config=config,
            atx=atx,
            receiver=receiver,
            use_precise=not use_broadcast,
            download_attempts=attempts,
            max_downloads=downloads,
            elevation_mask=elevation_mask,
            minimal_overlap=minimal_overlap
        )
    

@click.command()
@click.argument("archive", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", "output_dir", type=click.Path(exists=False, file_okay=False, path_type=Path), default=None, help="Extract files into given folder (default: parent of archive)")
@click.option("--rover", "rover_obs", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Target rover OBS for RTKP processing")
@click.option("-n", "--nav", "extract_nav", is_flag=True, help="Extract NAV file")
@click.option("--verbose", is_flag=True, help="Verbose mode.")
def extract_reach(archive: Path, output_dir: Path|None, rover_obs: Path|None, extract_nav: bool, verbose: bool) -> None:
    """Extracts a Reach ZIP archive to produce:
    (1) A RINEX OBS file for a single site,
    (2) A mocoref.moco for the OBS file, and
    (3) A RINEX NAV file (optional).
    
    Optionally takes a rover OBS file as input to extract from the archive the OBS file which has the greatest overlap with the
    input rover OBS. Otherwise extracts the longest segment."""

    reachz2rnx(archive=archive, output_dir=output_dir, rnx_file=rover_obs, nav=extract_nav, verbose=verbose)

@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("-d","--dry", is_flag=True, help="Do not generate file (displays result)")
@click.option("--csv", is_flag=True, help="File is a CSV file (default for .csv and .CSV files)")
@click.option("--json", is_flag=True, help="File is a JSON file (default for .json and .JSON files)")
@click.option("--llh", is_flag=True, help="File is an LLH log (default for .llh and .LLH files)")
@click.option("-l", "--line", type=int, default=1, help="Line in CSV file to read data from (default=1)")
@click.option("--offset", type=float, default=-0.079, help="Specify vertical PCO between mocoref data log receiver and drone processing receiver (default=-0.079) for CSV files")
def mocoref(path, dry, csv, json, llh, line, offset) -> None:
    """Generate a mocoref file from a data file"""
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
    _, mocoref_file = generate_mocoref(data=path, type=type, line=line, generate=not dry, verbose=dry, pco_offset=offset)
    print(f"Mocoref file {local(mocoref_file)} generated from {local(path)}")

@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--stations", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the SWEPOS coordinate list CSV")
@click.option("--downloads", type=int, default=10, help="Max number of parallel downloads (default: 10)")
@click.option("--attempts", type=int, default=3, help="Max number of attempts for each file (default: 3)")
@click.option("-o", "--output", type=click.Path(path_type=Path), default="SWEPOS", help="Output directory for SWEPOS RINEX files")
@click.option("-d", "--dry", is_flag=True, help="Dry run without downloads")
@click.option("--cont", is_flag=True, help="Continue run after downloads complete")
@click.option("-n","--nav", is_flag=True, help="Also fetch nav files.")
def fetch_swepos(path, stations, downloads, attempts, output, dry, cont, nav) -> None:
    """Extract GNSS info from rover gnss log or RINEX observation file and find nearest SWEPOS station.
    Then download files into output directory."""
    run_fetch_swepos(
        filepath=path,
        stations_path=stations,
        max_downloads=downloads,
        max_retries=attempts,
        dry=dry,
        output_dir=output,
        fetch_nav=nav,
        cont=cont
    )

@click.command()
@click.argument("obs_file", type=click.Path(exists=True, path_type=Path))
@click.option("-n", "--navglo", type=click.Path(exists=True, path_type=Path), default=None, help="Path to navigation data for GLONASS (can be general/merged NAV file)")
@click.option("-a", "--atx", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the satellite antenna .atx file")
@click.option("-r", "--receiver", type=click.Path(exists=True, path_type=Path), default=None, help="Path to the .atx file containing receiver antenna info")
@click.option("--downloads", type=int, default=10, help="Max number of parallel downloads (default: 10)")
@click.option("--attempts", type=int, default=3, help="Max number of attempts for each file (default: 3)")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=False, flag_value=True, help="Generate .out file at specified path (default: next to OBS file)")
@click.option("-d", "--dry", is_flag=True, help="Dry run without downloads")
@click.option("-x", "--no-header", 'header', is_flag=True, default=True, flag_value=False, help="Do not modify OBS file header with new position.")
@click.option("--retain", is_flag=True, help="Retain ephemeris data files after finishing")
@click.option("-m", "--mocoref", "make_mocoref", is_flag=True, help="Generate mocoref.moco file with position")
def station_ppp(
    obs_file: Path,
    navglo: Path|None,
    atx: Path|None,
    receiver: Path|None,
    downloads: int,
    attempts: int,
    output: Path|bool,
    dry: bool,
    header: bool,
    retain: bool,
    make_mocoref: bool
) -> None:
    """Attempt to determine base station position using PPP post processing."""
    if output == True:
        output = obs_file.with_suffix(".out")
    if output == False:
        output = None
    results = run_station_ppp(
        obs_path=obs_file,
        navglo_path=navglo,
        atx_path=atx,
        antrec_path=receiver,
        max_downloads=downloads,
        max_retries=attempts,
        dry=dry,
        out_path=output,
        header=header,
        retain=retain,
        mocoref=make_mocoref
    )

    print(f"Determined position:\n   lat: {results['lat']}\n   lon: {results['lon']}\n   h  : {results['h']}")

@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("-l", "--linear", type=int, default=0, help="Specify linear track index to modify radar-[...].inf (0 for spiral flights)")
@click.option("-v", "--verbose", is_flag=True, help="Print detailed output")
@click.option("-d", "--dry", is_flag=True, help="Don't save or modify files")
@click.option("--dem", type=click.Path(exists=True, path_type=Path), default=None, help="Path to DEM file or folder to combine with DEMS_GROUND")
@click.option("--npar", type=int, default=None, help="Number of parallel processes (default: CPU count)")
def trackfinder(path, linear, verbose, dry, dem, npar) -> None:
    """Run trackfinder on a .moco file containing integrated GNSS and IMU data."""
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
    """Forge slices into Tomogram Directories."""

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