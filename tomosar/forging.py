# Imports
import os
import re
from datetime import datetime, timezone
from collections import defaultdict
import math
import json
import pandas as pd
from pathlib import Path

from .utils import warn
from .core import ImageInfo, SliceInfo, TomoInfo, TomoScene, TomoScenes, regroup
from .apperture import SARModel

# Configuration constants
DB0_1M2 = 5 * 10**3.75     # Raw backscatter corresponding to 1 dB across 1 meter squared
TARGET_RES = 0.2           # Target resolution in meter for multilooked tomograms
SIGMA_XI = 0.9             # Sigma_xi range for filtering
FILTER_SIZE = 9            # Filter size
POINT_PERCENTILE = 98.0    # Percentile for identifying potential point targets
POINT_THRESHOLD = 9        # Threshold voxel count for identifying point targets

def recursive_search(paths: str|Path|list[str|Path], filter: ImageInfo = None) -> tuple[SliceInfo, list[Path], list[Path]]:
    """
    Recursively search for complex .tif files by calling sliceinfo on each subdirectory.
    Returns a list of sliceInfo dictionaries.
    """
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    else:
        paths = [Path(path) for path in paths]

    slice_info = SliceInfo()
    flight_infos = {}
    moco_cuts = {}

    timestamp_spiral_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-(\d{2})")
    timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}")

    print("\nScanning directories ...")

    for root_path in paths:
        info = SliceInfo.scan(path=root_path, read=False, filter=filter)
        if info:
            slice_info.extend(info)
        for dirpath in root_path.rglob("*"):
            if dirpath.is_dir():
                print(dirpath)

                for file in dirpath.iterdir():
                    if file.is_file():

                        if 'spiral.moco_cut' in file.name:
                            match = timestamp_spiral_pattern.search(file.name)
                            if match:
                                try:
                                    dt = datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")
                                    spiral_id = int(match.group(2))
                                    key = (dt, spiral_id)
                                    moco_cuts[key] = file
                                except ValueError:
                                    pass

                        elif 'flight_info.json' in file.name:
                            match = timestamp_pattern.search(file.name)
                            if match:
                                try:
                                    dt = datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")
                                    key = dt  # No spiral_id here
                                    flight_infos[key] = file
                                except ValueError:
                                    pass

                info = SliceInfo.scan(str(dirpath), read=False, filter=filter)
                if info:
                    slice_info.extend(info)

    return slice_info, flight_infos, moco_cuts

def find_pairs(band_groups, single=False) -> defaultdict[SliceInfo]:
    """
    Find pairs of slices in the band groups for interferometric processing.
    Returns a dictionary of new band groups with paired slices.
    """
    def make_key(slice) -> tuple:
        return tuple(slice.get(k) for k in ImageInfo.PAIR_PARAMETERS)

    new_band_groups = defaultdict(SliceInfo)

    # Pairing logic for each interferometric band
    for band1, band0, composite_band in [('phh1', 'phh0', 'phh'), ('cvv1', 'cvv0', 'cvv')]:
        group_map = defaultdict(lambda: {'1': [], '0': []})

        for s in band_groups.get(band1, SliceInfo()):
            group_map[make_key(s)]['1'].append(s)
        for s in band_groups.get(band0, SliceInfo()):
            group_map[make_key(s)]['0'].append(s)

        for group in group_map.values():
            for s1, s0 in zip(group['1'], group['0']):
                if s1 and s0:
                    composite_slice = s1.compose(s0)
                    new_band_groups[composite_band].append(composite_slice)

    # Include other bands
    for band, slices in band_groups.items():
        if band not in ['phh1', 'phh0', 'cvv1', 'cvv0'] or single:
            new_band_groups[band].extend(slices)

    return new_band_groups

def generate_tomograms(band_groups, flight_infos, moco_cuts, 
                   sub=False, sup=False, canopy=False, fused=False, npar: int = os.cpu_count(), 
                   RR: bool = True, masks: str = "", tag: str = "") -> TomoScenes:
    """
    Find and process tomograms based on the provided band groups and flags.
    Returns a TomoList of tomogram information.
    """

    tomo_scenes = []
    scenes = regroup(band_groups, ['date','spiral'])
    print(f"{len(scenes)} tomographic scene(s) detected.")
    for key, scene in scenes.items():
        tomo_scene = TomoScene()

        # Extract meta data
        tomo_scene.date = key[0]
        tomo_scene.spiral = key[1]
        # today = datetime.now().isoformat(timespec='seconds')
        print(f"{tomo_scene.date.isoformat(timespec='seconds')} Spiral ID:{tomo_scene.spiral}")
        linux_time = []
        # Loop through bands
        for band, group in scene.items():
            tomo_slices = group.tomograms()
            if len(tomo_slices) > 1:
                warn(f"Multiple tomograms detected in {band}. Selecting first found.")
            else:
                print(f"Band: {band}.")
            tomo_slices = tomo_slices[0]

            # Read slices
            tomo_slices.read(db0=DB0_1M2, npar=npar)
            
            # Calculate multilooking factor 
            ml_factor = max(math.ceil(TARGET_RES / tomo_slices[0].res), 2)

            # Find latest slice processing time 
            linux_time.append(max(tomo_slices.get('linuxTime')))
            
            # Forge tomogram
            tomo_scene[band] = TomoInfo.forge(tomo_slices, multilook=ml_factor, sigma_xi=SIGMA_XI, filter_size=FILTER_SIZE,
                                    point_percentile=POINT_PERCENTILE, point_threshold=POINT_THRESHOLD,
                                    fused=fused, sub=sub, sup=sup, canopy=canopy, npar=npar, RR=RR, masks=masks)

        if tag is None:
            processed = datetime.fromtimestamp(max(linux_time), tz=timezone.utc)
            tag = "P" + processed.strftime("%Y%m%d")

        # Form ID
        tomo_scene.id = f"{tomo_scene.date.strftime("%Y-%m-%d-%H-%M-%S")}-{tomo_scene.spiral:02}-{tag}"

        # Look for matching flight_info files
        if key[0] in flight_infos:
            try:
                with open(flight_infos[key[0]], 'r') as f:
                    data = json.load(f)
                    tomo_scene.info = data['Spirals'][key[1]]
            except (KeyError, TypeError) as e:
                warn(f"Spiral ID {key[1]} not found in flight info for {key[0]}: {e}")
            except json.JSONDecodeError as e:
                warn(f"Could not parse JSON in {flight_infos[key[0]]}: {e}")
            except FileNotFoundError:
                warn(f"File containing flight info not found: {flight_infos[key[0]]}")
        else:
            warn("No file containing flight info found.")


        # Look for matching binned_sar files
        if key in moco_cuts:
            try:
                tomo_scene.moco = pd.read_csv(moco_cuts[key])
                tomo_scene._model = SARModel(tomo_scene.moco)
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                warn(f"Could not parse CSV in {moco_cuts[key]}: {e}")
            except FileNotFoundError:
                warn(f"File containing moco cut not found: {moco_cuts[key]}")
            except ValueError:
                warn(f"Missing data in moco cut.")
        else:
            warn("No file containing moco cut found.")
        
        tomo_scenes.append(tomo_scene)

    tomo_scenes = TomoScenes(tomo_scenes)

    return tomo_scenes

def tomoforge(*,paths: str|Path | list[str|Path] = ".", filter: ImageInfo = None, 
                single: bool = False, nopair: bool = False, RR: bool = False,
                fused: bool = False, sub: bool = False, sup: bool = False, canopy: bool = False,
                masks: str = None, npar: int = os.cpu_count(), out: str = ".", tag: str = "") -> TomoScenes:
    """
    Processes tomographic data from .srf or complex .tif files.

    Parameters:
        paths (list): List of input paths to process.
        Optional flags controlling processing behavior.
        masks (str or list): Path(s) to mask files or folders.
        npar (int): Number of parallel threads to use.
        out (str): Output directory path.

    Returns:
        None
    """

    # Process first category found if not specified (in order specified below)
    if not fused and not sub and not sup and not canopy:
        fused = True
        sub = True
        sup = True
        canopy = True

    # Search for complex .tif files in the provided paths recursively
    slice_info, flight_infos, moco_cuts = recursive_search(paths, filter=filter)

    # If no slices are found exit early, else print the number of slices found
    if not slice_info:
        print("No slices found. Exiting.")
        return
    else:
        print(f"\n{len(slice_info)} slices found,", end=' ')
        # Check for unique slices
        slice_info = slice_info.unique()
        print(f"of which {len(slice_info)} are unique.")

    # Group slices by band (and antenna)
    band_groups = slice_info.group('band')

    # Print the number of slices per band
    for band, slices in band_groups.items():
        # Print the number of slices
        print(f"\t{len(slices)} slices in {band} band")
    
    # Find pairs of slices unless the nopair flag is set
    if not nopair:
        band_groups = find_pairs(band_groups, single=single)
        print("Forming ...")
        for band, slices in band_groups.items():
            print(f"\t{len(slices)} slices in {band} band") if band in ['phh', 'cvv'] else None

    # Tomographic processing
    tomo_scenes = generate_tomograms(band_groups, flight_infos=flight_infos, moco_cuts=moco_cuts, tag=tag,
                                 sub=sub, sup=sup, canopy=canopy, fused=fused, npar=npar, RR=RR, masks=masks)

    # Save results
    tomo_scenes.save(out)

    return tomo_scenes
