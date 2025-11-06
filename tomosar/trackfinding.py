import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
from datetime import datetime, timedelta
from pyproj import Transformer
from scipy.optimize import minimize
from pathlib import Path
import math
import re
from collections import defaultdict, Counter
import json
from matplotlib.figure import Figure

from .utils import find_inliers, format_duration, add_meta
from .binaries import elevation
from .apperture import SARModel
from .config import Frequencies
FREQUENCIES = Frequencies()

# Find flights
def find_flights(imu_log: pd.DataFrame):
    # Parameters
    minimum_flight_alt = 30  # meters
    minimum_flight_dur = timedelta(minutes=1)
    minimum_boot_dur = timedelta(minutes=1)
    tol = 0.1  # tolerance for derivative of altitude

    # Time step and window size
    time_step = imu_log["% GPST (s)"].iloc[1] - imu_log["% GPST (s)"].iloc[0]
    step = int(timedelta(seconds=10).total_seconds() / time_step)
    window_size = np.ones(step) / step

    # Filter altitude signal
    alt = np.convolve(imu_log["alt (m)"], window_size, mode='same')
    da = np.diff(alt) / time_step
    idx = np.where(np.abs(da) < tol)[0] # Approximately constant altitude

    # Split into segments of approximately constant altitude
    split_points = np.where(np.diff(idx) > 1)[0]
    split_points = np.concatenate(([0], split_points, [len(idx)]))
    segments = [idx[split_points[i]+1:split_points[i+1]] for i in range(len(split_points)-1)]

    # Find boot sequence
    durations = [len(seg) * time_step for seg in segments]
    boot_sequence = next((i for i, dur in enumerate(durations) if dur > minimum_boot_dur.total_seconds()), None)
    if boot_sequence is None:
        return [], time_step, window_size

    ground_alt = imu_log["alt (m)"].iloc[segments[boot_sequence]].mean()
    imu_log = imu_log.iloc[segments[boot_sequence][0]:].reset_index(drop=True)

    # Identify flight segments
    idx = imu_log.index[imu_log["alt (m)"] > ground_alt + minimum_flight_alt]
    split_points = np.where(np.diff(idx) > 1)[0]
    split_points = np.concatenate(([0], split_points, [len(idx)]))
    segments = [idx[split_points[i]+1:split_points[i+1]] for i in range(len(split_points)-1)]

    # Extract flights
    flights = [imu_log.iloc[seg] for seg in segments if not seg.empty]

    # Remove spurious flights
    durations = [flight["% GPST (s)"].iloc[-1] - flight["% GPST (s)"].iloc[0] for flight in flights]
    flights = [flight for flight, dur in zip(flights, durations) if dur > minimum_flight_dur.total_seconds()]

    return flights, time_step, window_size

# Classify flights
def classify_flights(flights):
    required_turns = 2
    spiral_flights = []
    linear_flights = []

    for flight in flights:
        dif = _yaw_dif(flight)
        if dif > required_turns * 2 * np.pi:
            spiral_flights.append(flight)
        else:
            linear_flights.append(flight)

    return [('Spiral', flight) for flight in spiral_flights] + [('Linear', flight) for flight in linear_flights]

def _yaw(flight):
    # Convert heading from degrees to radians and unwrap
    return np.unwrap(np.pi * flight["heading (deg)"] / 180)

def _yaw_dif(flight):
    y = _yaw(flight)
    return np.max(y) - np.min(y)

# Refine flights
def refine_flights(flights, time_step, window_size, npar: int = os.cpu_count()):

    with Pool(processes=npar) as pool:
        results = pool.map(_refine, [
            (tagged_flight, time_step, window_size) for tagged_flight in flights
            ])
        
    return [(tag, flight, track) for tag, flight, track in results if 
            (isinstance(track, pd.DataFrame) and not track.empty) or
            (isinstance(track, list) and track)
        ]

def _refine(args):
    tagged_flight, time_step, window_size = args
    tag = tagged_flight[0]
    flight = tagged_flight[1]
    if tag == 'Linear':
        return tag, flight, _refine_linear(flight, time_step=time_step)
    if tag == 'Spiral':
        return tag, flight, _refine_spiral(flight, time_step=time_step, window_size=window_size)
    
## Refine spiral
def _refine_spiral(flight, time_step, window_size):
    tol = 3e-3  # tolerance for second derivative of yaw

    # Step 1: Get yaw and smooth it
    y = _yaw(flight)
    y = np.convolve(y, window_size, mode='full')[:len(y)]

    # Step 2: First and second derivatives
    dy = np.gradient(y, time_step)
    dy = np.convolve(dy, window_size, mode='full')[:len(dy)]
    ddy = np.gradient(dy, time_step)

    # Step 3: Find indices with low second derivative
    idx = np.where(np.abs(ddy) < tol)[0]

    # Step 4: Split into segments
    split_points = np.where(np.diff(idx) > 1)[0]
    split_points = np.concatenate(([0], split_points, [len(idx)]))
    segments = [idx[split_points[i]+1:split_points[i+1]] for i in range(len(split_points)-1)]

    # Step 5: Find longest segment
    if not segments:
        return pd.DataFrame() # Empty if no valid segment

    longest = max(segments, key=len)

    # Step 6: Cut log for longest segment
    ext = 2 * len(window_size)
    start = max(0, longest[0] - ext)
    end = longest[-1]
    segment = flight.iloc[start:end + 1].copy()

    # Step 7: Estimate azimuth
    az, _, _, _ = _get_azimuth(segment)
    daz = np.gradient(az) / np.gradient(segment['% GPST (s)'])

    # Step 8: Find change points in azimuth derivative
    inliers = find_inliers(daz, min_samples=0.9, relative_threshold=0.2)
    #inliers = _extend_indices(inliers, len(segment), time_step, extension=1)

    # Step 9: Final cut
    refined_segment = segment.iloc[inliers].copy()
    return refined_segment

def _get_azimuth(spiral_track):
    # Get lat/lon coordinates for conversion to UTM coordinates
    lat = spiral_track["lat (deg)"].values
    lon = spiral_track["lon (deg)"].values

    # Use pyproj to convert to UTM (auto zone detection could be added)
    transformer = Transformer.from_crs("epsg:4326", "epsg:32633", always_xy=True)
    x, y = transformer.transform(lon, lat)
    coords = np.column_stack((x, y))

    # Initial guess: centroid
    initial_center = coords.mean(axis=0)

    # Objective function: deviation from linear radius vs angle
    def spiral_error(center):
        dx = coords[:, 0] - center[0]
        dy = coords[:, 1] - center[1]
        radius = np.sqrt(dx**2 + dy**2)
        angle = np.unwrap(np.arctan2(dy, dx))
        p = np.polyfit(angle, radius, 1)
        fit_radius = np.polyval(p, angle)
        return np.mean((radius - fit_radius)**2)

    result = minimize(spiral_error, initial_center, method='Nelder-Mead')
    optimal_center = result.x

    # Compute radius and azimuth
    dx = coords[:, 0] - optimal_center[0]
    dy = coords[:, 1] - optimal_center[1]
    r = np.sqrt(dx**2 + dy**2)
    az = np.unwrap(np.arctan2(dx, dy)) * 180 / np.pi  # degrees

    # Convert center back to lat/lon
    transformer_back = Transformer.from_crs("epsg:32633", "epsg:4326", always_xy=True)
    lon0, lat0 = transformer_back.transform(optimal_center[0], optimal_center[1])

    return az, r, lat0, lon0

## Refine linear
def _refine_linear(flight, time_step) -> list[pd.DataFrame]:
    tol_const = 1.1
    tol_par = 0.01
    min_flight_time = 5  # seconds

    # Calculate heading
    heading = np.arctan2(flight['vn (m/s)'], flight['ve (m/s)'])
    heading = np.unwrap(heading)

    # Derivative of heading
    dh = np.diff(heading) / time_step
    idx = np.where(np.abs(dh) < tol_const)[0]

    # Split into segments
    split_points = np.where(np.diff(idx) > 1)[0]
    segments = [idx[i+1:j+1] for i, j in zip([0]+split_points.tolist(), split_points.tolist()+[len(idx)-1])]

    # Filter short segments
    segments = [seg for seg in segments if len(seg) * time_step >= min_flight_time]

    # Remove first segment, corresponding to the drone flight to mission
    if len(segments) > 2:
        segments = segments[1:]

    # Compute mean heading for each segment
    segment_headings = [np.mean(heading[seg]) for seg in segments]

    # Find longest segment
    lengths = [len(seg) for seg in segments]
    if not lengths:
        return []
    longest_idx = np.argmax(lengths)
    ref_heading = segment_headings[longest_idx]

    # Filter segments that are approximately parallel
    parallel_segments = [seg for seg, h in zip(segments, segment_headings) if abs(h - ref_heading) < tol_par
                            or abs(h - (ref_heading + np.pi)) < tol_par]

    # Extract refined tracks
    if parallel_segments:
        tracks = [flight.iloc[seg] for seg in parallel_segments]

    return tracks

# Plot results
def plot_tracks(tracks: tuple[str, pd.DataFrame, pd.DataFrame] | tuple[str, pd.DataFrame, list[pd.DataFrame]],
                path: Path = '.', dry: bool = False):
    path=Path(path)
    n = len(tracks)
    n_spirals = 0
    n_linear = 0
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
    axs = axs.flatten()
    for i, pack in enumerate(tracks):
        tag = pack[0]
        flight = pack[1]
        track = pack[2]
        if tag == 'Spiral':
            n_spirals += 1
            axs[i].plot(flight['lon (deg)'], flight['lat (deg)'], 'r', label="Full flights" if i==0 else None)
            axs[i].plot(track['lon (deg)'], track['lat (deg)'], 'g', label="Tracks found" if i==0 else None)
            axs[i].set_xlabel("lon (deg)")
            axs[i].set_ylabel("lat (deg)")
            axs[i].set_title(f"Spiral {n_spirals}")
        if tag == 'Linear':
            n_linear += 1
            axs[i].plot(flight['lon (deg)'], flight['lat (deg)'], 'r', label="Full flights" if i==0 else None)
            for j, tr in enumerate(track):
                axs[i].plot(tr['lon (deg)'], tr['lat (deg)'], 'g', label="Tracks found" if i==0
                            and j==0 else None)
            axs[i].set_xlabel("lon (deg)")
            axs[i].set_ylabel("lat (deg)")
            axs[i].set_title(f"Linear {n_linear}: {len(track)} tracks")
    for i in range(n,len(axs)):
       axs[i].axis('off')

    # Collect handles and labels
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=len(labels), bbox_to_anchor=(0.5, 1))
    fig.canvas.manager.set_window_title("Trackfinder results")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if dry:
        print("Showing tracks ...", end=" ", flush=True)
        plt.show()
        print("done.")
    else:
        fig_name = path.with_name(path.stem + "_trackfinder_results.pdf")
        fig.savefig(fig_name, format="pdf")
        print(f"Trackfinder results saved to {fig_name}")

# Analyze tracks
def analyze_tracks(tracks, flight_info, base_ele, dem_path, npar: int = os.cpu_count()):
    spiral_tracks = {}
    linear_tracks = {}
    with Manager() as manager:
        counters = manager.dict()
        lock = manager.Lock()

        with Pool(processes=npar) as pool:
            results = pool.starmap(_analyze, [
                (tagged_track, flight_info, base_ele, dem_path, counters, lock) for tagged_track in tracks
            ])
            
            for tag, i, track, updated_info in results:
                if tag == 'Spiral':
                    spiral_tracks[i] = track
                    flight_info['Spirals'][i] = updated_info
                if tag == 'Linear':
                    linear_tracks[i] = track
                    flight_info[f'Linear_{i}'] = updated_info
    
    return spiral_tracks, linear_tracks

def _analyze(tagged_track, flight_info, base_ele, dem_path, counters, lock):
    tag = tagged_track[0]
    track = tagged_track[2]
    with lock:
        counters[tag] = counters.get(tag, 0) + 1
        i = counters[tag]
    if tag == 'Spiral':
        track_info = flight_info['Spirals'][i]
        track, updated_info = analyze_spiral(track, track_info, base_ele, dem_path)
    if tag == 'Linear':
        track_info = flight_info[f'Linear_{i}']
        track, updated_info = analyze_linear(track, track_info, base_ele)
    return (tag, i, track, updated_info)

## Analyze spirals
def analyze_spiral(track, track_info, base_ele, dem_path):
    # Correct time stamps
    track_info['t_start'] = format_duration(track['% GPST (s)'].iloc[0])
    track_info['t_end'] = format_duration(track['% GPST (s)'].iloc[-1])
    track_info['duration (s)'] = track['% GPST (s)'].iloc[-1] - track['% GPST (s)'].iloc[0]
    # Calculate new variables and parameters
    az, r, lat0, lon0 = _get_azimuth(track)
    h0 = elevation(lat0, lon0, dem_path)
    flight_alt = track['alt (m)'].to_numpy() - h0 # Flight altitude relative center point
    # Add variables to track data
    track['radius (m)'] = r
    track['azimuth (deg)'] = az
    track['flight_alt (m)'] = flight_alt
    # Add parameters to flight_info
    track_info['center_lat'] = lat0
    track_info['center_lon'] = lon0
    track_info['center_elevation (m)'] = h0
    track_info['base_altitude (m)'] = base_ele - h0 # Base altitude relative center point
    track_info['top_radius'] = round(r.min())
    track_info['bottom_radius'] = round(r.max())
    track_info['top_flight_altitude'] = round(flight_alt.max())
    track_info['bottom_flight_altitude'] = round(flight_alt.min())

    return track, track_info

## Analyze linear tracks
def analyze_linear(tracks, tracks_info, base_ele):
    for i, track in enumerate(tracks):
        track_info = tracks_info[i+1]
        # Correct time stamps
        track_info['t_start'] = format_duration(track['% GPST (s)'].iloc[0])
        track_info['t_end'] = format_duration(track['% GPST (s)'].iloc[-1])
        # Add track start and end positions
        track_info['lat_start'] = track['lat (deg)'].iloc[0]
        track_info['lon_start'] = track['lon (deg)'].iloc[0]
        track_info['lat_end'] = track['lat (deg)'].iloc[-1]
        track_info['lon_end'] = track['lon (deg)'].iloc[-1]
        # Add base elevation and flight altitude information relative base elevation
        track_info['base_elevation'] = base_ele
        flight_alt = track['alt (m)'].to_numpy() - base_ele
        yaw = _yaw(track)
        heading = np.unwrap(np.arctan2(track['vn (m/s)'], track['ve (m/s)']))
        track_info['flight_alt (m)'] = {
            'mean': np.mean(flight_alt),
            'std': np.std(flight_alt)
        }
        track_info['yaw (deg)'] = {
            'mean': np.mean(yaw),
            'std': np.std(yaw)
        }
        track_info['heading (deg)'] = {
            'mean': np.mean(heading),
            'std': np.std(heading)
        }
    return track, tracks_info

# Modify the radar[...].inf file
def modify_radar_inf(path: Path, info: dict, dry: bool = False):
    """
    Modify radar_logger_dat-[...].inf file with new track timestamps.
    
    Parameters:
        info: dict with start and end times as strings (dd:hh:mm:ss).
        path: Path to the imu_logger_dat-[...].moco file
    """
    t_start = []
    t_end = []
    for i, ts in info.items():
        if not isinstance(ts, dict):
            continue
        # print(json.dumps(ts))
        t_start.append(ts['t_start'])
        t_end.append(ts['t_start'])
    folder = path.parent
    date, timestamp = _extract_timestamp(path.name)
    base_name = f"radar_logger_dat-{date}-{timestamp}.inf"
    inf_path = folder / base_name

    # Try ±1 second if file not found
    def try_alternatives(ts):
        dt = datetime.strptime(ts, "%H-%M-%S")
        for delta in [1, -1]:
            new_ts = (dt + timedelta(seconds=delta)).strftime("%H-%M-%S")
            alt_name = f"radar_logger_dat-{date}-{new_ts}.inf"
            alt_path = folder / alt_name
            if alt_path.exists():
                return alt_path
        return None

    if not inf_path.exists():
        alt_path = try_alternatives(timestamp)
        if alt_path:
            inf_path = alt_path

    radar_inf = [
        "Flight", "Info", "{CH}", "================", "Number", "of", "tracks", "{i}", ":", str(len(t_start)),
        "time", "begin", "of", "track", "[hh:mm:ss]", "{s}", ":",
        "time", "end", "of", "track", "[hh:mm:ss]", "{s}", ":",
        "time", "shift", "of", "moco", "data", "{f}", ":", "0"
    ]

    if dry:
        print()
        print(" ".join(radar_inf[:3]))
        print(radar_inf[3])
        print()
        print(" ".join(radar_inf[4:8]) + "                    " + " ".join(radar_inf[8:10]))
        print(" ".join(radar_inf[10:16]) + "      " + radar_inf[16], end=" ")
        print(" ".join(t_start))
        print(" ".join(radar_inf[17:19]) + "   " + " ".join(radar_inf[19:23]) + "      " + radar_inf[23], end=" ")
        print(" ".join(t_end))
        print(" ".join(radar_inf[24:-2]) + "             " + " ".join(radar_inf[-2:]))
        print()
        return
    with open(inf_path, "w") as f:
        f.write("\n")
        f.write(" ".join(radar_inf[:3]) + "\n")
        f.write(radar_inf[3] + "\n")
        f.write("\n")
        f.write(" ".join(radar_inf[4:8]) + "                    " + " ".join(radar_inf[8:10]) + "\n")
        f.write(" ".join(radar_inf[10:16]) + "      " + radar_inf[16])
        f.write(" ".join(t_start) + "\n")
        f.write(" ".join(radar_inf[17:19]) + "   " + " ".join(radar_inf[19:23]) + "      " + radar_inf[23])
        f.write(" ".join(t_end) + "\n")
        f.write(" ".join(radar_inf[24:-2]) + "             " + " ".join(radar_inf[-2:]))

def _extract_timestamp(filename) -> tuple[str|None, str|None]:
    # Match pattern like 2025-09-02-18-10-42
    match = re.search(r'(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})', filename)
    if match:
        date = match.group(1)
        timestamp = match.group(2)
        return date, timestamp  
    return None, None

# Orchestrating functions
## trackfinder
def trackfinder(
        path: str|Path,
        dem_path: str|Path = None,
        linear: int = 0,
        verbose: bool = False,
        dry: bool = False,
        npar: int = os.cpu_count()
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:

    ## 1. Read file into DataFrame and get base altitude
    path=Path(path)

    # Get date and timestamp as strings for file naming
    date, timestamp = _extract_timestamp(path.name)
    base_path = path.with_name(f"{date}-{timestamp}")
    
    if not (path.is_file and path.suffix == '.moco'):
        raise ValueError("trackfinder must be called with a path to a .moco file")
    print("Reading .moco file ...", end=" ", flush=True)
    imu_log = pd.read_csv(path, sep='\t', skipinitialspace=True)
    base_ele = np.mean(imu_log['alt (m)'][0:10])
    print("done.")

    ## 2. Segment flights
    print("Segmenting flights ...", end=" ", flush=True)
    flights, time_step, window_size = find_flights(imu_log)

    ## 3. Classify flights
    flights = classify_flights(flights)
    tag_counts = Counter(tag for tag, _ in flights)

    ## 4. Refine tracks
    tracks = refine_flights(flights=flights, time_step=time_step, window_size=window_size, npar=npar)
    print("done.")

    ## 5. Plot tracks
    plot_tracks(tracks, path=base_path, dry=dry)

    ## 6. Get track timestamps
    flight_info = defaultdict(dict)
    if verbose:
        print() # Empty line
        result = f"{len(flights)} flights found: {tag_counts['Spiral']} spiral and {tag_counts['Linear']} linear."
        print(result)
    # Get spiral track timestamps
    n_spiral = 0
    n_linear = 0
    for tag, _, track in tracks:
        if tag == 'Spiral':
            n_spiral += 1
            t_start = track['% GPST (s)'].iloc[0]
            t_end = track['% GPST (s)'].iloc[-1]
            flight_info['Spirals'][n_spiral] = {'t_start': format_duration(t_start), 't_end': format_duration(t_end)}
        if tag == 'Linear':
            n_linear += 1
            for i, tr in enumerate(track):
                t_start = tr['% GPST (s)'].iloc[0]
                t_end = tr['% GPST (s)'].iloc[-1]
                flight_info[f'Linear_{n_linear}'][i+1] = {'t_start': format_duration(t_start), 't_end': format_duration(t_end)}
    
    # Modify radar_inf file
    if not dry:
        if linear == 0:
            modify_radar_inf(path, flight_info['Spirals'], dry=dry)
        elif linear:
            modify_radar_inf(path, flight_info[f'Linear_{linear}'], dry=dry)

    # 7. Perform rudimentary analysis of tracks
    print("Analyzing tracks ...", end=" ", flush=True)
    spiral_tracks, linear_tracks = analyze_tracks(tracks=tracks, flight_info=flight_info, base_ele=base_ele, dem_path=dem_path, npar=npar)
    print("done.")

    meta_str = "Altitude is counted relative the center coordinate (ground level)."
    flight_info['Spirals'] = add_meta(flight_info['Spirals'], meta_str)
    meta_str = "Altitude is counted relative the base position (take off)."
    for i in range(tag_counts['Linear']):
        flight_info[f'Linear_{i+1}'] = add_meta(flight_info[f'Linear_{i+1}'], meta_str)
    
    # 8. Print track info
    if verbose:
        print(json.dumps(flight_info, indent=4))
         
    # 9. File generation
    if not dry:
        # Save flight_info
        flight_info = add_meta(flight_info, result, '__flights__')
        fi_path = base_path.with_name(base_path.base + "_flight_info.json")
        with open(fi_path) as f:
            json.dump(flight_info, f, indent=4)
        print(f"Information about tracks saved to {fi_path}")

        # Save .moco cuts:
        counter = Counter()
        print(f"Saving .moco cuts:")
        for i, track in spiral_tracks:
            dst = base_path.with_name(base_path.base + f"-{i:02}_spiral.moco_cut")
            print(dst)
            track.to_csv(dst, index=False)
    
            # Save linear track .moco cuts
        for i, tracks in linear_tracks:
            for j, track in enumerate(tracks):
                dst = base_path.with_name(base_path.base + f"-{j:02}_linear_{i}.moco_cut")
                print(dst)
                track.to_csv(dst, index=False)
            
    print("All done.")
    return spiral_tracks, linear_tracks

## Model spiral tracks
def model_spirals(tracks, path, dry, verbose, npar: int = os.cpu_count()):
    with Pool(processes=npar) as pool:
        results = pool.starmap(_model, [(i, track, dry) for i, track in tracks.items()])

        for i, fig, evaluation in sorted(results, key=lambda x: x[0]):
            if verbose:
                print(f"Spiral {i}:", end=" ", flush=True)
                print(json.dumps(evaluation, indent=4))
            if not dry:
                fig_path = path.with_name(path.stem + f"-{i:02}_spiral_model.pdf")
                eval_path = fig_path.with_suffix(".json")
                fig.savefig(fig_path, format="pdf")
                with open(eval_path, 'w') as dst:
                    json.dump(evaluation, dst, indent=4)
                print(f"Model evaluation for Spiral {i} saved to {fig_path} and {eval_path}")

def _model(i: int, track: pd.DataFrame, dry: bool = False) -> tuple[int, Figure, defaultdict[dict]]:
    model = SARModel(track)
    fig, evaluation = model.evaluate()
    try:
        fig.canvas.manager.set_window_title(f"SAR parameters: Spiral {i}")
    except Exception:
        pass
    if dry:
        if i == 1:
            print("Showing model plots ...", end=" ", flush=True)
        plt.show()
        if i == 1:
            print("done.")

    return i, fig, evaluation
