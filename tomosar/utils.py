# Imports
import os
import subprocess
import shutil
import platform
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy
from scipy.special import polygamma
from scipy.stats import gamma
from scipy.linalg import svd
from scipy.optimize import least_squares
from scipy.ndimage import binary_closing
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.cluster import DBSCAN
import json
import inspect
from datetime import timedelta, datetime, date, time, timezone
import math
from collections import defaultdict
from ftplib import FTP, error_perm
from getpass import getpass
from pathlib import Path
import gzip
import code
import sys
import inspect
import hashlib

from .tomogram_processing import circularize
from .transformers import geo_to_ecef, ecef_to_geo
from .config import Settings

# Warning message
def warn(message) -> None:
    # Get the current stack
    stack = inspect.stack()

    # Get the caller (who called warn) and its parent (if available)
    caller_frame = stack[1]
    parent_frame = stack[2] if len(stack) > 2 else None

    # Extract info
    caller_func = caller_frame.function
    if parent_frame:
        filename = parent_frame.filename
        lineno = parent_frame.lineno
    else:
        filename = caller_frame.filename
        lineno = caller_frame.lineno

    # ANSI escape code for yellow text
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    print(f"{YELLOW}{filename}:{lineno} in {caller_func}(): {message}{RESET}", file=sys.stderr)

# Localize a path(s) if possible
def local(paths: list[Path|str]|Path|str, root: Path|str = '.') -> list[str] | str:
    """Returns a string representation of specified path(s) relative the root directory (default: CWD)"""
    def localize(path: Path|str, root: Path = Path.cwd()) -> str:
        if path is None:
            return None
        path = Path(path)
        try:
            return str(path.relative_to(root))
        except:
            return str(path)
    root = Path(root).resolve()
    if not isinstance(paths, list):
        return localize(paths, root)
    return [localize(p, root) for p in paths]

# Load interactive console
def interactive_console(var_dict: dict) -> None:
    pink = "\033[95m"
    reset = "\033[0m"
    bold = "\033[1m"
    bold_off = "\033[22m"

    sys.ps1 = f"{pink}>>> {reset}"
    sys.ps2 = f"{pink}... {reset}"

    print(f"{pink}{bold}Printing loaded variables ...{reset}")

    lines = [
        f"{pink}{bold}{name}:{bold_off} {value}{reset}"
        for name, value in var_dict.items()
    ]

    banner = "\n".join(lines)

    # Launch console with variables available
    code.interact(banner=banner, local=var_dict)

def drop_into_terminal(target_dir: str|Path) -> None:
    target_dir = os.path.abspath(target_dir)
    system = platform.system()

    if system == "Windows":
        # Launch a new cmd.exe window detached from Python
        subprocess.Popen(
            ["cmd.exe"],
            cwd=target_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )

    elif system == "Darwin":  # macOS
        # Use AppleScript to open Terminal.app in the target directory
        script = f'''
        tell application "Terminal"
            do script "cd '{target_dir}'; exec $SHELL -i"
            activate
        end tell
        '''
        subprocess.Popen(["osascript", "-e", script])

    elif system == "Linux":
        # Try common terminal emulators
        for terminal in ["gnome-terminal", "xfce4-terminal", "konsole", "xterm"]:
            if shutil.which(terminal):
                if terminal == "xterm":
                    subprocess.Popen([terminal, "-e", f"bash -i"], cwd=target_dir)
                else:
                    subprocess.Popen([
                        terminal,
                        "--working-directory", target_dir,
                        "--", "bash", "-i"
                    ])
                return
        raise RuntimeError("No supported terminal emulator found on Linux.")

    else:
        raise RuntimeError(f"Unsupported platform: {system}")
        
# Hashing
def changed(hash_file: Path|str, input: list[Path|str]|Path|str, generate_hash: bool = True) -> bool:
    """Generates hash from input and compare against hash stored in hash file.
    Updates hash in hash file if a change was found."""
    def generate_hash(input: list) -> str:
        hasher = hashlib.sha256()
        for path in sorted(input):  # sort to ensure consistent order
            try:
                full_path = str(Path(path).resolve())
                stat = os.stat(full_path)
                hasher.update(full_path.encode())
                hasher.update(str(stat.st_mtime).encode())
                hasher.update(str(stat.st_size).encode())
            except FileNotFoundError:
                continue 
        return hasher.hexdigest()
    
    if isinstance(input, Path|str):
        input = [input]

    new_hash = generate_hash(input)

    # Compare against previous hash
    hash_file = Path(hash_file)
    if hash_file.exists():   
        # Read hash file
        with open(hash_file, 'r') as src:
            old_hash = src.read()

        # Compare
        if new_hash == old_hash:
            return False
        
    # Update hash
    if generate_hash:
        with open(hash_file, 'w') as dst:
            dst.write(new_hash)

    return generate_hash

# Transformation to local ENU frame from ECEF
def ecef2enu(lat: float, lon: float) -> np.ndarray:
    lon = np.radians(lon)
    lat = np.radians(lat)
    return np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.cos(lon)*np.sin(lat), -np.sin(lon)*np.sin(lat), np.cos(lat)],
        [np.cos(lon)*np.cos(lat), np.sin(lon)*np.cos(lat), np.sin(lat)]
    ])

# Find change points in linear statistics
def find_inliers(signal, min_samples: int|float = 0.5, residual_threshold: float|None = None,
                 relative_threshold: float|None=0.2):
    n = len(signal)
    x = np.arange(n).reshape(-1,1)

    # Use RANSAC algorithm to estimate straight line
    ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=min_samples,
                             residual_threshold=residual_threshold)
    ransac.fit(x, signal)
    predictions = ransac.predict(x)
   
    if relative_threshold:
        # Calculate inliers from relative threshold
        residuals = np.abs(signal - predictions)
        relative_residuals = residuals / np.abs(signal)
        inlier_mask = relative_residuals < relative_threshold
    else:
        # Get inliers from the RANSAC algorithm
        inlier_mask = ransac.inlier_mask_

    # Close small gaps
    inlier_mask = binary_closing(inlier_mask, structure=np.ones(3))

    return np.where(inlier_mask)[0]

# Statistics
def apply_variable_descriptions(df: pd.DataFrame):
    df.attrs["VariableUnits"] = {
        "height": "m",
        "mean_backscatter": "dB",
        "SD": "dB",
        "contrast": "dB"
    }

    df.attrs["VariableDescriptions"] = {
        "mean_backscatter": "Mean logarithmic backscatter.",
        "SD": "Standard deviation of logarithmic backscatter.",
        "contrast": "Logarithmic backscatter contrast.",
        "E": "Entropy of intensity image."
    }

    if 'mean_phase' in df.columns:
        df.attrs.setdefault("VariableUnits", {})["mean_phase"] = "n/a"
        df.attrs.setdefault("VariableDescriptions", {})["mean_phase"] = "Mean phase of raw tomogram."

    if 'SD_phase' in df.columns:
        df.attrs.setdefault("VariableUnits", {})["SD_phase"] = "n/a"
        df.attrs.setdefault("VariableDescriptions", {})["SD_phase"] = "Standard deviation of phase of raw tomogram."
    
    if 'RR' in df.columns:
        df.attrs.setdefault("VariableUnits", {})["RR"] = "n/a"
        df.attrs.setdefault("VariableDescriptions", {})["RR"] = "Estimated radiometric resolution."

    if 'cFactor' in df.columns:
        df.attrs.setdefault("VariableUnits", {})["cFactor"] = "n/a"
        df.attrs.setdefault("VariableDescriptions", {})["cFactor"] = "Estimated spatial speckle correlation factor."

def collect_statistics(tomogram: np.ndarray, height: np.ndarray, circ: bool = True) -> pd.DataFrame:
    # Circularize
    if circ:
        tomogram = circularize(tomogram)

    # Convert to intensity
    if np.isrealobj(tomogram):
        clx = False
        tomogram = 10 ** (tomogram / 10)
    else:
        clx = True
        phase = np.angle(tomogram)
        tomogram = np.abs(tomogram) ** 2


    N = tomogram.shape[0]
    mean_backscatter = []
    SD = []
    contrast = []
    E = []
    if clx:
        mean_phase = []
        SD_phase = []

    for n in range(N):
        slice_ = tomogram[n, ...]
        mean_val = np.nanmean(slice_)
        std_val = np.nanstd(slice_)
        max_val = np.nanmax(slice_)
        min_val = np.nanmin(slice_)
        entropy_val = shannon_entropy(slice_.astype(np.float64)) / 8

        mean_backscatter.append(10 * np.log10(mean_val))
        SD.append(10 * np.log10(std_val))
        contrast.append(10 * np.log10(max_val) - 10 * np.log10(min_val))
        E.append(entropy_val)

        if clx:
            slice_ = phase[n, :,:]
            mean_val = np.nanmean(slice_)
            std_val = np.nanstd(slice_)
            mean_phase.append(mean_val)
            SD_phase.append(std_val)

    df = pd.DataFrame({
        "height": height,
        "mean_backscatter": mean_backscatter,
        "SD": SD,
        "contrast": contrast,
        "E": E
    })

    if clx:
        df["mean_phase"] = mean_phase
        df["SD_phase"] = SD_phase

    apply_variable_descriptions(df)

    return df

# RR estimation
def estimaterr(tomogram, NNL=1, ds=1, tolerance=1E-2, npar=os.cpu_count()):
    if isinstance(ds, (list, tuple, np.ndarray)) and any(np.array(ds) > 1):
        tomogram = tomogram[::ds[0], ::ds[1], :]
    elif isinstance(ds, int) and ds > 1:
        tomogram = tomogram[::ds, ::ds, :]

    N = tomogram.shape[2]
    RR = np.zeros(N)
    cFactor = tolerance + np.ones(N)

    sz = tomogram.shape[:2]
    if sz[0] != sz[1]:
        min_sz = min(sz)
        tomogram = tomogram[:min_sz, :min_sz, :]

    for n in tqdm(range(N), desc="Estimating RR: ", leave=False):
        while cFactor[n] > tolerance:
            RR[n], cFactor[n] = _estimaterr_slice(tomogram[:, :, n], npar, NNL, tolerance=tolerance)

    return RR, cFactor

def _estimaterr_slice(I, npar, X0=None, ds=1, tolerance=1E-2):

    # Noise model function
    def noise_fun(x, xdata):
        return x[1] * np.sqrt(x[0] + xdata) + x[2]

    # Subsampling function
    def subsample(I, ds):
        return I[::ds, ::ds]
    
    if X0 is None:
        X0 = [polygamma(1, 1), 10, 0.1]

    if isinstance(X0, (int, float)):
        L0 = X0
        X0 = [polygamma(1, L0), 10, 0.1]
    else:
        L0 = 1 / (X0[0] + 5/3 - np.pi**2/6) + 0.5

    L1 = np.linspace(L0 + 2, L0, 500)
    L2 = np.linspace(L0, L0 / 2, 500)
    L = np.concatenate((L1, L2))
    N = len(L1)
    VAR1 = polygamma(1, L1)
    VAR2 = polygamma(1, L2)
    VAR = np.concatenate((VAR1, VAR2))

    J0 = subsample(I, ds) if ds > 1 else I
    l = min(512, min(J0.shape))
    J0 = J0[:l, :l]
    M = int(3 * l / 4)

    def process_noise(i):
        L_i = L[i]
        g_i = gamma.rvs(L_i, scale=1/L_i, size=J0.shape)
        J = 10 * np.log10(g_i * J0)
        S = svd(J, compute_uv=False)
        return np.mean(S[-M:])

    with ThreadPoolExecutor(max_workers=npar) as executor:
        P = list(executor.map(process_noise, range(len(L))))

    P = np.array(P)
    P1 = P[:N]
    P2 = P[N:]

    lower_bound = [0, 0, 0]
    upper_bound = [100, 100, 100]

    X1 = least_squares(lambda x: noise_fun(x, VAR1) - P1, X0, bounds=(lower_bound, upper_bound)).x
    X2 = least_squares(lambda x: noise_fun(x, VAR2) - P2, X0, bounds=(lower_bound, upper_bound)).x
    cFactor = abs(np.arctan(X1[1]) - np.arctan(X2[1]))

    if cFactor >= tolerance:
        if min(I.shape) / (ds + 1) < 512:
            return X0[0], cFactor
        X = least_squares(lambda x: noise_fun(x, VAR) - P, X2, bounds=(lower_bound, upper_bound)).x
        return _estimaterr_slice(I, npar, X, ds + 1, tolerance)
    else:
        X = least_squares(lambda x: noise_fun(x, VAR) - P, X2, bounds=(lower_bound, upper_bound)).x
        return X[0], cFactor

# Helper function to format duration from seconds to 'dd:hh:mm:ss' or 'hh:mm:ss'
def format_duration(seconds: int|float|timedelta, print_days: bool = False) -> str :
    if isinstance(seconds, (int, float)):
        duration = timedelta(seconds=seconds)
    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if print_days:
        return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Helper function to convert 'dd:hh:mm:ss' or 'hh:mm:ss' to seconds
def duration_seconds(duration: str) -> int:
    match = re.search(r'(\d{2}):(\d{2}):(\d{2})(?::(\d{2}))?')
    num_matched = sum(1 for g in match.groups() if g is not None)
    t = 0
    if num_matched == 3:
        t += int(match.group(1)) * 3600
        t += int(match.group(2)) * 60
        t += int(match.group(3))
    if num_matched == 4:
        t += int(match.group(1)) * 3600 * 24
        t += int(match.group(2)) * 3600
        t += int(match.group(3)) * 60
        t += int(match.group(4))
    return t

# Helper function to parse a string to datetime, date or time object
def parse_datetime_string(s: str) -> datetime|date|time:
    s = s.strip()
    
    datetime_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]
    
    date_formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
    ]
    
    time_formats = [
        "%H:%M:%S",
        "%H:%M",
    ]
    
    for fmt in datetime_formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    
    for fmt in date_formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    
    for fmt in time_formats:
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse '{s}' as datetime, date, or time.")

# Bin variables from a dict or pd.DataFrame according to the corresponding angular value
def bin_by_angle(theta, vars, bin_count=None, units='degrees', rotate: bool = False) -> tuple[dict[np.ndarray],str]:
    """
    Bins unwrapped angles into wrapped bins and computes the median of associated variables.

    Parameters:
        theta (array-like or str): Unwrapped angles (in degrees by default), or name of field in `vars`.
        vars (dict or pd.DataFrame): Dictionary or DataFrame of variables to bin.
        bin_count (int, optional): Number of bins over [0, 360). If None, estimated from gradient.
        units (str): 'degrees' or 'radians', the unit of theta

    Returns:
        tuple with (dict of np.array (2D) with binned medians for each variable and wrapping, name of angle key)
    """
    if isinstance(theta, str):
        theta_name = theta
        theta = vars[theta_name]
        angle_is_field = True
    else:
        angle_is_field = False

    theta = np.asarray(theta)

    if units == 'radians':
        theta = np.degrees(theta)

    if bin_count is None:
        bin_count = int(np.floor(360 / np.max(np.gradient(theta))))

    theta_wrapped = np.mod(theta, 360)

    wrap_index = np.round((theta - theta_wrapped) / 360).astype(int)
    unique_wraps, wrap_map = np.unique(wrap_index, return_inverse=True)
    wrap_count = len(unique_wraps)

    bin_edges = np.linspace(0, 360, bin_count + 1)
    bin_idx = np.digitize(theta_wrapped, bin_edges) - 1
    bin_idx[bin_idx == bin_count] = bin_count - 1

    var_names = vars.columns if isinstance(vars, pd.DataFrame) else vars.keys()
    binned = {name: [[] for _ in range(wrap_count)] for name in var_names}
    for i in range(len(theta)):
        b = bin_idx[i]
        if b < 0 or b >= bin_count:
            continue
        w = wrap_map[i]
        for name in var_names:
            value = vars[name].iloc[i] if isinstance(vars, pd.DataFrame) else vars[name][i]
            binned[name][w].append((b, value))

    result = {}
    for name in var_names:
        mat = np.full((bin_count, wrap_count), np.nan)
        for w in range(wrap_count):
            bin_values = [[] for _ in range(bin_count)]
            for b, val in binned[name][w]:
                bin_values[b].append(val)
            for b in range(bin_count):
                if bin_values[b]:
                    mat[b, w] = np.median(bin_values[b])
        result[name] = mat

    if angle_is_field:
        result[theta_name] = np.nanmedian(np.mod(result[theta_name], 360), axis=1)
        if units == 'radians':
            result[theta_name] = np.radians(result[theta_name])
    else:
        result['theta'] = (bin_edges[:-1] + bin_edges[1:]) / 2
        theta_name = 'theta'
    if rotate:
        _rotate_bins(result, theta_name)

    return result, theta_name

def _rotate_bins(binned_matrices, theta_name):
    """
    This functions rotates the output of abin so that the first row contains the start of the first
    wrapping, and counts angles from this position instead.
    """

    # Get the angle vector
    theta = binned_matrices[theta_name]  # shape: (bin_count,)

    # Get all matrices
    matrices = [binned_matrices[k] for k in binned_matrices if k != theta_name]

    # Find the first bin index where all matrices have non-NaN in the first wrapping column
    valid_mask = np.all([~np.isnan(mat[:, 0]) for mat in matrices], axis=0)
    first_valid_bin = np.argmax(valid_mask)
    start_theta = theta[first_valid_bin]

    for key in binned_matrices:
        binned_matrices[key] = np.roll(binned_matrices[key], -first_valid_bin, axis=0)
    
    binned_matrices[theta_name] = np.mod(binned_matrices[theta_name] - start_theta, 360)

    return binned_matrices

# Compute and handle stats from a dictionary with 1D np.ndarrays or pd.DataFrames with nesting
def compute_stats(d: dict):
    result_mean = {}
    result_std = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively compute stats for nested dict
            mean_sub, std_sub = compute_stats(value)
            result_mean[key] = mean_sub
            result_std[key] = std_sub
        elif isinstance(value, np.ndarray):
            # Compute mean and stdiance for 1D array
            if len(value.shape) > 1:
                raise TypeError(f"Key {key} shape: {value.shape}")
            result_mean[key] = value.mean()
            result_std[key] = value.std()
        elif isinstance(value, pd.Series):
            result_mean[key] = value.mean()
            result_std[key] = value.std()
        elif isinstance(value, pd.DataFrame):
            df = value.iloc[:,1:]
            result_mean[key] = df.mean().to_dict()
            result_std[key] = df.std().to_dict()
        else:
            raise TypeError(f"Unsupported type for key '{key}': {type(value)}")
    return result_mean, result_std

def round_to_sig_digits(value, digits=3):
    if not isinstance(value, (float, np.floating)):
        raise TypeError(f"Value: {value} of type {type(value)}")
    if value == 0:
        return 0
    return round(value, -int(math.floor(math.log10(abs(value))) - (digits - 1)))

def round_to_same_decimal(value, reference):
    if reference == 0:
        return value
    if abs(reference) > abs(value):
        return round_to_sig_digits(value, 1)
    decimal_pos = -int(math.floor(math.log10(abs(reference))))
    return round(value, decimal_pos)

def combine_stats(means: dict, stds: dict, sig_digits=3) -> dict:
    combined = {}
    for key in means:
        mean_val = means[key]
        std_val = stds.get(key)

        if isinstance(mean_val, dict) and isinstance(std_val, dict):
            combined[key] = {}
            for subkey in mean_val:
                raw_std = std_val.get(subkey, 0)
                if not isinstance(raw_std, (float, np.floating)):
                    raise TypeError(f"{key}: {subkey}: {raw_std} of type {type(raw_std)}")
                std_dev = round_to_sig_digits(raw_std, sig_digits)
                rounded_mean = round_to_same_decimal(mean_val[subkey], std_dev)
                
                combined[key][subkey] = {
                    'mean': rounded_mean,
                    'std_dev': std_dev
                }
        else:
            raw_std = std_val or 0
            std_dev = round_to_sig_digits(raw_std, sig_digits)
            rounded_mean = round_to_same_decimal(mean_val, std_dev)

            combined[key] = {
                'mean': rounded_mean,
                'std_dev': std_dev
            }
    return combined

# Compute normalized RMSE between signal and ideal (symmetric):
# Relative root mean size of energy difference and mean energy (stable replacement for relative RMSE with values bounded in [0,sqrt(2)])
def normalized_rmse(signal: list[float]|np.ndarray, ideal: list[float]|np.ndarray) -> np.floating:
    if isinstance(signal, np.ndarray) and len(signal.shape) > 1:
        raise ValueError("Signal input is not a 1D array.")
    if isinstance(ideal, np.ndarray) and len(ideal.shape) > 1:
        raise ValueError("Ideal input is not a 1D array.")
    if len(signal) != len(ideal):
        raise ValueError("Signal and ideal do not match.")
    denominator = signal**2 + ideal**2
    numerator = 2 * (signal - ideal)**2
    relative_square_errors = np.zeros_like(denominator)
    nonzero_mask = denominator != 0
    relative_square_errors[nonzero_mask] = numerator[nonzero_mask] / denominator[nonzero_mask]

    return math.sqrt(relative_square_errors.mean())

# Add meta data at the beginning of a dict
def add_meta(data: dict, info_str: str, key: str = '__meta__') -> dict:
    if key in data and isinstance(data[key], str):
        new_data = data.copy()
        new_data[key] = data[key] + info_str
    elif key in data: 
        raise ValueError(f"Key {key} already exists and does not contain a string.")
    else:
        new_data = {key: info_str}
        new_data.update(data)
    return new_data

# Update nested dicts, merging pd.DataFrames by stacking columns when possible:
def update_nested_dict(original: dict, updates: dict) -> dict:
    for key, subdict in updates.items():
        if key in original and isinstance(original[key], dict) and isinstance(subdict, dict):
            update_nested_dict(original[key], subdict)
        elif key in original and isinstance(original[key], pd.DataFrame) and isinstance(subdict, pd.DataFrame):
            df1 = original[key]
            df2 = subdict
            if len(df1) != len(df2):
                original[key] = df2 # Overwrite mismatching DataFrames
            df2 = df2.set_index(df1.index) # Align indices if needed
            overlapping = df1.columns.intersection(df2.columns)
            df1_clean = df1.drop(columns=overlapping) # Overwrite columns in original
            original[key] = pd.concat([df1_clean, df2], axis=1) # Concatenate 
        else:
            original[key] = subdict  # Add new key or overwrite non-dict
    return original

def invert_nested_dict(nested: dict) -> dict:
    # Get all outer keys
    outer_keys = set(nested.keys())

    # Invert the dictionary
    inverted = defaultdict(dict)
    for outer_key, inner_dict in nested.items():
        for inner_key, value in inner_dict.items():
            inverted[inner_key][outer_key] = value

    # Sort keys: complete ones first, incomplete ones last
    sorted_keys = sorted(
        inverted.keys(),
        key=lambda k: len(inverted[k]) < len(outer_keys)  # False < True → complete first
    )

    # Reconstruct sorted dict
    return {k: inverted[k] for k in sorted_keys}

# Function to return a string representation of the model resulting from a LinearRegression().fit()
def linear_model_str(model: LinearRegression, var: str = 't', rounded: bool = True) -> str:
    if model.coef_[0] == 0 and model.intercept_ == 0:
        return "0"
    if rounded:
        return f"{f'{model.intercept_:.3g}' if model.intercept_ != 0 else ''}{
            f' + {model.coef_[0]:.3g} * {var}' if model.coef_[0] > 0 else f' - {abs(model.coef_[0]):.3g} * {var}' if model.coef_[0] < 0 else ''
        }"
    else:
        return f"{f'{model.intercept_}' if model.intercept_ != 0 else ''}{
            f' + {model.coef_[0]} * {var}' if model.coef_[0] > 0 else f' - {abs(model.coef_[0])} * {var}' if model.coef_[0] < 0 else ''
        }"

def prompt_ftp_login(server: str, max_attempts: int = 3, user: str = None, pw: str = None, anonymous: bool = False):
    """
    Prompts for FTP login credentials and retries if login fails.
    Returns a connected FTP object.
    """
    for attempt in range(1, max_attempts + 1):
        if anonymous:
            ftp_user = "anonymous"
            ftp_pass = "none"
        else:
            if user:
                ftp_user = user
            else:
                ftp_user = input(f"Enter username for {server}: ")
            if pw:
                ftp_pass = pw
            else:
                ftp_pass = getpass(f"Enter password for {server}: ")
            
        try:
            ftp = FTP(server)
            ftp.login(user=ftp_user, passwd=ftp_pass)
            print(f"Login successful ({server}).")
            return ftp, ftp_user, ftp_pass
        except error_perm as e:
            print(f"Login failed ({attempt}/{max_attempts}): {e}")
            if attempt == max_attempts:
                raise ConnectionError("Maximum login attempts exceeded.")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

def gunzip(input_path: Path|str, output_path: Path|str = None) -> Path:
    input_path = Path(input_path)
    if input_path.suffix != '.gz':
        raise ValueError(f"{input_path} is not a .gz file")

    if not output_path:
        output_path = input_path.with_suffix('')  # Strip .gz
    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())

    input_path.unlink()  # Delete the original .gz file
    return output_path

def extract_datetime(filename) -> datetime | None:
    """Matches pattern like 2025-09-02-18-10-42 and returns a matching timezone aware datetime object (UTC)"""
    match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', filename)
    if match:
       dt = datetime.strptime(match.group(), "%Y-%m-%d-%H-%M-%S")
       return dt.replace(tzinfo = timezone.utc)
    return None

# Function to read mocoref data from a data file
def generate_mocoref(data: str|Path|dict|pd.DataFrame, type: str = None, output_dir: Path|str|None = None, line: int = 1, pco_offset: float = -0.079, tstart: datetime|None = None, tend: datetime|None = None, tolerance: float = 0.2, generate: bool = True, verbose: bool = False) -> tuple[tuple[float, float, float], Path]:
    """Reads mocoref data from a data file. Valid types: CSV, JSON, LLH and mocoref. If not specified attempts to determine file type from file extension.
    The line parameter specifies which line in a CSV file the mocoref data is read from. Optionally generates a mocoref.moco file.
    
    If data is dict or DataFrame instead of Path or string, mocoref data will be read from the dict or DataFrame instead.
    A DataFrame will be interpreted as having the mocoref data in a single line if the line parameter is positive, and as being an LLH log if it is zero.
    
    LLH data read from a file is assumed to lack a header and have identical columns to Reach RS3 output:
    date, time, latitude, longitude, height, Q, satellites, sdn, sde, sdu, sdne, sdeu, sdun, age, AR_ratio.

    If tstart and/or tend are specified the LLH data will be limited to matching timestamps, otherwise the entire log is used. Note however that from within the timestamps used,
    there is extracted the position which gives the LONGEST TOTAL FIX with a tolerance of position variation specified by the tolerance parameter.
    
    The pco_offset parameter allows the user to specify a vertical PCO offset between the receiver used to record mocoref data and the receiver used in drone processing.
    Note: this applies to CSV files ONLY.
    
    Returns:
    - latitude
    - longitude
    - ellipsoidal height
    - antenna height"""

    # Check if data or data file was passed
    data_file = None
    if isinstance(data, dict):
        type = "JSON"
    elif isinstance(data, pd.DataFrame):
        if line == 0:
            type = "LLH"
        else:
            type = "CSV"
    else:
        data_file = Path(data)
        if not data_file.is_file():
            raise FileNotFoundError(f"File {data_file} cannot be found.")
        # Automatically interpret type unless specified
        if type is None:
            if data_file.suffix in (".CSV", ".csv"):
                type = "CSV"
            elif data_file.suffix in (".JSON", ".json"):
                type = "JSON"
            elif data_file.suffix in (".LLH", ".llh"):
                type = "LLH"
            elif data_file.name == "mocoref.moco":
                type = "mocoref"
            else:
                raise RuntimeError("Failed to interpret file type.")
    
    if output_dir:
        output_dir = Path(output_dir)
    elif data_file:
        output_dir = data_file.parent
    else:
        output_dir = Path.cwd()
    
    type = type.casefold()
    # Validate type
    valid_types = ["csv", "json", "llh", "mocoref"]
    if type not in valid_types:
        raise TypeError(f"Invalid type {type}. Valid types: {valid_types}")
    
    # Get mocoref data 
    settings = Settings()
    match type:
        case "csv":
            # Read data from file
            if data_file:
                data = pd.read_csv(data_file)

            # Validate the presence of mocoref data
            if settings.MOCOREF_LATITUDE not in data.columns:
                raise KeyError(f"{settings.MOCOREF_LATITUDE} key not found in {data_file}")
            if settings.MOCOREF_LONGITUDE not in data.columns:
                raise KeyError(f"{settings.MOCOREF_LONGITUDE} key not found in {data_file}")
            if settings.MOCOREF_HEIGHT not in data.columns:
                raise KeyError(f"{settings.MOCOREF_HEIGHT} key not found in {data_file}")
            if settings.MOCOREF_ANTENNA not in data.columns:
                raise KeyError(f"{settings.MOCOREF_ANTENNA} key not found in {data_file}")
            
            # Validate line parameter
            if not isinstance(line, int) or not line > 0:
                raise TypeError(f"The line parameter must specify a valid positive integer line for this data. Current: {line}")
            if line > len(data):
                raise IndexError(f"The line {line} does not exist in data.")
            
            mocoref_latitude = data[settings.MOCOREF_LATITUDE].iloc[line - 1]
            mocoref_longitude = data[settings.MOCOREF_LONGITUDE].iloc[line - 1]
            mocoref_height = data[settings.MOCOREF_HEIGHT].iloc[line - 1]
            mocoref_antenna = data[settings.MOCOREF_ANTENNA].iloc[line - 1]

            # Modify antenna offset to account for difference between RS3 and CHCI83 PCO:
            mocoref_antenna = mocoref_antenna + pco_offset

        case "json":
            # Read data from file
            if data_file:
                data = json.load(data_file)

            # Validate the presence of mocoref data
            if settings.MOCOREF_LATITUDE not in data:
                raise KeyError(f"{settings.MOCOREF_LATITUDE} key not found in {data_file}")
            if settings.MOCOREF_LONGITUDE not in data:
                raise KeyError(f"{settings.MOCOREF_LONGITUDE} key not found in {data_file}")
            if settings.MOCOREF_HEIGHT not in data:
                raise KeyError(f"{settings.MOCOREF_HEIGHT} key not found in {data_file}")
            if settings.MOCOREF_ANTENNA not in data:
                raise KeyError(f"{settings.MOCOREF_ANTENNA} key not found in {data_file}")
            
            mocoref_latitude = data[settings.MOCOREF_LATITUDE]
            mocoref_longitude = data[settings.MOCOREF_LONGITUDE]
            mocoref_height = data[settings.MOCOREF_HEIGHT]
            mocoref_antenna = data[settings.MOCOREF_ANTENNA]
        case "llh":
            # Read data from file
            if data_file:
                data = pd.read_csv(data_file, sep=r'\s+',  names=['date','time', settings.MOCOREF_LATITUDE, settings.MOCOREF_LONGITUDE,
                                                     settings.MOCOREF_HEIGHT, 'quality', 'satellites', 'sdn', 'sde', 'sdu',
                                                     'sdne', 'sdeu', 'sdun', 'age', 'ar_ratio'])

            # Validate the presence of mocoref data
            if settings.MOCOREF_LATITUDE not in data:
                raise KeyError(f"{settings.MOCOREF_LATITUDE} key not found in {data_file}")
            if settings.MOCOREF_LONGITUDE not in data:
                raise KeyError(f"{settings.MOCOREF_LONGITUDE} key not found in {data_file}")
            if settings.MOCOREF_HEIGHT not in data:
                raise KeyError(f"{settings.MOCOREF_HEIGHT} key not found in {data_file}")
            
            # Extract matching timestamps if a RINEX file is provided:
            if tstart or tend:
                data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%Y/%m/%d %H:%M:%S.%f', utc=True)
                if tstart:
                    data = data[data['timestamp'] >= tstart]
                if tend:
                    data = data[data['timestamp'] <= tend]
            # LLH logs record phase center position
            mocoref_antenna = 0
            
            # If quality is specified limit to Q=1 if present and find most dense cluster of continuous segments
            if 'quality' in data and np.where(data['quality'] == 1):
                data = data.iloc[np.where(data['quality'] == 1)]
                
                # Weight and mean for each continuous segment in ECEF
                idiff = np.diff(data.index)
                split_points = np.where(idiff > 1)[0]
                start = 0
                weights = []
                points = []
                for point in split_points:
                    end = point + 1
                    segment = data.iloc[start:end]
                    weights.append(len(segment))
                    points.append(geo_to_ecef.transform(
                        segment[settings.MOCOREF_LONGITUDE].mean(),
                        segment[settings.MOCOREF_LATITUDE].mean(),
                        segment[settings.MOCOREF_HEIGHT].mean()
                    ))
                    start = end
                segment = data.iloc[start:]
                weights.append(len(segment))
                points.append(geo_to_ecef.transform(
                    segment[settings.MOCOREF_LONGITUDE].mean(),
                    segment[settings.MOCOREF_LATITUDE].mean(),
                    segment[settings.MOCOREF_HEIGHT].mean()
                ))

                # Convert to numpy arrays
                weights = np.array(weights)
                points = np.array(points)

                # Use DBSCAN clustering to find dense regions within the threshold
                db = DBSCAN(eps=tolerance, min_samples=1, metric='euclidean')
                labels = db.fit_predict(points)

                # Find the cluster with maximum total weight
                best_cluster_label = None
                max_weight = 0
                for label in set(labels):
                    cluster_mask = labels == label
                    cluster_weight = weights[cluster_mask].sum()
                    if cluster_weight > max_weight:
                        max_weight = cluster_weight
                        best_cluster_label = label

                # Extract best cluster
                best_mask = labels == best_cluster_label
                points = points[best_mask]
                weights = weights[best_mask]

                # Perform weighted mean
                pos = np.sum(weights[:, np.newaxis] * points, axis=0) / np.sum(weights)
                mocoref_longitude, mocoref_latitude, mocoref_height = ecef_to_geo.transform(*pos)
            else:
                mocoref_latitude = data[settings.MOCOREF_LATITUDE].mean()
                mocoref_longitude = data[settings.MOCOREF_LONGITUDE].mean()
                mocoref_height = data[settings.MOCOREF_HEIGHT].mean()
        case "mocoref":
            generate = False
            if not data_file:
                raise RuntimeError("No mocoref file was specified")
            with open(data_file, 'r') as file:
                lines = file.readlines()
            value = re.compile(r"\d+(?:[\.\,]\d*)?")
            mocoref_antenna = float(value.search(lines[3]).group())
            mocoref_latitude = float(value.search(lines[4]).group())
            mocoref_longitude = float(value.search(lines[5]).group())
            mocoref_height = float(value.search(lines[6]).group())

    if generate or verbose:
        lines = []
        lines.append("Moco reference {CH}\n")
        lines.append("===================\n")
        lines.append("\n")
        lines.append(f"Antenna height [m] {{d}}: {mocoref_antenna}\n")
        lines.append(f"Latitude [deg]     {{d}}: {mocoref_latitude}\n")
        lines.append(f"Longitude [deg]    {{d}}: {mocoref_longitude}\n")
        lines.append(f"Ground [m]         {{d}}: {mocoref_height}\n")
    if generate:
        if data_file:
            mocoref_path = data_file.parent / "mocoref.moco"
        else:
            mocoref_path = Path.cwd() / "mocoref.moco"
        with open(mocoref_path, 'w') as file:
            file.writelines(lines)
    else:
        mocoref_path = None
    if verbose or settings.VERBOSE:
        print(''.join(lines))

    return (mocoref_latitude, mocoref_longitude, mocoref_height + mocoref_antenna), mocoref_path
