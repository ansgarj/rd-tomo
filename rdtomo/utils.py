# Imports
from __future__ import annotations
import os
import subprocess
import shutil
import platform
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.measure import shannon_entropy
from scipy.special import polygamma
from scipy.stats import gamma
from scipy.linalg import svd
from scipy.optimize import least_squares
from scipy.ndimage import binary_closing
from sklearn.linear_model import RANSACRegressor, LinearRegression
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
from typing import TypeAlias, Sequence

from .tomogram_processing import circularize

IndexType = int | slice | Sequence[int] | Sequence[bool] | np.ndarray

class Angles:
    __slots__ = ("_degs", "_rads")
    _degs: npt.NDArray[np.float64]|None     
    _rads: npt.NDArray[np.float64]|None

    def __init__(self, angles: npt.ArrayLike, degrees: bool = False) -> Angles:
        """Initiates Angles object. By default the angles are interpreted in radians,
        but setting degrees to True changes to degrees."""
        angles = np.asarray(angles, dtype=float)
        if angles.ndim != 1:
            raise ValueError("Angles stores only only one-dimensional array-like objects.")
        if degrees:
            self._degs = angles
            self._rads = None
        else:
            self._rads = angles
            self._degs = None

    @property
    def rads(self) -> npt.NDArray[np.float64]:
        if self._rads is None:
            if self._degs is None:
                raise ValueError("Object initiated with neither degrees nor radians.")
            self._rads = np.radians(self._degs)
        return self._rads.copy()
    
    @property
    def degs(self) -> npt.NDArray[np.float64]:
        if self._degs is None:
            if self._rads is None:
                raise ValueError("Object initiated with neither degrees nor radians.")
            self._degs = np.degrees(self._rads)
        return self._degs.copy()

    def unwrap(self, degrees: bool = False) -> npt.NDArray[np.float64]:
        unwrapped = np.unwrap(self.rads)        
        if degrees:
            return np.degrees(unwrapped)
        else:
            return unwrapped
        
    def wrap(self, degrees: bool = False) -> npt.NDArray[np.float64]:
        if degrees:
            return self.degs % 360
        else:
            return self.rads % (2*np.pi)
    
    def __bool__(self) -> bool:
        return bool(self._degs) or bool(self._rads)
    
    def __len__(self) -> int:
        if self._rads is not None:
            return len(self._rads)
        elif self._degs is not None:
            return len(self._degs)
        raise ValueError("Object initiated with neither degrees nor radians.")
    
    def __getitem__(self, idx: IndexType) -> Angles:
        """Returns Angles object with a set of coordinates determined by idx."""
        if self._rads is not None:
            obj =  Angles(self.rads[idx])
            if self._degs is not None:
                obj._degs = self.degs[idx]
        else:
            obj = Angles(self.degs[idx], degrees=True)
        return obj

    def __setitem__(self, idx: IndexType, value: Angles|npt.NDArray[np.floating]):
        obj = Angles(value)
        if len(obj) == len(self[idx]):
            self._coords = obj.coords
        else:
            raise ValueError(f"The value must match the idx, and be serializable as a Angles object, not {value}")
        
    def copy(self) -> Angles:
        if self._rads is not None:
            cp = Angles(self.rads)
            if self._degs is not None:
                cp._degs = self.degs
        else:
            cp = Angles(self.degs, degrees=True)
        return cp
    
    def join(self, other: Angles|npt.NDArray[np.floating]) -> None:
        """Serializes the other object as a Angles object and joins it to the current."""
        if self._rads is not None:
            self._rads = np.hstack((self._rads, other.rads))
        if self._degs is not None:
            self._degs = np.hstack((self._degs, other.degs))

    def __str__(self) -> str:
        return f"degrees({self.degs})"
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

# Infer Reference frame from string lines
def infer_rf(lines: list[str], new_rf: None|str = None) -> str:
    """Infers reference frame from a list of lines. These lines are assumed to
    be from an ASCII header, and the reference frame line is supposed to have the
    shape:
    
    % REF FRAME :   <RF>\\n
    
    If new_rf is not None, then infer_rf will update lines to match the reference
    frame.
    
    Returns:
        - Inferred reference frame (string)
        - lines object with new reference frame"""

    for i, line in enumerate(lines):
        if line.startswith("% REF FRAME"): 
            inferred_rf = line.split()[-1]          
            if new_rf is not None:
                lines[i] = f"% REF FRAME :   {new_rf}\n"
            return inferred_rf, lines

    if new_rf is not None:
        lines = [f"% REF FRAME :   {new_rf}\n"] + lines

    return None, lines

# ASCII tabular data reader to avoid pandas errors when misaligned
def ascii_reader(input: str|Path, content: bool = False) -> tuple[np.ndarray, list[str], list[str]]:
    """Reads tabular ASCII data and returns as a single array. The
    header (lines starting with %) is stored separately and returned as
    a list of lines. Also infers the format of numeric data and return as a
    list of strings. Set content=True if passing the content of the ASCII
    file as a string directly.

    Returns:
        - Tabular data (numeric)
        - Header lines
        - Format of numeric data
    """
    def infer_formats(data_lines: list[str]) -> list[str]:
        rows = [line.split() for line in data_lines]
        n_cols = len(rows[0])
        formats = []
        for col in range(n_cols):
            col_toks = [row[col] for row in rows]
            decimals = max((len(t.split(".")[1]) if "." in t else 0) for t in col_toks)
            width = max(len(t) for t in col_toks) + 1  # +1 margin for sign changes etc.
            if any(("e" in t or "E" in t) for t in col_toks):
                formats.append(f"%{width}.{decimals}e")
            elif decimals:
                formats.append(f"%{width}.{decimals}f")
            else:
                formats.append(f"%{width}d")
        return formats

    if content:
        lines = input
    with open(input, 'r') as f:
        lines = f.readlines()

    # Header
    header_lines = []
    rf = None
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('%'):
            header_lines.append(line)
        else:
            break

    # Data
    data_lines = lines[i:]
    fmt = infer_formats(data_lines)
    try:
        data = [list(map(float, line.split())) for line in data_lines]
        data = np.array(data)
    except ValueError:
        raise ValueError(f"Could not parse numeric data from {input}")

    return data, header_lines, fmt

def ascii_writer(path: str|Path,
                 data: np.ndarray,
                 header_lines: list[str] = [],
                 format: list[str] = [],
                 label_spans: dict[str, int] | None = None):
    """Writes numeric data as an ASCII tabular file. Header lines can be passed separately,
    as well as the format for the numeric data as a list of string (matching the columns).
    The last line of the header (if any) is assumed to contain labels, and the label_spans
    parameter can be used to cause any label to span multiple columns (labels not listed
    default to span 1)."""

    def align_header_to_fmt(header_line: str, fmt: list[str], label_spans: dict[str, int] | None = None) -> str:
        """Re-pad a header line's labels to match the field widths used by np.savetxt's fmt list.

        label_spans: optional {label: n_columns} for labels that cover more than one
        data column (e.g. "GPST" spanning gps_week + tow). Labels not listed default to span 1.
        """
        label_spans = label_spans or {}
        labels = header_line.lstrip('%').split()
        widths = [int(re.match(r'%(\d+)', f).group(1)) for f in fmt]

        spans = [label_spans.get(lbl, 1) for lbl in labels]
        if sum(spans) != len(widths):
            # Spans don't account for every data column - can't safely realign
            return header_line

        aligned_parts = []
        idx = 0
        for lbl, span in zip(labels, spans):
            # Sum the widths of the columns this label covers, plus one space
            # per internal join (since those columns are also space-joined by savetxt)
            w = sum(widths[idx:idx + span]) + (span - 1)
            # Remove one space from first entry (to account for %)
            if not aligned_parts:
                w -= 1
            aligned_parts.append(f"{lbl:>{w}}")
            idx += span

        aligned = " ".join(aligned_parts)
        return f"%{aligned}\n"

    with open(path, "w") as f:
        for line in header_lines[:-1]:
            f.write(line)
        f.write(align_header_to_fmt(header_lines[-1], format, label_spans))

        np.savetxt(f, data, fmt=format)

# SRF file reader
def srf_reader(path: str|Path) -> tuple[np.ndarray, None|str]:
    """Reads the Sun Raster (binary) File (SRF), parsing the header, and returns
    the numeric data stored as an array. Reads the 7th header element (cmap) as
    determining a reference frame:
        - 0: None
        - 1: 'ITRF2020'
        - 2: 'ETRF2020'
        - 3: 'SWEREF99'
        - 4: 'EUREF89'
        - 5: 'EUREF-FIN'
        - 6: 'EUREF-DK94'
        - 7: 'LKS94'
        - 8: 'LKS92'
        - 9: 'EUREF-EST97'"""
    
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=8)
    
    # Header entries
    magic_number = header[0]
    width = header[1]
    height = header[2]
    pixel_bits = header[3]
    byte_length = header[4]
    data_type = header[5]
    depth = header[6]               # Traditionally cmap
    ref_frame = header[7]           # Traditionally cmap_length

    rf_map = {
        0: None,
        1: 'ITRF2020',
        2: 'ETRF2020',
        3: 'SWEREF99',
        4: 'EUREF89',
        5: 'EUREF-FIN',
        6: 'EUREF-DK94',
        7: 'LKS94',
        8: 'LKS92',
        9: 'EUREF-EST97'
    }

    # Check magic number
    if magic_number != 1504078485: # 0x59a66a95
        raise RuntimeError(f"You attempted to read file {path} as an SRF file, but it returned the wrong magic number: {hex(magic_number)} (expected 0x59a66a95)")
    
    # Extract data type
    match data_type:
        case 10:
            bits_per_pix = 16 # IDL data type: int
            dtype = np.dtype('int16')
        case 11:
            bits_per_pix = 32 # IDL data type: long
            dtype = np.dtype('int32')
        case 12:
            bits_per_pix = 64 # IDL data type: long64
            dtype = np.dtype('int64')
        case 20:
            bits_per_pix = 32 # IDL data type: float
            dtype = np.dtype('float32')
        case 21:
            bits_per_pix = 64 # IDL data type: double
            dtype = np.dtype('float64')
        case 30:
            bits_per_pix = 64 # IDL data type: complex
            dtype = np.dtype('complex64')
        case 31:
            bits_per_pix = 128 # IDL data type: dcomplex
            dtype = np.dtype('complex128')
        case _:
            raise RuntimeError(f"SRF file {path} returned invalid raster type ID: {data_type}")
        
    # Redundancy
    if (width * height * bits_per_pix / 8) != byte_length:
        raise RuntimeError(f"SRF file {path} length did not match format.")

    # Channel count
    if (pixel_bits != bits_per_pix):
        raise RuntimeError(f"Data type {data_type} ({dtype}) did not match pixel bits {pixel_bits}")
    
    # Read array
    return np.memmap(path, dtype=dtype, mode="r", offset=32, shape=(height, width, depth)).squeeze(), rf_map[ref_frame]

# SRF file writer
def srf_writer(path: str | Path, arr: np.ndarray, ref_frame: str) -> None:
    """Stores numeric array data as a Sun Raster (binary) File (SRF). Allows
    reference frame information to be stored in the header in place of cmap,
    for reading with srf_reader() function. If the string fails to associate with a
    numeric value according to the map below, 0 is used as a fallback:
        - 1: 'ITRF2020'
        - 2: 'ETRF2020'
        - 3: 'SWEREF99'
        - 4: 'EUREF89'
        - 5: 'EUREF-FIN'
        - 6: 'EUREF-DK94'
        - 7: 'LKS94'
        - 8: 'LKS92'
        - 9: 'EUREF-EST97'"""
    
    arr = np.asarray(arr)

    # Ensure 3D layout: (height, width, depth)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    elif arr.ndim != 3:
        raise ValueError("Array must be 2D or 3D")

    height, width, depth = arr.shape

    # SRF type mapping
    dtype_map = {
        np.dtype("int16"): (10, 16),
        np.dtype("int32"): (11, 32),
        np.dtype("int64"): (12, 64),
        np.dtype("float32"): (20, 32),
        np.dtype("float64"): (21, 64),
        np.dtype("complex64"): (30, 64),
        np.dtype("complex128"): (31, 128),
    }

    try:
        data_type, pixel_bits = dtype_map[arr.dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")

    byte_length = arr.nbytes

    rf_map = {
        'ITRF2020': 1,
        'ETRF2020': 2,
        'SWEREF99': 3,
        'EUREF89': 4,
        'EUREF-FIN': 5,
        'EUREF-DK94': 6,
        'LKS94': 7,
        'LKS92': 8,
        'EUREF-EST97': 9
    }
    rf = rf_map[ref_frame] if ref_frame in rf_map else 0

    header = np.array(
        [
            1504078485,  # magic number (0x59a66a95)
            width,
            height,
            pixel_bits,
            byte_length,
            data_type,
            depth, 
            rf,  # cmap_length
        ],
        dtype=np.int32,
    )

    with open(path, "wb") as f:
        header.tofile(f)
        arr.tofile(f)

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
    if not isinstance(paths, (list, tuple)):
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

# Interpolate short gaps using polynomial splice
def close_gaps(positions: np.ndarray,
                mask: np.ndarray,
                k: int = 1,
                degree: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolates short gaps (length <= k) using polynomial splicing.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with e.g. 3D coordinates.
    mask : np.ndarray
        Boolean mask of shape (N,) indicating positions to be corrected by interpolation.
    k : int
        Maximum gap length to interpolate (default = 1).
    degree : int
        Degree of polynomial for splicing (default = 1 for linear).

    Returns
    -------
    updated_positions : np.ndarray
        Positions with short gaps interpolated using polynomial splice.
    remaining_mask : np.ndarray
        Boolean mask of positions still unfilled.
    """
    updated_positions = positions.copy()
    N = positions.shape[0]
    remaining_mask = mask.copy()

    i = 0
    while i < N:
        if remaining_mask[i]:
            start = i
            while i < N and remaining_mask[i]:
                i += 1
            end = i
            gap_len = end - start

            if gap_len <= k and start > 0 and end < N:
                # Left side
                left_idx = []
                j = 1
                while j < 6:
                    if not remaining_mask[start-j]:
                        left_idx.append(start-j)
                    else:
                        break
                    j += 1
                
                # Right side
                right_idx = []
                j = 0
                while j < 5:
                    if not remaining_mask[end+j]:
                        right_idx.append(end+j)
                    else:
                        break
                    j += 1
                
                # Combine
                x_known = left_idx.copy()
                x_known.extend(right_idx)
                for dim in range(3):
                    y_known = np.array([updated_positions[x, dim] for x in x_known])
                    poly_degree = min(degree, len(x_known) - 1)  # fallback to linear if only two points
                    coeffs = np.polyfit(x_known, y_known, deg=poly_degree)
                    for j in range(gap_len):
                        idx = start + j
                        updated_positions[idx, dim] = np.polyval(coeffs, idx)
                remaining_mask[start:end] = False
        else:
            i += 1

    return updated_positions, remaining_mask

# Slice 1D boolean mask
def slice_mask(a: np.ndarray):
    a = np.asarray(a, dtype=bool)

    # Detect transitions
    d = np.diff(a.astype(np.int8))

    # Start positions (False -> True)
    starts = np.flatnonzero(d == 1) + 1
    if a[0]:
        starts = np.r_[0, starts]

    # End positions (True -> False)
    ends = np.flatnonzero(d == -1) + 1
    if a[-1]:
        ends = np.r_[ends, a.size]

    # Return slices
    return [slice(s, e) for s, e in zip(starts, ends)]

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
def estimaterr(tomogram: npt.NDArray, NNL: int = 1, ds: int = 1, tolerance: float = 1E-2, npar: int = os.cpu_count()) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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

def _estimaterr_slice(I: npt.NDArray, npar: int, X0: float|None, ds=1, tolerance: float = 1E-2):

    # Noise model function
    def noise_fun(x, xdata) -> npt.NDArray:
        return x[1] * np.sqrt(x[0] + xdata) + x[2]

    # Subsampling function
    def subsample(I, ds) -> npt.NDArray:
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

    def process_noise(i) -> np.floating:
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

# Helper function to format duration from seconds to '[d day(s), ]hh:mm:ss[.ffffff]
def format_duration(seconds: int|float|timedelta, print_days: bool = False, whole_seconds: bool = True) -> str:
    if isinstance(seconds, (int, float)):
        duration = timedelta(seconds=seconds)
    if not print_days:
        duration = duration - timedelta(days=duration.days)
    if whole_seconds:
        duration = duration - timedelta(microseconds=duration.microseconds)
    return str(duration)

# Helper function to convert '[d days, ]hh:mm:ss[.ffffff]' to seconds
def parse_duration_string(duration: str) -> float:
    """
    Parse a duration string in the format:
        hh:mm:ss[.ffffff]
        d day, hh:mm:ss[.ffffff]
        d days, hh:mm:ss[.ffffff]

    Returns total seconds as float.

    Examples:
        "00:10:00"            -> 600.0
        "01:02:03.500000"     -> 3723.5
        "1 day, 00:00:00"     -> 86400.0
        "3 days, 12:00:00.25" -> 302400.25
    """

    # Regex:
    # - optional days part: one or more digits + whitespace + 'day' or 'days', followed by comma
    # - then hours:minutes:seconds with optional fractional part
    # - allow optional leading/trailing spaces
    pattern = r"""
        ^\s*
        (?:                             # Optional days group
            (?P<days>\d+)\s+
            (?P<dayword>day|days)\s*,\s+
        )?
        (?P<hours>\d{2})
        :
        (?P<minutes>\d{2})
        :
        (?P<seconds>\d{2}(?:\.\d{1,6})?)
        \s*$
    """

    match = re.match(pattern, duration, flags=re.IGNORECASE | re.VERBOSE)
    if not match:
        raise ValueError(f"Invalid duration format: {duration!r}")

    days_str = match.group("days")
    hours_str = match.group("hours")
    minutes_str = match.group("minutes")
    seconds_str = match.group("seconds")

    # Convert parts
    days = int(days_str) if days_str is not None else 0
    hours = int(hours_str)
    minutes = int(minutes_str)
    seconds = float(seconds_str)  # handles optional .ffffff

    # Optional sanity checks (uncomment if you want strict validation)
    # if hours < 0 or hours > 23:
    #     raise ValueError("Hours must be in 00–23 when days are specified.")
    # if minutes < 0 or minutes > 59:
    #     raise ValueError("Minutes must be in 00–59.")
    # if not (0.0 <= seconds < 60.0):
    #     raise ValueError("Seconds must be in 00–59[.ffffff].")

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds

# Helper function to parse a string to datetime, date or time object
def parse_datetime_string(s: str, require_datetime: bool = False) -> datetime|date|time:
    s = s.strip()
    
    datetime_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d-%H-%M-%S"
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
    
    if not require_datetime:
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
    
    raise ValueError(f"Could not parse '{s}' as datetime{'' if require_datetime else ', date, or time'}.")

# replace N:th occurence of pattern in a string
def string_sub(text: str, pattern: str, replacement: str, n: int = 1) -> str:
    """
    Replace the nth occurrence of a regex pattern in the given text.

    :param text: The input string.
    :param pattern: The regex pattern to search for.
    :param replacement: The string to replace the nth occurrence with.
    :param n: The occurrence index (1-based).
    :return: Modified string with nth occurrence replaced.
    """
    matches = list(re.finditer(pattern, text))
    if n <= 0 or n > len(matches):
        return text  # No change if n is out of range

    start, end = matches[n - 1].span()
    return text[:start] + replacement + text[end:]

# Bin variables from a dict or pd.DataFrame according to the corresponding angular value
def bin_by_angle(theta, vars, bin_count=None, units='degrees', rotate: bool = False) -> tuple[dict[str, np.ndarray], str]:
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

def _rotate_bins(binned_matrices: dict[str, np.ndarray], theta_name: str) -> dict[str, np.ndarray]:
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
def compute_stats(d: dict) -> tuple[dict, dict]:
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

def round_to_sig_digits(value: float|np.floating, digits: int = 3) -> float:
    if not isinstance(value, (float, np.floating)):
        raise TypeError(f"Value: {value} of type {type(value)}")
    if value == 0:
        return 0
    return round(value, -int(math.floor(math.log10(abs(value))) - (digits - 1)))

def round_to_same_decimal(value: float|np.floating, reference: float|np.floating) -> float:
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
        return f"{f'{model.intercept_:.2f}' if model.intercept_ != 0 else ''}{
            f' + {model.coef_[0]:.3f} * {var}' if model.coef_[0] > 0 else f' - {abs(model.coef_[0]):.3f} * {var}' if model.coef_[0] < 0 else ''
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

# GPS time manipulation
datetime_object: TypeAlias = datetime | date | np.datetime64 | npt.NDArray[np.datetime64]

GPS_EPOCH= np.datetime64('1980-01-06')
LEAP_SECONDS = np.array([
    np.datetime64('1981-07-01'), np.datetime64('1982-07-01'), np.datetime64('1983-07-01'),
    np.datetime64('1985-07-01'), np.datetime64('1988-01-01'), np.datetime64('1990-01-01'),
    np.datetime64('1991-01-01'), np.datetime64('1992-07-01'), np.datetime64('1993-07-01'),
    np.datetime64('1994-07-01'), np.datetime64('1996-01-01'), np.datetime64('1997-07-01'),
    np.datetime64('1999-01-01'), np.datetime64('2006-01-01'), np.datetime64('2009-01-01'),
    np.datetime64('2012-07-01'), np.datetime64('2015-07-01'), np.datetime64('2017-01-01')
])

def _to_np_datetime64(obj: object) -> npt.NDArray[np.datetime64]:
    """Convert input to np.datetime64 array."""
    if isinstance(obj, (datetime, date)):
        if hasattr(obj, 'tzinfo'):
            return np.array([np.datetime64(obj.astimezone(timezone.utc).replace(tzinfo=None))])
        return np.array([np.datetime64(obj)])
    elif isinstance(obj, np.datetime64):
        return np.array([obj])
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return np.asarray(obj, dtype='datetime64[ms]')
    else:
       raise TypeError(f"Unsupported type: {type(obj)}")

def date_to_gps_week(input_dates: datetime_object) -> int | npt.NDArray[np.int_]:
    dt_array = _to_np_datetime64(input_dates)
    delta_days = (dt_array - GPS_EPOCH).astype('timedelta64[D]').astype(int)
    result = delta_days // 7
    return result[0] if result.size == 1 else result

def gps_week_to_date(input_weeks: int | npt.NDArray[np.int_]) -> np.datetime64 | npt.NDArray[np.datetime64]:
    weeks = np.asarray(input_weeks, dtype=int)
    result = GPS_EPOCH + weeks.astype('timedelta64[W]')
    return result[0] if result.size == 1 else result

def leap_seconds(acquisition_dates: datetime_object) -> int | npt.NDArray[np.int_]:
    dt_array = _to_np_datetime64(acquisition_dates)
    result = np.array([(LEAP_SECONDS <= d).sum() for d in dt_array]).astype("timedelta64[s]")
    return result[0] if result.size == 1 else result

def gpst_to_dt(gpst: float|npt.NDArray[np.floating], reference_date: datetime|np.datetime64) -> npt.NDArray[np.datetime64]:
    """Converts GPST (s) to the corresponding array of np.datetime64 objects."""

    gps_median = float(np.median(gpst))
    gpst = np.asarray(gpst*1E6).astype('timedelta64[us]')
    reference_date = np.datetime64(reference_date)

    # Detect format
    if gps_median > 1e9:
        # Absolute GPS time
        naive_time = GPS_EPOCH + gpst
        fmt = "absolute"
    elif gps_median < 604800:
        # Seconds-of-week
        gps_week = ((reference_date - GPS_EPOCH).astype('timedelta64[D]') // 7)*7
        gps_week_start = GPS_EPOCH + gps_week
        naive_time = gps_week_start + gpst
        fmt = "seconds-of-week"
    else:
        # Check for vendor offset (e.g., minus 1e9)
        expected_abs = (reference_date - GPS_EPOCH).astype(timedelta).total_seconds()
        diff = abs(expected_abs - gps_median)
        if abs(diff - 1e9) < 5e7:  # tolerance ~50 million seconds (~1.5 years)
            naive_time = GPS_EPOCH + gpst + np.timedelta64(10**9, type='timedelta64[s]')
            fmt = "offset(+1e9)"
        else:
            raise ValueError(f"Unknown GPS time format: median={gps_median}, diff={diff}")
    
    # Apply leap second correction
    return naive_time - np.asarray(leap_seconds(naive_time)).astype('timedelta64[s]')

def gps_week_start(dt: datetime) -> datetime:
    """
    Return the UTC datetime for Sunday 00:00:00 of the GPS week containing dt.
    dt may be naive (assumed UTC) or timezone-aware.
    """
    # Normalize to UTC
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)

    # Python weekday: Monday=0 ... Sunday=6
    days_since_sunday = (dt_utc.weekday() + 1) % 7
    start_date = (dt_utc - timedelta(days=days_since_sunday)).date()
    utc_start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)

    return utc_start - leap_seconds(utc_start)

def decimal_year_to_datetime(year: float) -> datetime:
    y = datetime(year=int(year), month=1, day=1)
    next_y = datetime(year=int(year) + 1, month=1, day=1)
    seconds_in_year = (next_y - y).total_seconds()
    seconds_passed = seconds_in_year * (year - int(year))
    return y + timedelta(seconds=seconds_passed)

def gpst(dt: datetime) -> float:
    """
    Return the GPST (s) of the datetime object since concurrent GPS week.
    """
    start = gps_week_start(dt)
    
    # dt in UTC to match start
    dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    return (dt_utc - start).total_seconds()

def verify_td64(ti: object) -> np.timedelta64:
    """Verifies input as np.timedelta64[us]"""
    if isinstance(ti, timedelta):
        return np.timedelta64(ti, 'us')
    elif isinstance(ti, np.timedelta64):
        return ti.astype('timedelta64[us]')
    if isinstance(ti, str):
        return np.timedelta64(timedelta(seconds=parse_duration_string(ti)), 'us')
    if isinstance(ti, float):
        return np.timedelta64(timedelta(seconds=ti), 'us')
    raise ValueError(f"{ti} could not be interpreted as a timedelta")
            
def verify_dt64(ti) -> np.datetime64:
    """Verifies input as np.datetime64"""
    if isinstance(ti, datetime):
        if ti.tzinfo:
            ti = ti.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(ti, 'us')
    elif isinstance(ti, np.datetime64):
        return ti.astype('datetime64[us]')
    elif isinstance(ti, str):
        return np.datetime64(parse_datetime_string(ti), 'us')
    elif isinstance(ti, float):
        return np.datetime64(decimal_year_to_datetime(ti), 'us')
    raise ValueError(f"{ti} could not be interpreted as a datetime")
