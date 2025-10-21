import click
from pathlib import Path
import json
from getpass import getpass
import re

from ..config import Settings, save_default
from ..utils import warn

def read_three_numbers(prompt) -> list:
    user_input = input(f"{prompt} (enter 3 numbers): ")

    # Regex to match floats including scientific notation
    pattern = r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'

    # Find all matches
    matches = re.findall(pattern, user_input)

    # Convert to float
    float_list = [float(m) for m in matches]

    # Validate count
    if len(float_list) != 3:
        print("Please enter exactly 3 numbers.")
        float_list = read_three_numbers()
    return float_list

def verify_atx(file_path: str|Path):
    """
    Checks if the .atx file has a valid header with:
    - Line 1: exactly 5 leading spaces before version, 12 spaces between version and system flag,
              valid system flag (G, R, E, C, J, S, M), and ends with 'ANTEX VERSION / SYST'
    - Line 2: starts with 'A' as PCV TYPE, 60 spaces before 'PCV TYPE / REFANT',
              REFANT field must be empty, and ends with 'PCV TYPE / REFANT'
    Returns True if all conditions are met, otherwise False.
    """
    valid_system_flags = {'G', 'R', 'E', 'C', 'J', 'S', 'M'}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line1 = f.readline()
            line2 = f.readline()

            # Line 1 checks
            if not line1.endswith("ANTEX VERSION / SYST\n"):
                return False
            if not line1.startswith("     "):  # 5 leading spaces
                return False
            version_str = line1[5:10].strip()
            inter_space = line1[10:22]
            if inter_space != " " * 12:
                return False
            system_flag = line1[22].strip()
            try:
                version = float(version_str)
            except ValueError:
                return False
            if version < 1.4 or system_flag not in valid_system_flags:
                return False

            # Line 2 checks
            if not line2.endswith("PCV TYPE / REFANT\n"):
                return False
            pcv_type = line2[0].strip()
            refant_field = line2[1:60]
            if pcv_type != 'A' or refant_field != " " * 59:
                return False

            return True
    except Exception as e:
        print(f"Error reading file: {e}")
        return False


@click.command()
def default() -> None:
    """Restore default settings"""
    save_default()
    print("TomoSAR settings restored to default.")

@click.command()
def settings() -> None:
    """Display settings"""
    Settings().print()

@click.command()
def verbose() -> None:
    """Toggle verbose mode"""
    st = Settings()
    if st.VERBOSE:
        st.VERBOSE = False
        print("VERBOSE toggled OFF")
    else:
        st.VERBOSE = True
        print("VERBOSE toggled ON")
    st.save()

@click.command()
@click.argument("key")
@click.argument("value")
def set(key, value) -> None:
    """Set value for settings. Valid keys are:
    RTKP_CONFIG, DATA_DIRS, PROCESSING_DIRS, TOMO_DIRS, SWEPOS_USERNAME, SWEPOS_PASSWORD, MOCOREF_LONGITUDE, MOCOREF_LATITUDE, MOCOREF_HEIGHT, MOCOREF_ANTENNA"""
    
    valid_keys = ["RTKP_CONFIG", "DATA_DIRS", "PROCESSING_DIRS", "TOMO_DIRS", "SWEPOS_USERNAME", "SWEPOS_PASSWORD",
                  "MOCOREF_LONGITUDE", "MOCOREF_LATITUDE", "MOCOREF_HEIGHT", "MOCOREF_ANTENNA"]
    settings = Settings()
    if not key in valid_keys:
        raise RuntimeError(f"Invalid key {key}. Valid keys: {valid_keys}")
    
    settings.set(key, value)
    settings.save()

@click.command()
@click.argument("keys", nargs=-1)
def clear(keys) -> None:
    """Clear Settings value. Valid keys are:
    RTKP_CONFIG, DATA_DIRS, PROCESSING_DIRS, TOMO_DIRS, SWEPOS_USERNAME, SWEPOS_PASSWORD, MOCOREF_LONGITUDE, MOCOREF_LATITUDE, MOCOREF_HEIGHT, MOCOREF_ANTENNA, DEMS, CANOPIES, MASKS"""
    
    valid_keys = ["RTKP_CONFIG", "DATA_DIRS", "PROCESSING_DIRS", "TOMO_DIRS", "SWEPOS_USERNAME", "SWEPOS_PASSWORD",
                  "SATELLITES", "RECEIVERS", "MOCOREF_LONGITUDE", "MOCOREF_LATITUDE", "MOCOREF_HEIGHT", "MOCOREF_ANTENNA",
                  "DEMS", "CANOPIES", "MASKS"]
    settings = Settings()
    default = Settings().reset()
    for key in keys:
        if not key in valid_keys:
            warn(f"Key {key} cannot be cleared")
            continue
        settings.set(key, default.get(key))
        settings.save()

@click.command()
@click.argument("key")
@click.argument("files", nargs=-1)
@click.option("--antenna", help="Antenna type", default=None)
@click.option("--radome", help="Radome type", default="NONE")
def add(key, files, antenna: str|None, radome: str) -> None:
    """Add files or folders to TomoSAR. Valid keys are:
    DEM, DEMS, CANOPY, CANOPIES, MASK, MASKS, RECEIVER"""
    settings = Settings()
    settings.add(key, files, antenna=antenna, radome=radome)
    settings.save()

@click.command()
@click.argument("key")
@click.argument("files", nargs=-1)
@click.option("--antenna", help="Antenna type", default=None)
@click.option("--radome", help="Radome type", default="NONE")
def remove(key, files, antenna: str|None, radome: str) -> None:
    """Remove files or folders from TomoSAR"""
    settings = Settings()
    settings.remove(key, files, antenna=antenna, radome=radome)
    settings.save()
