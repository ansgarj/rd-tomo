from importlib.resources import path as importpath, files as importfiles
from contextlib import contextmanager, ExitStack
from typing import Iterator
from pathlib import Path
import re
import shutil
import json
import subprocess

from .config import Settings, LOCAL
from .utils import warn, local, changed

# Custom exception for missing directories
class DirNotFoundError(FileNotFoundError):
    """Raised when a required directory is not found."""
    pass

class DirExistsError(FileExistsError):
    """Raised when a directory already exists."""
    pass


# Check if required binary is installed
def require_binary(name: str, hint: str|None = None) -> str:
    try:
        dep = load_dependencies()[name]
    except:
        raise RuntimeError(f"Failed to find required binary '{name}' in rdtomo.setup.dependencies.json")
    path = shutil.which(name)
    if path is None:
        if not hint:
            raise RuntimeError(f"Required binary '{name}' not found in PATH.\n\033[1mSource\033[22m: {dep['Source']}")
        else:
            raise RuntimeError(f"Required binary '{name}' not found in PATH.\n{hint}")
    return path

def load_dependencies() -> dict[str,dict]:
    path = importfiles("rdtomo.setup").joinpath("dependencies.json")
    with open(path) as f:
        return json.load(f)
    
def check_required_binaries() -> None:
    deps = load_dependencies()
    missing = []
    for name, dep in deps.items():
        try:
            hint = json.dumps(dep, indent=4)
            require_binary(name, hint=hint)
        except RuntimeError as e:
            print(f"[Missing] {e}")
            missing.append(name)
    if missing:
        print(f"{len(missing)} missing binaries: {missing}")
        return
    else:
        print("All required binaries are available.")
 
# Catch-all run command for binariy executables
def run(cmd: str | list, capture: bool = True) -> subprocess.CompletedProcess:
    
    if not isinstance(cmd, list):
        cmd = [cmd]
    binary_name = cmd[0]
    require_binary(binary_name)
    cmd = [local(part) if isinstance(part, Path) else str(part) for part in cmd]
    if Settings().VERBOSE:
            print(' '.join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=capture, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed with exit code {e.returncode}.\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        ) from e

# Get resource file
def substitute_flags(path: Path, resolved: dict):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(r"\{\{(\w+)\}\}")

    def replacer(match: re.Match|None) -> str:
        return resolved.get(match.group(1), match.group(0))

    substituted = pattern.sub(replacer, content)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(substituted)

@contextmanager
def _resolve_resources(keys: set[str], **kwargs) -> Iterator[dict]:
    """Context manager that resolves multiple resource keys and keeps them alive."""
    with ExitStack() as stack:
        resolved = {}
        for key in keys:
            if (path := kwargs.get(key.casefold())) and Path(path).is_file():
                resolved[key] = str(path)
            else:
                try:
                    ctx = resource(None, key, **kwargs)
                    path = stack.enter_context(ctx)
                    resolved[key] = str(path) if path and path.is_file() else ""
                except Exception as e:
                    warn(f"Could not resolve resource for key '{key}': {e}")
                    resolved[key] = ""
        yield resolved

@contextmanager
def resource(path: str | Path | None, key: str = None, **kwargs) -> Iterator[Path]:
    """Provides a temporary local copy of the file specified in path. If path is empty or None,
    provides a temporary local copy of a file specified in settings or fetched internally based on key.

    Will resolve patterns matching {{KEY}} inside files to point to matching files, which are also provided as temporary copies.

    If no file is provided None is yielded.

    Valid keys: 'SATELLITES', 'SWEPOS_COORDINATES', 'RTKP_CONFIG', 'RECEIVER' 'DEM'', 'CANOPY', 'TEST_FILE_SVB', 'TEST_FILE_SAVAR'
                'NKG_VEL, NKG_CORR'
    NOTE: RTKP_CONFIG takes satellites (optional) and receiver (optional) keywords 
    NOTE: RECEIVER takes antenna (required) and radome (optional) keywords
    NOTE: DEM and CANOPY take bounds (required) and res (required) keywords"""
    # Generate a local copy of a file
    def local_copy(file: Path) -> Path:
        if file.is_file():
            tmp_path = Path.cwd() / file.with_suffix(".tmp").name
            shutil.copy2(file, tmp_path)
            return tmp_path
        raise FileNotFoundError(f"{file} not found")
    
    # Prepare tmp_path
    tmp_path = None
    
    # Check if path was specified
    if path:
        path = Path(path)
        if path.exists() and path.is_file():
            tmp_path = local_copy(path)
        else:
            if key:
                warn(f"Path {path} does not point to a file. Ignoring.")
            else:
                raise RuntimeError(f"Path {path} does not point to a file.")
    
    if key:
        key = key.upper()
    # Match against file keys for Settings reference or internal pointers    
    match key:
        case "SATELLITES":
            filename = Settings().SATELLITES
            if filename:
                path = Path(filename)
                if path.exists():
                    tmp_path = local_copy(path)
                else:
                    raise ValueError(f"Path {path} read from settings {key} does not point to a file.")
        case "SWEPOS_COORDINATES":
            filename = Settings().SWEPOS_COORDINATES
            if filename:
                path = Path(filename)
                if path.is_file():
                    tmp_path = local_copy(path)
                else:
                   raise ValueError(f"Path {path} read from settings {key} does not point to a file.") 
        case "RTKP_CONFIG":
            filename = Settings().RTKP_CONFIG
            if filename:
                path = Path(filename)
                if path.is_file():
                    tmp_path = local_copy(path)
                else:
                    raise ValueError(f"Path {path} read from settings {key} does not point to a file.")
        case "RECEIVER":
            antenna = kwargs.get("antenna", None)
            if antenna is None:
                raise KeyError("To get a RECEIVER resource, antenna must be specified (antenna=)")
            radome = kwargs.get("radome", "NONE")
            filename = Settings().RECEIVER(receiver_id=antenna, radome=radome)
            if filename:
                path = Path(filename)
                if path.exists():
                    tmp_path = local_copy(path)
                else:
                    warn(f"Path {path} read from settings does not point to a file. Ignoring")
        case "DEM":
            bounds = kwargs.get("bounds", None)
            if bounds is None:
                raise KeyError("To get a DEM resource, bounds must be specified (bounds=)")
            res = kwargs.get("resolution", None)
            if res is None:
                raise KeyError("To get a DEM resource, resolution must be specified (resolution=)")
            vrt_path = LOCAL / "DEM.vrt"
            vrt_path = build_vrt(vrt_path, Settings().DEMS)
            tmp_path = generate_raster(vrt_path, bounds, res)   
        case "CANOPY":
            bounds = kwargs.get("bounds", None)
            if bounds is None:
                raise KeyError("To get a CANOPY resource, bounds must be specified (bounds=)")
            res = kwargs.get("resolution", None)
            if res is None:
                raise KeyError("To get a CANOPY resource, resolution must be specified (resolution=)")
            vrt_path = LOCAL / "CANOPY.vrt"
            vrt_path = build_vrt(Settings().CANOPIES)
            tmp_path = generate_raster(vrt_path)
        case "TEST_FILE_SVB":
            with importpath('rdtomo.test_files',"minimal_svb.ubx") as test_file:
                tmp_path = local_copy(test_file)
        case "TEST_FILE_SAVAR":
            with importpath('rdtomo.test_files',"minimal_savar.ubx") as test_file:
                tmp_path = local_copy(test_file)
        case "NKG_VEL":
            with importpath('rdtomo.resources', 'eur_nkg_nkgrf17vel.tif') as raster:
                tmp_path = local_copy(raster)
        case "NKG_CORR":
            with importpath('rdtomo.resources', 'no_kv_NKGETRF14_EPSG7922_2000.tif') as raster:
                tmp_path = local_copy(raster)
    # Find all {{KEY}} in the file
    try:
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        keys = set(re.findall(r"\{\{(\w+)\}\}", content))
    except (TypeError, UnicodeDecodeError):
        keys = {}

    # Resolve and substitute
    if keys:
        with _resolve_resources(keys, **kwargs) as resolved:
            substitute_flags(tmp_path, resolved)
            # Yield temporary local path
            try:
                yield tmp_path
            finally:
                if isinstance(tmp_path, Path): # Not None
                    tmp_path.unlink(missing_ok=True)  # Clean up after use
    else:
        # Yield temporary local path
        try:
            yield tmp_path
        finally:
            if isinstance(tmp_path, Path): # Not None
                tmp_path.unlink(missing_ok=True)  # Clean up after use

@contextmanager
def tmp(*args, temporary: bool = True, allow_dir: bool = False) -> Iterator[Path]:
    """Makes a path or several paths temporary within the context. If allow_dir is False, the path must point
    to a file or not exist; if the path does not exist, it will still be unlinked after the context ends (allowing
    file generation within the context). If allow_dir is True and path does not exist when tmp is called,
    a temporary directory matching the path will be generated. If the path points to a directory, ALL content
    is also made temporary.
    
    If a falsy value is passed tmp will yield None."""
    def unlink(paths: tuple[Path, ...]) -> None:
        """Helper function to unlink a tuple of paths"""
        for path in paths:
            if path.is_file():
                path.unlink()
            if path.is_dir():
                shutil.rmtree(path)

    if not args:
        yield None
    paths = tuple(Path(path) for path in args)
    for path in paths:
        if not allow_dir and path.is_dir():
            raise ValueError("To allow directories to be made temporary, specify allow_dir=True.")
        if allow_dir and not path.exists():
            try:
                path.mkdir(parents=False, exist_ok=True)
            except FileNotFoundError:
                raise DirNotFoundError(f"To create the temporary directory {path}, ensure that the parent folder {path.resolve().parent} exists first.")
    try:
        yield paths[0] if len(paths) == 1 else paths
    finally:
        if temporary:
            unlink(paths)

# VRT
def build_vrt(vrt_path: Path|str, paths: list[Path|str]) -> Path:
    """
    Build a .vrt mosaic from all .tif files, returning the file path if a single GeoTIFF file is passed
    """
    tif_files = []
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix in (".tif", ".tiff"):
            tif_files.append(path)
        else:
            new_files = list(path.rglob("*.tif"))
            new_files.extend(list(path.rglob(".tiff")))
            tif_files.extend(new_files)
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFF files found in {paths}")
    if len(tif_files) == 1:
        return tif_files[0]
    
    # Check if the current VRT matches input files and build new VRT otherwise
    vrt_path = Path(vrt_path)
    hash_path = vrt_path.with_suffix(".hash")
    if changed(hash_path, tif_files):
        print("Building new VRT ...", end=" ", flush=True)
        cmd = ["gdalbuildvrt", "-resolution", "highest", vrt_path] + [f for f in tif_files]
        run(cmd)

def generate_raster(
        vrt_path: Path|str,
        bounds: tuple[float, float, float, float],
        res: tuple[float, float]
    ) -> Path:

    out_path = Path.cwd() / Path(vrt_path).with_suffix(".tmp").name
    cmd = ["gdalwarp", "-te", *bounds, "-tr", *res, "-r", "average", "-of", "GTiff", vrt_path, out_path]
    run(cmd)
    return out_path

# Config editing
def modify_config(
        config_path: Path,
        standard: bool = False,
        precise: bool = False,
        raw: bool = False,
        ionofile: Path|str|None = None
    ) -> None:
    """
    Modifies an existing RTKLIB config file to enable precise ephemeris mode
    and sets the paths to SP3 and CLK files.

    :param standard: disable Explorer specific option
    :param precise: enable precise ephemeris mode
    :param raw: disable all internal rdtomo functions and run raw
    :param ionofile: path to Ionosphere map file
    """
    
    # Read the original config file
    with open(config_path, 'r') as file:
        lines = file.readlines()

    # Flags to check if the required field is found
    if precise:
        sateph_found = False
    if ionofile:
        iono_found = False
    
    # Pattern to find files that have been added beforehand
    if raw:
        pattern = re.compile(r"/.+\n")
        armode_found = False
    
    # Modify the relevant lines
    for i, line in enumerate(lines):
        # Modify fields if found
        if precise:
            if line.strip().startswith('pos1-sateph'):
                lines[i] = 'pos1-sateph        =1\n'
                sateph_found = True
        if standard:
            # Remove explorer specific fields
            if "# This requires the Explorer version of RTKLIB" in line:
                lines[i] = ''
        if raw:
            # Remove internal resources
            lines[i] = pattern.sub('\n', line)
            if "pos2-armode" in line:
                # Set AR-mode to fix-and-hold due to lower constraints
                lines[i] = "pos2-armode        =fix-and-hold # (0:off,1:continuous,2:instantaneous,3:fix-and-hold)\n"
                armode_found = True
        if ionofile:
            if line.strip().startswith('file-ionofile'):
                lines[i] = f'file-ionofile      ={ionofile}\n'
                iono_found = True

    # Append missing fields if not found
    if precise and not sateph_found:
        lines.append('pos1-sateph        =1\n')

    if ionofile and not iono_found:
        lines.append(f'file-ionofile      ={ionofile}\n')

    if raw and not armode_found:
        lines.append("pos2-armode        =fix-and-hold # (0:off,1:continuous,2:instantaneous,3:fix-and-hold)\n")

    # Write the modified config to the output file
    with open(config_path, 'w') as file:
        file.writelines(lines)

# Manage read only states
import stat

def read_only(path: Path|str) -> Path:
    path = Path(path)
    for p in path.rglob('*'):
        if p.is_dir():
            p.chmod(stat.S_IREAD | stat.S_IEXEC)
        else:
            p.chmod(stat.S_IREAD)

    return path

def set_writable(path: Path|str) -> None:
    for p in path.rglob('*'):
        if p.is_dir():
            p.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        else:
            p.chmod(stat.S_IREAD | stat.S_IWRITE)

from typing import Callable, TypeVar
F = TypeVar("F", bound=Callable[..., object])

@contextmanager
def writable(path: Path | str):
    """
    Context manager to make a directory writable during the block,
    then restore read-only state.
    """
    path = Path(path)
    set_writable(path)
    try:
        yield path  # Provide the path to the block if needed
    finally:
        read_only(path)
