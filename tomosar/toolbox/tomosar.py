import click
from rich.console import Console
from rich.markdown import Markdown
import subprocess

from .. import __version__
from ..utils import warn
from ..config import PROJECT_PATH
from .setup_tools import dependencies, setup, warmup
from .settings_tools import settings, default, verbose, add, set, clear, remove
from .interact_tools import load, sliceinfo
from .processing_tools import forge, trackfinder, station_ppp, fetch_swepos
from .test_tools import test

# Dev tools
from .dev_tools import rnx_info, read_imu, inspect_out, compare_rtkp, minimize_ubx

@click.group()
def tomosar() -> None:
    """Entry point for TomoSAR CLI tools"""
    pass

@tomosar.command()
def version() -> None:
    """Print TomoSAR version"""
    version = subprocess.check_output(["hatch", "version"]).decode().strip()
    print(f"TomoSAR version: {__version__}")
    if version != __version__:
        warn(f"Dynamic version differs from installed version: {version}\nRun tomosar setup to update")

@tomosar.command()
def manual() -> None:
    """Prints the Docs/HELPFILE.md file"""

    with open(PROJECT_PATH / "Docs" / "HELPFILE.md", "r", encoding="utf-8") as f:
        readme_content = f.read()

    # Create a console and render the markdown
    console = Console()
    markdown = Markdown(readme_content)
    console.print(markdown)

# Setup
tomosar.add_command(setup)
tomosar.add_command(dependencies)
tomosar.add_command(warmup)

## Settings
tomosar.add_command(settings)
tomosar.add_command(default)
tomosar.add_command(verbose)
tomosar.add_command(set)
tomosar.add_command(clear)
tomosar.add_command(add)
tomosar.add_command(remove)

## Processing chain
tomosar.add_command(trackfinder)
tomosar.add_command(forge)
tomosar.add_command(station_ppp)
tomosar.add_command(fetch_swepos)

## Python interactive console entry ponts
tomosar.add_command(sliceinfo)
tomosar.add_command(load)

## Tests
tomosar.add_command(test)

# Dev tools
@tomosar.group(hidden=True)
def dev() -> None:
    """Entry point for dev tools"""
    pass

dev.add_command(rnx_info)
dev.add_command(read_imu)
dev.add_command(inspect_out)
dev.add_command(compare_rtkp)
dev.add_command(minimize_ubx)