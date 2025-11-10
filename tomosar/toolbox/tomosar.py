import click
from rich.console import Console
from rich.markdown import Markdown
import subprocess

from .. import __version__
from ..config import PROJECT_PATH
from .setup_tools import dependencies, setup, warmup, update_version_file
from .settings_tools import settings, default, verbose, add, set, clear, remove
from .interact_tools import load, sliceinfo
from .processing_tools import forge, trackfinder, station_ppp, fetch_swepos, mocoref, extract_reach, init
from .test_tools import test

# Dev tools
from .dev_tools import dev

@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Print TomoSAR version and exit")
@click.pass_context
def tomosar(ctx: click.Context, version: bool) -> None:
    """Entry point for TomoSAR CLI tools"""
    if version:
        try:
            hatch_version = subprocess.check_output(["hatch", "version"]).decode().strip()
            if hatch_version != __version__:
                update_version_file(hatch_version)
            click.echo(f"TomoSAR version: {hatch_version}")
        except subprocess.CalledProcessError:
            click.echo("Error retrieving version", err=True)
        ctx.exit()

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@tomosar.command()
def manual() -> None:
    """Prints the Docs/HELPFILE.md file."""

    with open(PROJECT_PATH / "Docs" / "HELPFILE.md", "r", encoding="utf-8") as f:
        readme_content = f.read()

    # Create a console and render the markdown
    console = Console()
    markdown = Markdown(readme_content)
    console.print(markdown)

@tomosar.command()
def changelog() -> None:
    """Prints the CHANGELOG.md file."""

    with open(PROJECT_PATH / "CHANGELOG.md", "r", encoding="utf-8") as f:
        changelog_content = f.read()

    # Create a console and render the markdown
    console = Console()
    markdown = Markdown(changelog_content)
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
tomosar.add_command(mocoref)
tomosar.add_command(extract_reach)
tomosar.add_command(station_ppp)
tomosar.add_command(fetch_swepos)
tomosar.add_command(init)
tomosar.add_command(trackfinder)
tomosar.add_command(forge)

## Python interactive console entry ponts
tomosar.add_command(sliceinfo)
tomosar.add_command(load)

## Tests
tomosar.add_command(test)

# Dev tools
tomosar.add_command(dev)