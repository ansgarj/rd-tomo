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
@click.option("--version", is_flag=True, help="Print rdtomo version and exit")
@click.pass_context
def rdtomo(ctx: click.Context, version: bool) -> None:
    """Entry point for rdtomo CLI tools"""
    if version:
        try:
            hatch_version = subprocess.check_output(["hatch", "version"]).decode().strip()
            if hatch_version != __version__:
                update_version_file(hatch_version)
            click.echo(f"rd-tomo version: {hatch_version}")
        except subprocess.CalledProcessError:
            click.echo("Error retrieving version", err=True)
        ctx.exit()

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@rdtomo.command()
def manual() -> None:
    """Prints the Docs/HELPFILE.md file."""

    with open(PROJECT_PATH / "Docs" / "HELPFILE.md", "r", encoding="utf-8") as f:
        readme_content = f.read()

    # Create a console and render the markdown
    console = Console()
    markdown = Markdown(readme_content)
    console.print(markdown)

@rdtomo.command()
def changelog() -> None:
    """Prints the CHANGELOG.md file."""

    with open(PROJECT_PATH / "CHANGELOG.md", "r", encoding="utf-8") as f:
        changelog_content = f.read()

    # Create a console and render the markdown
    console = Console()
    markdown = Markdown(changelog_content)
    console.print(markdown)

# Setup
rdtomo.add_command(setup)
rdtomo.add_command(dependencies)
rdtomo.add_command(warmup)

## Settings
rdtomo.add_command(settings)
rdtomo.add_command(default)
rdtomo.add_command(verbose)
rdtomo.add_command(set)
rdtomo.add_command(clear)
rdtomo.add_command(add)
rdtomo.add_command(remove)

## Processing chain
rdtomo.add_command(mocoref)
rdtomo.add_command(extract_reach)
rdtomo.add_command(station_ppp)
rdtomo.add_command(fetch_swepos)
rdtomo.add_command(init)
rdtomo.add_command(trackfinder)
rdtomo.add_command(forge)

## Python interactive console entry ponts
rdtomo.add_command(sliceinfo)
rdtomo.add_command(load)

## Tests
rdtomo.add_command(test)

# Dev tools
rdtomo.add_command(dev)