import click
from pathlib import Path
import os
import time as Time
from datetime import datetime
import json
from matplotlib import pyplot as plt

from ..dem import make_ellipsoidal as fmake_ellipsoidal

@click.group(invoke_without_command=True)
@click.pass_context
def dem(ctx: click.Context):
    """Entry point for CLI tools for managing DEMs."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@dem.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("-o", "--out", type=click.Path(path_type=Path), default=None, help="Specify out path (default: from input)")
@click.option("--rf", default=None, help="Specify Reference Frame (default: TARGET_FRAME)")
def make_ellipsoidal(path: Path, out: None|Path, rf: None|str):
    """Converts target DEM from orthometric to ellipsoidal height."""

    if out is None:
        out = path.parent / (path.stem + "_ellipsoidal" + path.suffix)
    fmake_ellipsoidal(path, out=out, rf=rf)