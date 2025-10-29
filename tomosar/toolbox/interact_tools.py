# Imports
import os
import click
from pathlib import Path
import json

from .. import tomoload, SliceInfo, tomoinfo
from ..utils import interactive_console

@click.command()
@click.argument("path", required=False, default='.', type=click.Path(exists=True, path_type=Path))
@click.option("-u", "--update", is_flag=True, help="Update cached masks")
@click.option("-i", "--info", is_flag=True, help="Print info on .tomo directory and exit")
@click.option("-n", "--npar", type=int, default=os.cpu_count(), help="Number of parallel threads for file reading")
@click.pass_context
def load(ctx: click.Context, path: Path, update: bool, info: bool, npar: int) -> None:
    """Loads a TomoScenes object into a Python terminal."""
    if info:
        info = tomoinfo(path)
        print(json.dumps(info, indent=4))
        ctx.exit()

    cached = not update
    # Call sliceinfo
    tomos = tomoload(path=path, cached=cached, npar=npar)
    interactive_console({"tomos": tomos})

@click.command()
@click.argument("paths", nargs=-1, required=False, default='.', type=click.Path(exists=True, path_type=Path))
@click.option("-R", "--recursive", is_flag=True, help="Collect slices recursively")
@click.option("-r", "--read", is_flag=True, help="Also read image data.")
@click.option("-n", "--npar", type=int, default=os.cpu_count(), help="Number of parallel threads for file reading.")
def sliceinfo(paths: list[Path], recursive: bool, read: bool, npar: int):
    """Loads a SliceInfo object into a Python terminal."""
    # Call sliceinfo
    slices = SliceInfo()
    for root_path in paths:
        info = SliceInfo.scan(path=root_path, read=read, npar=npar)
        if info:
            slices.extend(info)
        if recursive:
            for dirpath in root_path.rglob("*"):
                if dirpath.is_dir():
                    print(dirpath)
                    info = SliceInfo.scan(str(dirpath), read=False, filter=filter)
                    if info:
                        slices.extend(info)
    interactive_console({"slices": slices})