# Imports
import os
import click
from pathlib import Path
import yaml
from contextlib import ExitStack

from .. import SliceInfo
from ..utils import interactive_console
from ..data import LoadDir, TomoArchive, TomoDir, ProcessingDir, DataDir

@click.command()
@click.argument("directories", nargs=-1, type=click.Path(exists=True, path_type=LoadDir), default=[LoadDir.cwd()])
@click.option("-p", "--paths", multiple=True, type=click.Path(exists=True, path_type=LoadDir), default=[], help="Add paths to the list of defined variables in the interactive console")
@click.option("-i", "--info", is_flag=True, help="Print info on .tomo directory and exit")
@click.pass_context
def load(ctx: click.Context, directories: list[LoadDir], paths: list[Path], info: bool) -> None:
    """Loads a directory into a Python terminal."""
    if info:
        info = {}
        for dir in directories:
            info[str(dir)] = dir.info

        print(yaml.dump(info, default_flow_style=False, sort_keys=False, indent=4))
        ctx.exit()

    with ExitStack() as stack:
        datasets = {}

        for dir in directories:
            if isinstance(dir, DataDir):
                # Enter the context and keep it open until ExitStack closes
                data = stack.enter_context(dir.open())
                datasets[str(dir)] = data
                print(f"Loaded DataDir {dir} ...")
            elif isinstance(dir, ProcessingDir):
                dir.open()
                print(f"Loaded ProcessingDir {dir} ...")
            elif isinstance(dir, TomoDir):
                dir.open()
                print(f"Loaded TomoDir {dir} ...")
            elif isinstance(dir, TomoArchive):
                dir.open()
                print(f"Loaded TomoArchive {dir} ...")


        # Now all DataDir contexts are still active here
        vars = {
            "directories": directories,
            "datasets": datasets
        }
        # Add additional paths if any
        if paths:
            vars["paths"] = paths
        
        # Load interactive console
        interactive_console(vars)

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