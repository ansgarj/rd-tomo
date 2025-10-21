import click
import compileall
import shutil
import re
from pathlib import Path
import subprocess

from .. import __version__
from ..utils import warn
from ..binaries import check_required_binaries, run
from ..config import PACKAGE_PATH, PROJECT_PATH, SETTINGS_PATH, save_default

def warm_cache():
    """Pre-warm __pycache__ by compiling all modules."""

    compileall.compile_dir(PACKAGE_PATH, force=True, quiet=1)
    print(f"__pycache__ warmed for {PACKAGE_PATH}")

def pyproject_changed() -> bool:
    """Checks wheteher pyproject.toml was changed in the last merge."""
    try:
        # Run the git diff-tree command
        result = run(["git", "diff-tree", "-r", "--name-only", "--no-commit-id", "ORIG_HEAD", "HEAD"])

        # Check if pyproject.toml is in the output
        changed_files = result.stdout.splitlines()
        return "pyproject.toml" in changed_files
    
    except RuntimeError:
        return False

def parse_version_string(version_str: str):
    """
    Parses a complex version string into a tuple and extracts the commit hash.
    Example: '0.0.2.post1.dev0+g5ab868b42.d20251021'
    Returns:
        version_tuple: tuple of version components
        commit_id: extracted commit hash or None
    """
    base_part, _, local_part = version_str.partition('+')
    version_parts = base_part.split('.')

    version_tuple = []
    for part in version_parts:
        if part.isdigit():
            version_tuple.append(int(part))
        else:
            version_tuple.append(part)

    commit_id = None
    if local_part:
        for subpart in local_part.split('.'):
            if subpart.startswith('g'):
                commit_id = subpart[1:]
                version_tuple.append(commit_id)
            elif subpart.startswith('d'):
                version_tuple.append(subpart[1:])
            else:
                version_tuple.append(subpart)

    return tuple(version_tuple), commit_id

def update_version_file(version_str: str):
    """
    Updates version.py with the new version string, version tuple, and commit ID.
    """
    path = PACKAGE_PATH / "version.py"

    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    lines = path.read_text().splitlines()
    new_lines = []

    version_tuple, commit_id = parse_version_string(version_str)
    for line in lines:
        if line.startswith("__version__ =") or line.startswith("version ="):
            new_lines.append(f"__version__ = version = '{version_str}'")
        elif line.startswith("__version_tuple__ =") or line.startswith("version_tuple ="):
            new_lines.append(f"__version_tuple__ = version_tuple = {version_tuple}")
        elif line.startswith("__commit_id__ =") or line.startswith("commit_id ="):
            new_lines.append(f"__commit_id__ = commit_id = {repr(commit_id)}")
        else:
            new_lines.append(line)

    path.write_text("\n".join(new_lines) + "\n")
 

@click.command()
def setup() -> None:
    """Performs TomoSAR setup"""
    post_merge_path = PROJECT_PATH / ".git" / "hooks" / "post-merge"
    pre_push_path = PROJECT_PATH / ".git" / "hooks" / "pre-push"
    if not post_merge_path.exists():
        shutil.copy2(PROJECT_PATH / "setup" / "post-merge", post_merge_path)
        print("Project post-merge hook installed.")
    if not pre_push_path.exists():
        shutil.copy2(PROJECT_PATH / "setup" / "pre-push", pre_push_path)
        print("Project pre-push hook installed.")
    if not SETTINGS_PATH.exists():
        save_default()
        print("Default settings enabled (run tomosar settings to view)")
    if pyproject_changed:
        warn("TomoSAR project installation file updated. Run 'pip install -e /path/to/project'")
    version = subprocess.check_output(["hatch", "version"]).decode().strip()
    if version != __version__:
        update_version_file(version)
        print(f"TomoSAR version updated to: {version}")
    check_required_binaries()
    warm_cache()

@click.command()
def dependencies() -> None:
    """Scan PATH for required binaries"""
    check_required_binaries()

@click.command()
def warmup() -> None:
    """Pre-warm __pycache__ by compiling all modules"""
    warm_cache()