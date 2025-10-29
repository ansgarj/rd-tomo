# Changelog

All notable changes to this project will be documented in this file.

## [0.0.5] - 2025-10-29

### Added
- Added a named function `tomosar.reach2rnx` for converting Emlid reach RTCM3 files to RINEX in preparation of `tomosar init`
- Added Makefile with `make install` (run after clone to install) and `make update` commands (run to update installation if `pyproject.toml` changes: runs automatically on most builds) \[**Requires**: `make`\]
- Added `--info` option to `tomosar load` to extract and print basic information about a `.tomo` directory and then exiting,without requiring the entire directory structure to be loaded
- Added `--recursive` option to `tomosar sliceinfo`
- Added `tomosar.utils.mocoref()` function that reads mocoref data from data files (CSV, JSON, LLH or mocoref.moco) or from user input data
- Added `tomosar mocoref` command that generates a `mocoref.moco` file from a data file
- Added `CHCI83.atx` internal file with **unverified** calibration data for the CHCI83 receiver

### Changed
- Updated the `rtklib` dependency to suggest the Explorer fork instead
- `tomosar test gnss` is now fully operational
- Moved `tomosar version` to `tomosar --version`
- Updated default settings to show the path to the default internal file instead of `null` (this also means that the `tomosar.resource()` manager now takes all paths from the settings)
- Reduced 12h buffer on ephemeris files to 6h
- Renamed the `tomosar.processing` submodule to `tomosar.tomogram_processing` for clarity
- Updated `tomosar.rnx2rtkp` to use `tomosar.utils.mocoref()` to read the `mocoref_file` and added relevant options
- Updated internal `rnx2rtkp` config file to set antenna type to be read from the RINEX header (along with antenna deltas)
- Changed internal `rnx2rtkp` config file to use all available constellations, but modified the function to limit this by calibration data if applicable (similar to how `gLAB` is used)
- `rnx2rtkp` also uses the NONE radome as fallback, just as `gLAB`: this is achieved by a temporary copy of the base observation file

### Fixed
- Fixed another bug in `tomosar setup` that still caused it to warn that the `pyproject.toml` file had changed and migrated to hash control instead of relying on `git diff-tree`
- Fixed a bug in `tomosar clear` that caused the default settings to be read as `None`
- Improved `gLAB` interaction: it can now be considered stable (but not perfect)
- Fixed bug in `tomosar forge` and `tomosar.forging.tomoforge()` that caused it to miss slices at the root directory of specified paths

## [0.0.4] - 2025-10-21

### Changed
- Updated `pyproject.toml` to indicate that at least Python 3.10 is needed 

## Fixed
- Fixed bug in `tomosar trackfinder` that was caused by a leftover option from v0.0.1 and which caused a fatal error
- Fixed bug in `tomosar trackfinder` that caused it not to write a radar_logger_dat-[...].inf file by default
- Fixed bug in `tomosar setup` that caused it to warn that the `pyproject.toml` file had changed even on a fresh clone if the last commit pushed to the online repo contained a change.

## [0.0.3] - 2025-10-21

### Added
- `--nav` option to `tomoprocess swepos` to also fetch matching nav data and correspondingly in `tomosar.gnss.fetch_swepos`
- hatch dependency for dynamic version control

### Changed
- Streamlined CLI interface by collecting all commands under `tomosar`  (discontinued: `tomoprocess`, `tomotest` and `tomoview`) and other changes (see `tomosar manual` for more help)
- `tomosar setup` now dynamically updates the stored version file
- `tomosar version` warns if the dynamic version does not match the installed version (you may safely ignore if you are developing your own code)

### Fixed
- `tomoprocess trackfinder` and `tomosar.trackfinding.trackfinder` relied on submodule removed in v0.0.2, now uses the correct tools
- The attempt of `tomosar setup` to update the pip installation was causing a crash on Windows system. It now warns if the pyproject.toml file has been changed and prompts the user to run 'pip install -e /path/to/project' manually instead
- Removed `DEFAULT_POC` settings as a part of an ongoing change to fix interaction with gLAB which is incorrect at the moment.

## [0.0.2] - 2025-10-15

### Added
- Pre-push hook
- `SceneStats` class
- Named functions for running 3rd party dependencies
- CLI tools related to settings: tomosar settings, tomosar reset, tomosar verbose, tomosar set, tomosar clear, tomosar add, tomosar remove

### Changed
- Updated `tomosar setup` to install pre-push hook
- Added dict-like methods to Masks object
- Updated __init__ to match new named functions for 3rd party dependencies
- Moved from environment variables to local settings (`.local/settings.json`)
- Changed `data_path()` to general purpose resource context manager named `resource()`

### Fixed
- Bug in loading Masks

## [0.0.1] - 2025-10-09

### Added
- Initial alpha release