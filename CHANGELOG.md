# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `tomosar.gnss.modify_config` (see below)

### Changed
- `tomosar.binaries.tmp` no longer creates parents to temporary directories (which were not temporary), but fails if all parents do not exist
- `tomosar.gnss.station_ppp` now allows input of SP3 and/or CLK files instead of always downloading, and accepts full SP3 files as input; it also returns the SP3 and CLK paths if retain is used (but not rotation matrix and distance). The output directory for the files can also be specified without specifying a path for `.out` file by inputing a directory as `out_path`. 
- `tomosar.binaries.ppp` now also accepts full SP3 files as input (instead of only ORBIT/CLK pairs)
- Moved config file resource management from `tomosar.binaries.rnx2rtkp` to `tomosar.gnss.rtkp` and changed from multiple internal config files to dynamically updating the temporary config file copy by `tomosar.gnss.modify_config`
- `tomosar.gnss.rtkp` now allows input of SP3 and CLK files or downloading of SP3 and CLK files to run in precise mode
- `tomosar.binaries.rnx2rtkp` now allows input of SP3 and CLK files
- `tomosar.gnss.rtkp` now handles `out_path` and output directory similarly as `tomosar.gnss.station_ppp`, and if `out_path` is `None` then `tomosar.binaries.rnx2rtkp` captures output instead of writing a `.pos` file (**Note**: This means that the counter while it is running is not visible if no `out_path` is provided, but the total Q1 percentage is still displayed after finishing)
- Renamed `tomosar.gnss.read_pos_file` to `tomosar.gnss.read_rnx2rtkp_out` and `tomosar.gnss.read_out_file` to `tomosar.gnss.read_glab_out` and changed both to distinguish between file and stdout input by whether it is a string or a `pathlib.Path` object
- `tomosar init` now uses precise ephemeris data in its RTKP post processing by default

### Fixed
- Fixed bug in `tomosar --version` that caused it to sometimes display the previous version
- Fixed a fatal bug in `tomosar init` caused by a mixing up of the GNSS base and the `mocoref.moco` after the last update

## [0.0.6] - 2025-11-06

### Added
- `tomosar test station-ppp` which runs `station-ppp` and compares against ground truth as found in a mocoref data file, can also be run directly on Reach ZIP archives without unpacking them (does not require separate mocoref data)
- `tomosar.chc2rnx` for converting CHCI83 .HCN file to RINEX in case RINEX files are missing for some reason
- `tomosar.binaries.tmp` context manager that makes a path temporary (including ALL content for directories), and changed existing functions that downloaded files that were supposed to be temporary to use it
- `tomosar.gnss.reachz2rnx` and `tomosar extract-reach` that extracts RINEX OBS, `mocoref.moco` and optionally RINEX NAV files from a Reach ZIP archive
- `tomosar init` to collect files in a data directory and convert to the correct structure for processing, and copying or moving files into a processing directory in such a manner that the data directory is unaltered. Then initiate preprocessing in the processing directory. \[NOT IMPLEMENTED\]: IMU and IMU+GNSS integration and what follows.
- `tomosar.gnss.rtkp` which calls `tomosar.binaries.rnx2rtkp` and reads the `.pos` file

### Changed
- Changed `tomosar.gnss.station_ppp` to take explicit paths to the observation file and optionally the GLONASS navigation data, instead of a data directory, and correspondingly for `tomosar station-ppp`
- Moved the `local` function from `tomosar.binaries` to `tomosar.utils` and added support for lists
- Collected `pyproj.Transformer` objects for ECEF to geodetic and from geodetic to ECEF in `tomosar.transformers`
- Updated `tomosar.reach2rnx` to correctly handle multiple sites within a single RTCM3 log by splitting the output into multiple RINEX observation files.
- Renamed `tomosar.utils.mocoref` to `generate_mocoref` 
- `tomosar.gnss.station_ppp` and `tomosar station-ppp` now unlinks the ephemeris files after finishing as default (optionally retains them)
- `tomosar.binaries.merge_rnx` and `tomosar.binaries.merge_eph` no longer performs merge on a single file, but simply returns the single file

### Fixed
- Fixed bug in `tomosar forge` related to datetime filtering
- Fixed a bug in the `tomosar.resource` context manager related to resolving {{KEYS}} that are not set in the local settings
- Fixed a bug in `rnx2rtkp` related to the temporary copy when NONE radome fallback is used.
- Tweaked internal `rnx2rtkp` config to avoid false fixes
- Fixed `reach2rnx` to optionally take a referece (approximate) timestamp (to infer the correct GPS week): if not provided will infer it from the filename or raise an error if this fails

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