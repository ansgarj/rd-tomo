# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-11-28

### Note
- The new repo name is now `rd-tomo` and the Python module and CLI tool go under the common name `rdtomo` in order to avoid confusion with other TomoSAR related projects. 

### Added
- Implemented a unified Reference Frame workflow via `rdtomo.transformers.change_rf` and Settings to select which Reference Frame mocoref data is collected in and which Reference Frame is the target (output); implemented Reference Frames are ITRF2020, ETRF2020, SWEREF99, EUREF89, EUREF-FIN, EUREF-EST97, EUREF-DK94, LKS-94, LKS-92.
- `rdtomo.gnss.modify_config` (see below)
- `rdtomo test precise-rktp` which can be used to test precise RTKP mode against broadcast to determine which to use
- `rdtomo test rtkp` which tests the internal RTKP processing (with absolute antenna calibration) against raw RTKLIB `rnx2rtkp`, and optionally tests precise mode on internal or both
- Added automatic projected map coordinate detection to `rdtomo.trackfinder._get_azimuth`
- `rdtomo.data` module (see `rdtomo.resources` for old `rdtomo.data` module), with the following classes  `LoadDir(pathlib.Path)`, `DataDir(LoadDir)`, `ProcessingDir(LoadDir)`, `TomoDir(LoadDir)`, `TomoArchive(LoadDir)` and `DroneData`, where `LoadDir` dispatches to either `DataDir`, `ProcessingDir`, `TomoDir` or `TomoArchive`; the later classes function as interface with the respective directories; `DroneData` is a dataclass containing pointers to the files necessary for preprocessing and some helpful additionals
- `tomomsar.manager` module which contains basic file and resource manager functions (including the `resource` and `tmp` context managers, dependency checks and the basic `run` command for 3rd party binaries)
- `rdtomo.dem` module for DEM manipulation (resource management is handled in `rdtomo.manager`), currently mostly a placeholder

### Changed
- `rdtomo` now requires Python >=3.12 since it subclasses `pathlib.Path` directly
- The old `rdtomo.data` module is renamed to `rdtomo.resources`
- Removed `rdtomo.binaries`: some of its content is in `rdtomo.manager` and the gnss processing moved into `rdtomo.gnss` which now contains all GNSS processing and related functions (including `generate_mocoref`)
- `rdtomo.transformers` now uses a unified function interface instead of providing `pyproj.Transformer` objects
- The `rdtomo.transformers.geo_to_ecef` and `rdtomo.transformers.ecef_to_geo` now also optionally allows specifying of the Reference Frame, and correctly handles transformations within that Frame
- `rdtomo.manager.tmp` no longer creates parents to temporary directories (which were not temporary), but fails if all parents do not exist when `allow_dir=True`
- `rdtomo.gnss.station_ppp` now allows input of SP3 and/or CLK files instead of always downloading, and accepts full SP3 files as input as well as IONEX files; it also returns a dict with various data instead. The output directory for the files can also be specified without specifying a path for `.out` file by inputing a directory as `out_path`, or specified separately by the new `download_dir` parameter.
- `rdtomo.gnss.ppp` now also accepts full SP3 files as input (instead of only ORBIT/CLK pairs) as well as IONEX files
- Moved config file resource management from `rdtomo.gnss.rnx2rtkp` to `rdtomo.gnss.rtkp` and changed from multiple internal config files to dynamically updating the temporary config file copy by `rdtomo.gnss.modify_config`
- `rdtomo.gnss.rtkp` now optionally allows input of SP3 and CLK files or downloading of SP3 and CLK files to run in precise mode, and also allows input of IONEX files
- `rdtomo.gnss.rnx2rtkp` now allows input of SP3 and CLK files
- `rdtomo.gnss.rtkp` now handles `out_path` and output directory similarly as `rdtomo.gnss.station_ppp`, and if `out_path` is `None` then `rdtomo.gnss.rnx2rtkp` captures output instead of writing a `.pos` file (**Note**: This means that the counter while it is running is not visible if no `out_path` is provided, but the total Q1 percentage is still displayed after finishing)
- Renamed `rdtomo.gnss.read_pos_file` to `rdtomo.gnss.read_rnx2rtkp_out` and `rdtomo.gnss.read_out_file` to `rdtomo.gnss.read_glab_out` and changed both to distinguish between file and stdout input by whether it is a string or a `pathlib.Path` object
- `rdtomo init` now uses precise ephemeris data in its RTKP post processing by default
- Changed `rdtomo.trackfinding.analyze_spiral` to calculate flight altitude, radius and azimuth in the local ENU frame of the center
- `rdtomo.gnss.merge_ephemeris` now calls specific functions to splice SP3, CLK and IONEX files (`rdtomo.gnss.splice_sp3`, `rdtomo.gnss.splice_clk` and `rdtomo.gnss.splice_inx`)
- Renamed `rdtomo.gnss.fetch_sp3_clk` to `rdtomo.gnss.fetch_cod_files` and it now fetches COD Europe files for orbits (SP3), clock corrections (CLK) and Ionosphere maps (IONEX).
- Removed buffer on downloading ephemeris and Ionosphere files (`rdtomo.gnss.fetch_cod_files`)
- `Settings` and all classes in `rdtomo.core` now have `__slots__` preventing arbitrary attribute assignment
- `rdtomo.manager.resource` no longer takes the `standard` keyword for "RTKP_CONFIG"
- `rdtomo.manager.resource` now allows all flags to be passed as parameters (casefolded, i.e. lowercase), which takes precedence over the path specified by Settings
- `rdtomo.gnss.rtkp` now optionally takes `atx` and `receiver` keywords pointing to external ATX files
- `.tomo` directories should now be generated and maintained as read-only by `rdtomo` to avoid accidental modification of the internal structure
- DEMS, CANOPIES and MASKS in the Settings now have a dict of lists instead of a list as entries, indexed by the Reference Frame. When a path is added or removed, unless the Reference Frame is specified, it is added to or removed from the TARGET_FRAME. **Note**: `rdtomo` does not warp the files to convert them to another Reference Frame, but simply uses the dict as a mapping of where to find files in the correct Frame –– you must warp them yourselves if you plan to change Reference Frame.

### Fixed
- `rdtomo.gnss.station_ppp` (`rdtomo.gnss.ppp`) now achieves subcentimeter precision on a test RINEX OBS file of 4h length when the antenna has absolute callibration data, thanks to corrections in Reference Frame handling
- Fixed bug in `rdtomo --version` that caused it to sometimes display the previous version
- Fixed a fatal bug in `rdtomo init` caused by a mixing up of the GNSS base and the `mocoref.moco` after the last update
- `rdtomo.gnss.merge_ephemeris` and the splice functions it calls now merge ephemeris files without intermittent headers and EOF markers (which ensures that RTKLIB can parse them as well as gLAB)
- Fixed bug in `rdtomo.trackfinding.analyze_linear` that caused e.g. `rdtomo trackfinder` to incorrectly parse linear tracks

## [0.0.6] - 2025-11-06

### Note
- In v0.0.6 and older `rdtomo` went under the name of `tomosar`

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