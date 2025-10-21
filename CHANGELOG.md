# Changelog

All notable changes to this project will be documented in this file.

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