# General CLI help

All CLI tools in the **TomoSAR** toolbox are accessed as subcommands of the `tomosar` command which is installed by `pip` when you install the `tomosar` module.  Run `tomosar manual` to print this manual and `tomosar changelog` to print the changelog. You can also add `--help` to `tomosar` or any of the subcommands for basic syntax.

## `setup`
Running `tomosar setup` ensures that your current installation is up to date. It makes sure all Git hooks are installed, checks if the `pyproject.toml` file has changed since last time `make install` or `make update` was run, and runs `make update` if necessary (or prompts you to do it if it fails), updates the version file, checks if all required binaries can be found in the `PATH` and pre-warms the Python cache.

If the Git hooks are installed (as they are if you run `make install` or `tomosar setup`) then `tomosar setup` is automatically run when you pull and push.

You can perform the dependency check independently by running `tomosar dependencies` and you can always pre-warm the Python cache by running `tomosar warmup`.

## `version`
Running `tomosar version` prints the current installed version. If the dynamic versioning differs, it will warn and prompt you to run `tomosar setup` to update the versioning. This can safely be ignored if you are working on your own code. 

## `settings`
Running `tomosar settings` prints the current settings. These are stored locally. The commands `tomosar set/clear/add/remove` allow you to modify your settings, and `tomosar default` resets to default values. The VERBOSE setting is toggled by `tomosar verbose`. 

**Note**: `null` values indicate an internal file. You can change this to use your own file, and reset by `tomosar clear`. 

## `init`
Running `tomosar init` Searches recursively to find matching files:
- Drone GNSS .bin and .log;
- Drone IMU .bin and .log;
- Drone Radar .bin, .log and .cfg;
- GNSS base station; and
- Mocoref data or precise position of GNSS base station.

If the GNSS base station file is missing, can fetch files from the nearest Swepos station, and can supplement Mocoref data by performing static PPP on the base station. Note that the path must point to a directory which contains exactly one set of drone data. For other files, tomosar init will use the first matching file it finds.

For the GNSS base station a RINEX OBS file is prioritized over other files: HCN files and RTCM3 files are also accepted, as well as Reach ZIP archives.

For mocoref data a mocoref.moco file is prioritized followed by a .json file, with the underlying assumption that these have been generated from raw mocoref data; then a .llh log is prioritized over a .csv file. If a Reach ZIP archive is used as the source of the GNSS base station file, the mocoref file will also be generated from there.

The files are converted where applicable and copied/moved into a processing directory, in such a way that the content of the data directory where tomosar init was initiated is left unaltered. Then preprocessing is initiated \[ONLY GNSS IMPLEMENTED\]. By default `tomosar init` will use precise ephemeris data for the RTKP post processing, and will download this data if not available (disable by running with `--broadcast`)

Note that tomosar init can also be run inside a processing directory, in which case it simply initiates preprocessing \[ONLY GNSS IMPLEMENTED\]. Any directory inside the settings specified PROCESSING_DIRS is assumed to be a processing directory, and any directory outside is by default assumed to be a data directory (this behaviour can be overridden by the --processing option).

## `trackfinder`
Running `tomosar trackfinder` on a `.moco` CSV file correctly identifies all tested flight timestamps and generates the `radar[...].inf` file for spiral flight processing. Can be used to generate the correct timestamps for linear flights by `trackfinder -l` or more generally `trackfinder -l X`. 

## `forge`
Running `tomosar forge` scans paths for slice files and intelligently combines them into _Tomogram Directories_, ordered by actual spiral flight and radar band. If multiple tomograms can be constructed for a single spiral and radar band, `tomosar forge` will warn and construct the first tomogram it finds. This is because the name of the _Tomogram Directories_ generated contain an ID constructed from the flight timestamp and the Spiral ID, along with an optional tag. Multiple tomograms for the same spiral and band would therefore overwrite each other. To filter slices see `tomosar forge --help`.

**Note**: this function will be included in the planned  `tomosar process` command, but will always remain a standalone tool as well.

## `fetch-swepos`
Running `tomosar fetch-swepos` on a drone `gnss_logger_dat-[...].bin` file or RINEX observation file produced from it will find and download matching RINEX observation files from the nearest station in the _Swepos_ network.

_Swepos_ stations usually benefit from a lower _elevation mask_ than the mobile GNSS: if you run `tomosar init --swepos` and don't specify the elevation mask manually, `tomosar init` will use a lower default, but if you fetch _Swepos_ data first manually with `tomosar fetch-swepos` then you may benefit from specifying a lower elevation mask when running `tomosar init`.

**Note**: this can be used as a fallback if we lack GNSS base station files.

## `station-ppp`
Running `tomosar station-ppp` on a GNSS base station RINEX observation file will run static PPP post-processing on the observation file to determine a better approximation of its position than what is provided in the RINEX header, and will update the header by default.

**Note**: this function, while operational, still needs some tweaking, but can be used as a fallback if we lack an "exact" position of the GNSS base station.

## `mocoref`
Running `tomosar mocoref` on a data file (CSV, JSON or LLH) generates a `mocoref.moco` file. For a CSV file it defaults to the first line, but this can be changed with `--line`, and reads columns with names matching the names specified in the settings. A JSON file is assumed to contain a dict with the keys specified in the settings. A LLH log is assumed to have no header and to have columns matching the LLH log from the Emlid Reach RS3. 

## `extract-reach`
Running `tomosar extract-reach` extracts a Reach ZIP archive to produce:
- A RINEX OBS file for a single site,
- A mocoref.moco for the OBS file, and
- A RINEX NAV file (optional).

Optionally takes a RINEX OBS file as input to extract from the archive the OBS file which has the greatest overlap with the input RINEX file. Otherwise extracts the longest segment.

## `load`
Running `tomosar load` loads a single _Tomogram Directory_ or multiple _Tomogram Directories_ into a `TomoScenes` object, and then opens an interactive Python console with the `TomoScenes` object stored under `tomos`. It can be used as an entry point instead of having to manually import and run inside Python where path auto-completion may not work.  Running `tomosar load --info` instead extracts and prints basic information without requiring the entire directory structure to be loaded, and then exits.

## `sliceinfo`
Running `tomosar sliceinfo` scans a directory for slice files and collects them into a `SliceInfo` object, and then opens an interactive Python console with the `SliceInfo` object stored under `slices`. It can be used as an entry point instead of having to manually import and run inside Python where path auto-completion may not work. 

## `test`
The `tomosar test` subcommand contains additional subcommands that can be used to test e.g. config files or to verify that all third party binaries are operational. Currently available are:
- `tomosar test gnss` which tests that all GNSS processing capabilities are operational
- `tomosar test station-ppp` which tests `station-ppp` against ground truth as found in a `mocoref` data file
- `tomosar test precise-rktp` which tests RTKP post processing with precise ephemeris data against broadcast ephemeris data (optionally with specified elevation mask)

## Planned additions
1. `tomosar process` \[**NOT IMPLEMENTED**\] chains `slice` and `forge` to generate a _Tomogram Directory_, or content for one. 
2. `tomsoar slice` \[**NOT IMPLEMENTED**\] initiates a _backprojection_ loop to generate all slices for the specified tomogram.
3. `tomosar view` \[**NOT IMPLEMENTED**\] contains multiple subcommands used for viewing tomograms, statistics, e.t.c 
4. `tomosar optimize` \[**NOT IMPLEMENTED**\] plans a flight for optimizing _nominal_ SAR parameters according to given restraints.
5. `tomosar plan` \[**NOT IMPLEMENTED**\] interactively models a _planned flight_ to allow validation of ideal SAR parameters across different tomograms (**Note**: this does not take into account flight instabilities that can occur during the actual flight).
6. `tomosar fetch-data` \[**NOT IMPLEMENTED**\] fetches the **most recent** _drone data_ and generates a _Data Directory_ inside the folder pointed to by the `DATA_DIRS` setting (**default**: `$HOME/Radar/Data`)
7. `tomoprocess analysis` \[**NOT IMPLEMENTED**\] analyzes the spiral flights and models them. Used to verify _idealized flight_ vs. _planned flight_, and to inspect _realized flight_ parameters, including anisotropies from flight instabilities. Can provide optimal processing parameters for `tomo`/`slice`. 