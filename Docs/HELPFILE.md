# General CLI help

All CLI tools in the **TomoSAR** toolbox are accessed as subcommands of the `tomosar` command which is installed by `pip` when you install the `tomosar` module.  Run `tomosar manual` to print this manual. You can also add `--help` to `tomosar` or any of the subcommands for basic syntax.

## `setup`
Running `tomosar setup` ensures that your current installation is up to date. It makes sure all Git hooks are installed, checks if the `pyproject.toml` file was changed in the last merge (or last commit if no merge exists) and warns you to reinstall via `pip` if it was, updates the version file, checks if all required binaries can be found in the `PATH` and pre-warms the Python cache.

You can perform the dependency check independently by running `tomosar depedencies` and you can always pre-warm the Python cache by running `tomosar warmup`.\

## `version`
Running `tomosar version` prints the current installed version. If the dynamic versioning differs, it will warn and prompt you to run `tomosar setup` to update the versioning. This can safely be ignored if you are working on your own code. 

## `settings`
Running `tomosar settings` prints the current settings. These are stored locally. The commands `tomosar set/clear/add/remove` allow you to modify your settings, and `tomosar default` resets to default values. The VERBOSE setting is toggled by `tomosar verbose`. 

**Note**: `null` values indicate an internal file. You can change this to use your own file, and reset by `tomosar clear`. 

## `trackfinder`
Running `tomosar trackfinder` on a `.moco` CSV file correctly identifies all tested flight timestamps and generates the `radar[...].inf` file for spiral flight processing. Can be used to generate the correct timestamps for linear flights by `trackfinder -l` or more generally `trackfinder -l X`. 

## `forge`
Running `tomosar forge` scans paths for slice files and intelligently combines them into _Tomogram Directories_, ordered by actual spiral flight and radar band. If multiple tomograms can be constructed for a single spiral and radar band, `tomosar forge` will warn and construct the first tomogram it finds. This is because the name of the _Tomogram Directories_ generated contain an ID constructed from the flight timestamp and the Spiral ID, along with an optional tag. Multiple tomograms for the same spiral and band would therefore overwrite each other. To filter slices see `tomosar forge --help`.

**Note**: this function will be included in the planned  `tomosar process` command, but will always remain a standalone tool as well.

## `fetch-swepos`
Running `tomosar fetch-swepos` on a drone `gnss_logger_dat-[...].bin` file or RINEX observation file produced from it will find and download matching RINEX observation files from the nearest station in the _Swepos_ network.

**Note**: this can be used as a fallback if we lack GNSS base station files.

## `station-ppp`
Running `tomosar station-ppp` on a GNSS base station RINEX observation file will run static PPP post-processing on the observation file to determine a better approximation of its position than what is provided in the RINEX header, and will update the header by default.

**Note**: this function, while operational, is currently buggy and does not deliver good results. When fixed it can be used as a fallback if we lack an "exact" position of the GNSS base station.

## `load`
Running `tomosar load` loads a single _Tomogram Directory_ or multiple _Tomogram Directories_ into a `TomoScenes` object, and then opens an interactive Python console with the `TomoScenes` object stored under `tomos`. It can be used as an entry point instead of having to manually import and run inside Python where path auto-completion may not work. 

## `sliceinfo`
Running `tomosar sliceinfo` scans a directory for slice files and collects them into a `SliceInfo` object, and then opens an interactive Python console with the `SliceInfo` object stored under `slices`. It can be used as an entry point instead of having to manually import and run inside Python where path auto-completion may not work. 

## `test`
The `tomosar test` subcommand contains additional subcommands that can be used to test e.g. config files or to verify that all third party binaries are operational. Currently the only test available is `tomosar test gnss`.

## Planned additions
1. `tomosar init` \[**NOT IMPLEMENTED**\] directly generates a processing directory from a _Data Directory_. It will identify what files are present, if necessary generate a `mocoref.moco` file from a CSV file or `.json` file or if necessary subsititute for a missing mocoref data by running `ppp` on the GNSS base station observation file, or subsitute for missing GNSS base station files by downloading rinex files from _Swepos_ using `swepos`. Then it will copy all necessary files into a processing directory located inside the folder pointed to by the `PROCESSING_DIRS` setting (**default**: `$HOME/Radar/Processing`). The generated directory will have the correct file structure for **Radaz** functions. Finally `tomprocess init` initiates preprocessing in the processing directory. **Note**: if run inside a processing directory, simply initiaties preprocessing.
2. `tomosar process` \[**NOT IMPLEMENTED**\] chains `slice` and `forge` to generate a _Tomogram Directory_, or content for one. 
3. `tomsoar slice` \[**NOT IMPLEMENTED**\] initiates a _backprojection_ loop to generate all slices for the specified tomogram.
4. `tomosar view` \[**NOT IMPLEMENTED**\] contains multiple subcommands used for viewing tomograms, statistics, e.t.c 
5. `tomosar optimize` \[**NOT IMPLEMENTED**\] plans a flight for optimizing _nominal_ SAR parameters according to given restraints.
6. `tomosar plan` \[**NOT IMPLEMENTED**\] interactively models a _planned flight_ to allow validation of ideal SAR parameters across different tomograms (**Note**: this does not take into account flight instabilities that can occur during the actual flight).
7. `tomosar test station-ppp` \[**NOT IMPLEMENTED**\] tests base station PPP performance against ground truth as given in a `mocoref.moco` file.
8. `tomosar fetch-data` \[**NOT IMPLEMENTED**\] fetches the **most recent** _drone data_ and generates a _Data Directory_ inside the folder pointed to by the `DATA_DIRS` setting (**default**: `$HOME/Radar/Data`)
9. `tomosar generate-mocoref` \[**NOT IMPLEMENTED**\] generates a correctly formatted `mocoref.moco` file from a CSV file,  reading the columns named `Longitude`, `Latitude`, `Ellipsoidal height` and `Antenna height` (default column names from _Emlid Reach_, can be changed with `tomosar set MOCOREF_LATITUDE`, `MOCOREF_LONGITUDE`, `MOCOREF_HEIGHT` and `MOCOREF_ANTENNA`) for mocoref data. If multiple lines are present in the CSV files it will by default read the first line (modify by `--line X`).
10. `tomoprocess analysis` \[**NOT IMPLEMENTED**\] analyzes the spiral flights and models them. Used to verify _idealized flight_ vs. _planned flight_, and to inspect _realized flight_ parameters, including anisotropies from flight instabilities. Can provide optimal processing parameters for `tomo`/`slice`. 
