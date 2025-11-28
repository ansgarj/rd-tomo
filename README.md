# README
This repository contains the `rdtomo` python module which is **in alpha development**, which also provides a selection of [CLI Tools](#core-cli-tools) that can be run directly from the terminal. It can be installed into your  `Python` environment via `pip`, and is constructed to be installed in **editable** mode.

If you clone the repository you will have local access to the code for experimentation and your own development (you can later create your own _branch_ and `git push` that branch and make _pull requests_ to merge your _branch_ to the `main` branch, see [Collaboration](#collaboration)), and you can `git pull` any updates and see the changes immediately reflected in your environment.

Included below is some basic GitHub usage, but there is also plenty of documentation online (e.g. pulling specific versions or branches).

## Installation
**NOTE**: It is recommended to use a Python virtual environment when installing the module. This allows you to have separate Python environments for different projects, avoiding any potential conflicts, and is also _required_ by some Linux distributions in order to avoid potential conflicts with _system Python_. It also makes it _easier to reproduce your setup_ on another computer should you wish to. This is automated if you follow these instructions.

Make sure you have the `make` utility available and run:
```sh
git clone https://github.com/ansgarj/rd-tomo.git
cd rd-tomo
make install
```


Among several other things, this will create a virtual environment inside a `.venv` folder inside the `TomoSar` project directory with `tomosar` installed. I find this helpful to contain the project in one directory. 

**NOTE**: add any other files or subdirectories inside the TomoSAR directory that should not be pushed to GitHub to `.gitignore` if I have not added them already.

**NOTE**: I have a simple shell function that I include into my `.bashrc` file to activate virtual environments by alias e.g.:
```sh
# Activate virtual environments by alias
activate() {
    local alias="$1"
        
    case "$alias" in 
        rd)
            source "$HOME/rd-tomo/.venv/bin/activate"
            ;;
        *)
            echo "Unknown project alias: $alias"
            return 1
            ;;
        esac
}
```
That way I can activate the virtual environment by running e.g. `activate rd` from any folder (say, where I have the radar data). 

### Required Binaries
The `rdtomo` module relies on some 3rd party software for GNSS processing. It is recommended to make sure you have everything set up before starting to use the toolbox, even though it is not _strictly_ necessary. If you run `make install` as suggested above, this is already done. Otherwise you can run `rdtomo dependencies` to perform the check independently. This will check if there are any binaries in your `PATH` with the correct names, but not actually test if it is the correct binaries, and provide helpful information if not (including source html links where applicable). The required binaries are:
1. `convbin` from `rtklib`
2. `rnx2rtkp` from `rtklib`
3. `crx2rnx` from GSI Japan
4. `gfzrnx` from GFZ Potsdam
5. `glab` from UPC/gAGE and ESA
6. `unimoco` from Radaz

You can also run `rdtomo test gnss` to test the GNSS processing capabilities, but this requires internet access and a valid login to the Swepos network.

**NOTE**: some of the binaries will have different names, e.g. `CRX2RNX` or `gLAB_linux` when downloaded, but the name provided above and by `rdtomo dependencies` are the names the `rdtomo` module uses and the binaries should either be _renamed_ and moved into the `PATH` _or_ you can create a _symlink_ with the correct name in the `PATH`. 

**NOTE**: _rd-tomo_ uses the `rdtomo.manager.resource` context manager to provide local copies of various files in the working directory (not radar, IMU or GNSS data which are expected to be inside the working directory tree). The basic reason for this is to allow the use of containers to run 3rd party binaries (such as docker). 

**NOTE**: I use the [Explorer](https://github.com/rtklibexplorer/RTKLIB) fork of `rtklib`, and this is the one _rd-tomo_ will suggest installing if it finds no `convbin` or `rnx2rtkp` binary, **but** it _should_ run on standard `rtklib` as well.

## Usage
The `rdtomo` _module_ is a work-in-progress to provide a one-stop toolbox for our tomographic SAR needs. Once installed it can be imported into Python by running `import rdtomo`, or you can select submodules or objects as usual  in Python. Currently, the only available documentation is the one present in the code, _but I hope to add separate documentation later._

The CLI tools are intended to provide a toolbox for the most common or predicted needs, the idea being that unless you are working on your own project with something not integrated into the CLI tools, you can use the module directly from the terminal by running a command without having to enter into Python and importing the module.  All tools are accessed as subcommands of `rdtomo` and can be called with `--help` for some basic syntax: `rdtomo --help` provides syntax help and `rdtomo manual` prints a general help overview.

**NOTE**: many of these tools are not yet fully implemented.

### Settings
The local _rd-tomo_ settings are stored inside the `.local` folder inside the project directory in a `settings.json` file. The current settings can be viewed with `rdtomo settings` which prints to stdout.

`FILES: ANTENNAS` can be used to point to antenna files containing absolute calibration data for the receiver antenna if not included in the `FILES: ANTENNAS: SATELLITES` file. Add files with `tomosar add RECEIVER` or change the `SATELLITES` file with `tomosar set SATELLITES`. **Note**: if a `RECEIVER` file is not specified for a specific antenna, _rd-tomo_ will look inside the `SATELLITES` file as a fallback. **Note 2**: The internal file for the CHCI83 receiver contains **unverified** calibration data (copied from GPS to other constellations), and causes PPP to fail.

Note that you can set login info for the Swepos network (so you don't have to specify manually) by `tomosar set SWEPOS_USERNAME` and `tomosar set SWEPOS_PASSWORD`, but that the password is stored inside `settings.json` in an **unencrypted state**. Use therefore with caution. 

Finally you can use `rdtomo add` to add files or folders to `FILES: DEMS`, `FILES: CANOPIES` and `FILES: MASKS`. These lists are used by _rd-tomo_ to find GeoTIFF files for DEM and canopy DSM references, and shape files for masking tomograms. The GeoTIFF files are used for slicing (`rdtomo process/slice` \[**NOT IMPLEMENTED**\]) with either the ground or the canopy as reference respectively. The shapefiles are used to generate masks by `rdtomo process/forge`, and can be updated for a [Tomogram Directory](#tomogram-directories) or multiple [Tomogram Directories](#tomogram-directories) by running `rdtomo load` and then inside the Python terminal identify the correct directory(-ies) (here X):
```python
directories[X].update()
directories[X].save()
```

**Note**: paths added to `DEMS`, `CANOPIES` and `MASKS` are in a specific Reference Frame (default: the REFERENCE_FRAMES: TARGET), and are used only when the REFERENCE_FRAME: TARGET coincide with the Frame of the path (they are stored in dicts indexed by Reference Frame).

### Data Directories
As a _data directory_ functions anything containing at least the drone data. There are no specific requirements on the _internal_ structure, but each data directory should contain **only one set of drone data** and matching base RINEX OBS files with mocoref data files (mocoref.moco, .csv, .llh or .json). Missing RINEX OBS files can be supplemented by fetching Swepos data, and missing mocoref data can be supplemented by PPP processing on the RINEX OBS file.

### Processing Directories
Processing directories have the internal structure required by the Radaz processing functions, even in those cases (if any) where _rd-tomo_ replaces them, for compatibility reasons. They are _generated_ by `rdtomo init`. **By default** they are generated inside `$HOME/Radar/Processing` but this can be altered with `rdtomo set PROCESSING_DIRS /your/path/here`.

### Tomogram Directories
A _Tomogram Directory_ is an output directory ending with `.tomo` generated by `rdtomo process` \[NOT IMPLEMENTED\] or `rdtomo forge`, which contains _all relevant files_ and serves as an output repository for _processed data_. It contains an _internal structure_ that must be maintained, and any files contained in there can be accessed normally for 3rd party software or file sharing et.c. It is thus a **unified** output format for storing processed data, making collaboration easier. **By default** they are generated inside `$HOME/Radar/Tomograms` but this can be altered with `rdtomo set TOMOGRAM_DIRS /your/path/here`. 

However, the **main advantage** of the `.tomo` directories is that they can be loaded directly by `rdtomo load` for programmatic access or `rdtomo view` \[NOT IMPLEMENTED\] for viewing the tomogram, plotting statistics or other analysis tools (**under implementation**).

```
yyyy-mm-dd-HH-MM-SS-XX-tag.tomo/
|-- flight_info.json
|-- moco_cut.csv
|-- phh
|    |-- processing_parameters.json
|    |-- raw_tomogram.tif
|    |-- multilooked_tomogram.tif
|    |-- filtered_tomogram.tif
|    |-- raw_statistics.csv
|    |-- multilooked_statistics.csv
|    |-- filtered_statistics.csv
|    |-- masked_statistics/
|    |       |-- <mask1>_raw_statistics.csv
|    |       |-- <mask1>_multilooked_statistics.csv
|    |       |-- <mask1>_filtered_statistics.csv
|    |       |-- <mask2>_raw_statistics.csv
|    |       |-- ...
|    |-- cached_masks/
|    |       |-- <mask1>.npy
|    |       |-- <mask1>.json
|    |       |-- <mask2>.npy
|    |       |-- ...
|    |-- .slices/
|    |       |-- dbr_[...]C.tif
|    |       |-- ...
|-- cvv
|    |-- ...
|-- lhh
|    |-- ...
|-- ...
```

### Tomogram Archives
As a _Tomogram Archive_ is counted any folder containing `.tomo` directories (non-recursively), with the additional demand that each `.tomo` directory must cover a unique scene (timestamp-spiral). Thus an archive can have child and parent archives. This is intended as an organizational help. A _Tomogram Archive_ can also be loaded into Python using e.g. `rdtomo view` \[NOT IMPLEMENTED\] or `rdtomo load`.

## Collaboration
If you want to modify the module or work on features to add, always **create your own branch**:
1. `git checkout -b feature/my-branch` (here _feature_ is a descriptive flag to signal a feature addition, but could be anything or nothing and _my-branch_ is a name for this particular branch)
2. Edit files, add content, et.c.
3. `git add .` inside the local repository
4. `git commit -m "Write a description here"`
5. `git push origin feature/my-branch`
6. You can then go to [GitHub](https://github.com/ansgarj/TomoSAR) and make a **pull request** for me to integrate it into the main branch

**NOTE**: If you're not a _collaborator_, please fork the repository first, then follow the same steps in your fork. After pushing your branch, open a pull request to this repository.

**NOTE**: to push changes you must set up your identity:
```sh
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```
Then set up authentication:
1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens.
2. Click **"Generate new token"** (classic or fine-grained).
3. Select scopes like `repo` and `workflow` (for private repos).
4. Copy the token and save it securely.
5. When Git asks for your password during `git push`, **paste the token instead**.

**NOTE**: you can also change to SSH authentication (more advanced)
1.  Generate an SSH key (if you don't have one):
```sh
ssh-keygen -t ed25519 -C "your.email@example.com"
```
2. Add your SSH key to the SSH agent (assuming you used default name):
```sh
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```
3. Add the **public** key to GitHub:
```sh
cat ~/.ssh/id_ed25519.pub
```
4. Copy the output and past into GitHub → Settings → SSH and GPG keys.
5. Change your remote URL to SSH:
```sh
git remote set-url origin git@github.com:ansgarj/TomoSAR.git
```

## License
This project is licensed under the BSD 3-Clause License – see the LICENSE file for details.