# README
This repository contains the `tomosar` python module which is **in alpha development**, which also provides a selection of [CLI Tools](#core-cli-tools) that can be run directly from the terminal. It can be installed into your  `Python` environment via `pip`.  You can either **clone** the repository and install it as a development module (`-e`), this is the route suggested below, or you can install it directly from the repository:
```sh
pip install git+https://github.com/ansgarj/TomoSAR.git
```

If you clone the repository you will have local access to the code for experimentation and your own development (you can later create your own _branch_ and `git push` that branch and make _pull requests_ to merge your _branch_ to the `main` branch, see [Collaboration](#collaboration)), and you can `git pull` any updates and see the changes immediately reflected in your environment.

If you choose to install directly from the online repository. You have access to it as is, but to update it you have to run:
```sh
pip install --force-reinstall --no-cache-dir git+https://github.com/yourusername/TomoSAR.git
```
You also will not be able to modify or develop the code yourself.

Included below is some basic GitHub usage, but there is also plenty of documentation online (e.g. pulling specific versions or branches).

## Installation
**NOTE**: It is recommended to use a Python virtual environment when installing the module. This allows you to have separate Python environments for different projects, avoiding any potential conflicts, and is also _required_ by some Linux distributions in order to avoid potential conflicts with _system Python_. It also makes it _easier to reproduce your setup_ on another computer should you wish to. Below is described an example setup _with a virtual environment_.

You can place your virtual environment anywhere, but I have placed it in the associated project folder:
```sh
git clone https://github.com/ansgarj/TomoSAR.git
cd TomoSAR
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
tomosar setup
```

**NOTE**: running `tomosar setup` is not strictly required, but will install two basic Git _hooks_ for you, check if all required binaries are present and help you if not, and pre-warm the \_\_pycache\_\_.

In the above example the Python virtual environment is created inside a `.venv` folder inside the `TomoSar` project directory. I find this helpful to contain the project in one directory. **NOTE**: do _not_ use another name for the virtual environment if placed inside the project directory, but _if you do_ then you must add this folder to the `.gitignore` file, e.g.:
```sh
...

# Python virtual environment
.venv/
my-venv/ 

# Byte-compiled / cache files
...
```
Also add any other files or subdirectories should not be pushed to GitHub that you keep inside the project directory that I have not added (if any).

**NOTE**:I have a simple shell function that I include into my `.bashrc` file to activate virtual environments by alias e.g.:
```sh
# Activate virtual environments by alias
activate() {
    local alias="$1"
    local project_path=""

    case "$alias" in 
        sar)
            project_path="$HOME/TomoSAR/"
            ;;
        *)
            echo "Unknown project alias: $alias"
            return 1
            ;;
    esac
    source "$project_path/.venv/bin/activate"
}
```
That way I can activate the virtual environment by running e.g. `activate sar` from any folder (say, where I have the radar data). 

### Required Binaries
The `tomosar` module relies on some 3rd party software for GNSS processing. It is recommended to make sure you have everything set up before starting to use the toolbox, even though it is not _strictly_ necessary. If you run `tomosar setup` as suggested above, this is already done. Otherwise you can run `tomosar dependencies` to perform the check independently. This will check if there are any binaries in your `PATH` with the correct names, but not actually test if it is the correct binaries, and provide helpful information if not (including source html links where applicable). The required binaries are:
1. `convbin` from `rtklib`
2. `rnx2rtkp` from `rtklib`
3. `crx2rnx` from GSI Japan
4. `gfzrnx` from GFZ Potsdam
5. `glab` from ESA
6. `unimoco` from Radaz

**NOTE**: some of the binaries will have different names, e.g. `CRX2RNX` or `gLAB_linux` when downloaded, but the name provided above and by `tomotest binaries` are the names the `tomosar` module uses and the binaries should either be _renamed_ and moved into the `PATH` _or_ you can create a _symlink_ with the correct name in the `PATH`. 

**NOTE**: _TomoSAR_ uses the `tomosar.resource` (from `.binaries`) context manager to provide local copies of various files in the working directory (not radar, IMU or GNSS data which are expected to be inside the working directory tree). The basic reason for this is to allow the use of containers to run 3rd party binaries (such as docker). 

**NOTE**: I use the [demo5](https://github.com/rinex20/RTKLIB-demo5) version of `rtklib`, and this is the one `tomosar setup`  and `tomosar dependencies` will suggest installing if it finds no `convbin` or `rnx2rtkp` binary, **but** it _should_ run on standard `rtklib` as well.

## Usage
The `tomosar` _module_ is a work-in-progress to provide a one-stop toolbox for our tomographic SAR needs. Once installed it can be imported into Python by running `import tomosar`, or you can select submodules or objects as usual  in Python. Currently, the only available documentation is the one present in the code, _but I plan to add separate documentation later._

The CLI tools are intended to provide a toolbox for the most common or predicted needs, the idea being that unless you are working on your own project with something not integrated into the CLI tools, you can use the module directly from the terminal by running a command without having to enter into Python and importing the module.  All tools are accessed as subcommands of `tomosar` and can be called with `--help` for some basic syntax: `tomosar --help` provides syntax help and `tomosar manual` prints a general help overview.

**NOTE**: many of these tools are not yet fully implemented.

### Settings
The local _TomoSAR_ settings are stored inside the `.local` folder inside the project directory in a `settings.json` file (this is generated by `tomosar setup` but if it does not exist any internal function that reads settings generates it at that point). The current settings can be viewed with `tomosar settings` which prints to stdout. If the `RTKP_CONFIG` setting is `null` _TomoSAR_ will use internal configuration files, and similarly if `FILES: ANTENNAS: SATELLITES` is `null`.

`FILES: ANTENNAS` can otherwise be used to point to antenna files containing absolute reference data for the receiver antenna. I am currently working on adding one internally for the mobile base station (CHCI83) that we have been using, but note that the use of this is not implemented. 

Note that you can set login info for the Swepos network (so you don't have to specify manually) by `tomosar set swepos-username` and `tomosar set swepos-password`, but that the password is stored inside `settings.json` in an **unencrypted state**. Use therefore with caution. 

Finally you can use `tomosar add` to add files or folders to `FILES: DEMS`, `FILES: CANOPIES` and `FILES: MASKS`. These lists are used by _TomoSAR_ to find GeoTIFF files for DEM and canopy DSM references, and shape files for masking tomograms. The GeoTIFF files are used for slicing (`tomoprocess slice` \[**NOTIMPLEMENTED**\]) with either the ground or the canopy as reference respectively. The shapefiles are used to generate masks by `tomoprocess forge`, and can be updated for a [Tomogram Directory](#tomogram-directories) or multiple [Tomogram Directories](#tomogram-directories) by running `tomosar load --update` and then inside the Python terminal:
```python
tomos.save()
```

### Data Directories
As a _data directory_ functions anything containing at least the drone data. There are no specific requirements on the _internal_ structure, but each data directory should contain **only one set of data** (one recording from the drone and matching GNSS data if available). Data directories are _generated_ by `tomoprocess data` \[**NOT IMPLEMENTED**\] if used to fetch drone data. **By default** they are generated inside `$HOME/Radar/Data` but this can be altered with `tomosar set DATA_DIRS /your/path/here`.

### Processing Directories
Processing directories have the internal structure required by the Radaz processing functions, even in those cases (if any) where TomoSAR replaces them, for compatibility reasons. They are _generated_ by `tomoprocess init` \[**NOT IMPLEMENTED**\]. **By default** they are generated inside `$HOME/Radar/Processing` but this can be altered with `tomosar set PROCESSING_DIRS /your/path/here`.

### Tomogram Directories
A _Tomogram Directory_ is an output directory ending with `.tomo` generated by `tomoprocess forge` or `tomoprocess tomo`, which contains _all relevant files_ and serves as an output repository for _processed data_. It contains an _internal structure_ that must be maintained, and any files contained in there can be accessed normally for 3rd party software or file sharing et.c. It is thus a **unified** output format for storing processed data, making collaboration easier. **By default** they are generated inside `$HOME/Radar/Tomograms` but this can be altered with `tomosar set TOMOGRAM_DIRS /your/path/here`. 

However, the **main advantage** of the `.tomo` directories is that they can be loaded directly by `tomoview` for viewing the tomogram, plotting statistics or other analysis tools (**under implementation**).

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

## Known issues
The PPP processing is currently not interacting correctly with gLAB and can be considered buggy. It does not deliver reliable results.

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