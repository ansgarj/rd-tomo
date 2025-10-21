from pathlib import Path
import json
from collections import defaultdict

# Project and package paths
PACKAGE_PATH = Path(__file__).resolve().parent
PROJECT_PATH = PACKAGE_PATH.parent
LOCAL = PROJECT_PATH / ".local"
SETTINGS_PATH = LOCAL / "settings.json"

# Frequency parameters
class Frequencies:
    __slots__ = ('BANDS', 'BANDWIDTHS', 'CENTRAL_FREQUENCIES', 'UNIT')

    def __init__(self):
        st = Settings()
        object.__setattr__(self, "BANDS", tuple(
            st.RADAR["POLARIZATIONS"].keys()
        ))
        object.__setattr__(self, "BANDWIDTHS", tuple(
            st.RADAR["BANDWIDTHS"][band] for band in self.BANDS
        ))
        object.__setattr__(self, "CENTRAL_FREQUENCIES", tuple(
            st.RADAR["CENTRAL_FREQUENCIES"][band] for band in self.BANDS
        ))
        object.__setattr__(self, "UNIT", "Hz")

    def __setattr__(self, key, value) -> None:
        raise AttributeError(f"{self.__class__.__name__} is immutable")

    def get(self, key: str, band: str = ""):
        if band:
            if band == "C":
                band = "C-band"
            if band == "L":
                band = "L-band"
            if band == "P":
                band = "P-band"
            value = getattr(self, key, None)
            for i, b in enumerate(self.BANDS):
                if b == band:
                    return value[i]
        else:
            return getattr(self, key, None)

    def zip(self, attributes: list[str]|tuple[str, ...] = []):
        if not (isinstance(attributes, list) or isinstance(attributes, tuple)):
            raise TypeError
        if not attributes:
            return zip(self.BANDS, self.BANDWIDTHS, self.CENTRAL_FREQUENCIES)
        values = []
        for attr in attributes:
            values.append(self.get(attr))
        return zip(*values)

# Beam parameters
class Beam:
    __slots__ = ("BAND_POLARIZATIONS", "BEAMWIDTHS", "DEPRESSION_ANGLES", "UNIT")

    def __init__(self):
        st = Settings()
        object.__setattr__(self, "BAND_POLARIZATIONS", tuple(
            (band, pol) for band, pol_list in st.RADAR["POLARIZATIONS"].items() for pol in pol_list
        ))
        object.__setattr__(self, "BEAMWIDTHS", tuple(
            st.RADAR["BEAMWIDTHS"][band][pol] for band, pol in self.BAND_POLARIZATIONS
        ))
        object.__setattr__(self, "DEPRESSION_ANGLES", tuple(
            st.RADAR["DEPRESSION_ANGLES"][band][pol] for band, pol in self.BAND_POLARIZATIONS
        ))
        object.__setattr__(self, "UNIT", "deg")

    def get(self, key: str, band: str = "", pol = "") -> float|None:
        if band and pol:
            if band in ["C", "c"]:
                band = "C-band"
            if band in ["L", "l"]:
                band = "L-band"
            if band in ["P", "p"]:
                band = "P-band"
            if pol in ["V", "v"]:
                pol = "V-pol"
            if pol in ["H", "h"]:
                pol = "H-pol"
            value = getattr(self, key, None)
            for i, (b, p) in enumerate(self.BAND_POLARIZATIONS):
                if b == band and p == pol:
                    return value[i] if value is not None else None         
        else:
            return getattr(self, key, None)
    
    def zip(self, attributes: list[str]|tuple[str, ...] = []):
        if not (isinstance(attributes, list) or isinstance(attributes, tuple)):
            raise TypeError
        if not attributes:
            return zip(self.BAND_POLARIZATIONS, self.BEAMWIDTHS, self.DEPRESSION_ANGLES)
        values = []
        for attr in attributes:
            values.append(self.get(attr))
        return zip(*values)

# Settings
class Settings:
    def __init__(self):
        if SETTINGS_PATH.exists() and SETTINGS_PATH.is_file():
            with open(SETTINGS_PATH, "r") as file:
                self.data = json.load(file)
        else:
            save_default()
            self.data = DEFAULT

    @property
    def VERBOSE(self):
        return self.data["VERBOSE"]
    
    @VERBOSE.setter
    def VERBOSE(self, value) -> None:
        if not isinstance(value, bool):
            raise ValueError("The VERBOSE setting takes only boolean values")
        self.data["VERBOSE"] = value

    @property
    def MOCOREF_LONGITUDE(self):
        return self.data["MOCOREF_LONGITUDE"]
    
    @MOCOREF_LONGITUDE.setter
    def MOCOREF_LONGITUDE(self, value) -> None:
        if not isinstance(value, str):
            raise ValueError("The MOCOREF_LONGITUDE setting takes a string as value")
        self.data["MOCOREF_LONGITUDE"] = value

    @property
    def MOCOREF_LATITUDE(self):
        return self.data["MOCOREF_LATITUDE"]
    
    @MOCOREF_LATITUDE.setter
    def MOCOREF_LATITUDE(self, value) -> None:
        if not isinstance(value, str):
            raise ValueError("The MOCOREF_LATITUDE setting takes a string as value")
        self.data["MOCOREF_LATITUDE"] = value

    @property
    def MOCOREF_HEIGHT(self):
        return self.data["MOCOREF_HEIGHT"]
    
    @MOCOREF_HEIGHT.setter
    def MOCOREF_HEIGHT(self, value) -> None:
        if not isinstance(value, str):
            raise ValueError("The MOCOREF_HEIGHT setting takes a string as value")
        self.data["MOCOREF_HEIGHT"] = value

    @property
    def MOCOREF_ANTENNA(self):
        return self.data["MOCOREF_ANTENNA"]
    
    @MOCOREF_ANTENNA.setter
    def MOCOREF_ANTENNA(self,value) -> None:
        if not isinstance(value, str):
            raise ValueError("The MOCOREF_ANTENNA setting takes a string as value")
        self.data["MOCOREF_ANTENNA"] = value
    
    @property
    def RTKP_CONFIG(self) -> Path:
        return Path(self.data["RTKP_CONFIG"])
    
    @RTKP_CONFIG.setter
    def RTKP_CONFIG(self, value):
        if not isinstance(value, str):
            raise ValueError("The RTKP_CONFIG settings take a path-like string as value")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"The file {value} does not exist")
        self.data["RTKP_CONFIG"] = str(path.resolve())

    @property
    def DATA_DIRS(self) -> Path:
        dirs = Path(self.data["DATA_DIRS"])
        dirs.mkdir(parents=True, exist_ok=True)
        return dirs
    
    @DATA_DIRS.setter
    def DATA_DIRS(self, value) -> None:
        if not isinstance(value, str):
            raise ValueError("The DATA_DIRS settings take a path-like string as value")
        path = Path(value)
        if path.is_file():
            raise FileExistsError(f"The path {value} points to a file")
        path.mkdir(parents=True, exist_ok=True)
        self.data["DATA_DIRS"] = str(path.resolve())
        
    
    @property
    def PROCESSING_DIRS(self) -> Path:
        dirs = self.data["PROCESSING_DIRS"]
        dirs.mkdir(parents=True, exist_ok=True)
        return dirs
    
    @PROCESSING_DIRS.setter
    def PROCESSING_DIRS(self, value) -> None:
        if not isinstance(value, str):
            raise ValueError("The PROCESSING_DIRS settings take a path-like string as value")
        path = Path(value)
        if path.is_file():
            raise FileExistsError(f"The path {value} points to a file")
        path.mkdir(parents=True, exist_ok=True)
        self.data["PROCESSING_DIRS"] = str(path.resolve())
    
    @property
    def TOMO_DIRS(self) -> Path:
        dirs = self.data["TOMO_DIRS"]
        dirs.mkdir(parents=True, exist_ok=True)
        return dirs
    
    @TOMO_DIRS.setter
    def TOMO_DIRS(self, value) -> None:
        if not isinstance(value, str):
            raise ValueError("The TOMO_DIRS settings take a path-like string as value")
        path = Path(value)
        if path.is_file():
            raise FileExistsError(f"The path {value} points to a file")
        path.mkdir(parents=True, exist_ok=True)
        self.data["TOMO_DIRS"] = str(path.resolve())
    
    @property
    def SWEPOS_LOGIN(self):
        return self.data["SWEPOS_LOGIN"]
    
    @property
    def SWEPOS_USERNAME(self):
        return self.SWEPOS_LOGIN["USERNAME"]
    
    @SWEPOS_USERNAME.setter
    def SWEPOS_USERNAME(self, value) -> None:
        if not isinstance(value, str):
            raise ValueError("The SWEPOS_USERNAME setting takes a string as value")
        self.data["SWEPOS_LOGIN"]["USERNAME"] = value

    @property
    def SWEPOS_PASSWORD(self):
        return self.SWEPOS_LOGIN["PASSWORD"]
    
    @SWEPOS_PASSWORD.setter
    def SWEPOS_PASSWORD(self, value) -> None:
        if not isinstance(value, str):
            raise ValueError("The SWEPOS_PASSWORD setting takes a string as value")
        self.data["SWEPOS_LOGIN"]["PASSWORD"] = value

    @property
    def FILES(self):
        return self.data["FILES"]

    def add(self, key: str, files: str|Path|list[str|Path], **kwargs) -> None:
        valid_keys = ["DEM", "DEMS", "CANOPY", "CANOPIES", "MASK", "MASKS", "RECEIVER"]
        if key not in valid_keys:
            raise KeyError(f"Invalid key {key}. Valid keys: {valid_keys}")
        if not isinstance(files, list):
            files = [files]
        match key:
            case "DEM":
                key = "DEMS"
            case "CANOPY":
                key = "CANOPIES"
            case "MASK":
                key = "MASKS"
            case "RECEIVER":
                if len(files) > 1:
                    raise ValueError("Only one RECEIVER can be added at a time")
                if not isinstance(files[0], str|Path):
                    raise ValueError("A RECEIVER setting must be a path-like string")
                elif not Path(files[0]).is_file():
                    raise FileNotFoundError(f"The file {files} does not exist")
                else:
                    file = files[0]
                antenna = kwargs.get("antenna", None)
                if antenna == "SATELLITES":
                    raise KeyError("To set the SATELLITES antenna file use the setter")
                if antenna is None:
                    raise KeyError("To add a RECEIVER file, antenna must be specified (antenna=)")
                radome = kwargs.get("radome", "NONE")
                if antenna in self.ANTENNAS:
                    self.ANTENNAS[antenna][radome] = str(file)
                else:
                    self.ANTENNAS[antenna] = {
                        radome: str(file)
                    }
                return

        self.FILES[key].extend(files)

    def remove(self, key, files: str|Path|list[str|Path], **kwargs) -> None:
        valid_keys = ["DEM", "DEMS", "CANOPY", "CANOPIES", "MASK", "MASKS", "RECEIVER"]
        if key not in valid_keys:
            raise KeyError(f"Invalid key {key}. Valid keys: {valid_keys}")
        if not isinstance(files, list):
            files = [files]
        match key:
            case "DEM":
                key = "DEMS"
            case "CANOPY":
                key = "CANOPIES"
            case "MASK":
                key = "MASKS"
            case "RECEIVER":
                if len(files) > 1:
                    raise ValueError("Only one RECEIVER can be rmoved at a time")
                else:
                    file = files[0]
                antenna = kwargs.get("antenna", None)
                if antenna == "SATELLITES":
                    raise KeyError("The SATELLITES file cannot be removed.")
                if antenna is None:
                    raise KeyError("To remove a RECEIVER file, antenna must be specified (antenna=)")
                radome = kwargs.get("radome", "NONE")
                if antenna in self.ANTENNAS:
                    self.ANTENNAS[antenna].pop(radome, None)
                    if not self.ANTEENNAS[antenna]:
                        self.ANTENNAS.pop(antenna, None)
                return
        
        old_files = self.get(key)
        self.set(key, [file for file in old_files if file not in files])

    @property
    def DEMS(self):
        return self.FILES["DEMS"]
       
    @property
    def CANOPIES(self):
        return self.FILES["CANOPIES"]
    
    @property
    def MASKS(self):
        return self.FILES["MASKS"]
    
    @property
    def ANTENNAS(self):
        return self.FILES["ANTENNAS"]
    
    @property
    def SATELLITES(self):
        return self.ANTENNAS.get("SATELLITES", None)
    
    @SATELLITES.setter
    def SATELLITES(self, value) -> None:
        if not isinstance(value, str):
            raise ValueError("The FILES : ANTENNAS : SATELLITES settings take a path-like string as value")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"The file {value} does not exist")
        self.ANTENNAS["SATELLITES"] = str(path.resolve())

    @property
    def RECEIVERS(self):
        return {key: value for key, value in self.ANTENNAS.items() if key != "SATELLITES"}
    
    @RECEIVERS.setter
    def RECEIVERS(self, value: dict) -> None:
        if not isinstance(value, (dict, defaultdict)):
            raise ValueError("RECEIVERS can only be set to a dict with keys that are the ANTENNA TYPE")
        for v in value.values():
            if not isinstance(v, (dict, defaultdict)):
                raise ValueError("Each value in the RECEIVERS dict must be a dict with keys that are the radome.")
        antennas = [self.SATELLITES]
        antennas.extend(value)
        self.data["FILES"]["ANTENNAS"] = antennas

    def RECEIVER(self, receiver_id: str, radome: str = None):
        if radome:
            return self.RECEIVERS.get(receiver_id,{}).get(radome, None)
        else:
            return self.RECEIVERS.get(receiver_id, {})
        
    @property
    def RADAR(self):
        return self.data["RADAR"]
    
    def get(self, key: str):       
        return self.getattr(key, None)
    
    def set(self, key: str, value):
        setattr(self, key, value)

    def print(self) -> None:
        print(json.dumps(self.data, indent=4))

    def save(self) -> None:
        LOCAL.mkdir(exist_ok=True)
        with open(SETTINGS_PATH, "w") as file:
            json.dump(self.data, file, indent=4)

    def reset(self) -> None:
        self.data = DEFAULT

DEFAULT = {
    "VERBOSE": False,
    "MOCOREF_LONGITUDE": "Longitude",
    "MOCOREF_LATITUDE": "Latitude",
    "MOCOREF_HEIGHT": "Ellipsoidal height",
    "MOCOREF_ANTENNA": "Antenna height",
    "RTKP_CONFIG": None,
    "DATA_DIRS": str(Path.home() / "Radar" / "Data"),
    "PROCESSING_DIRS": str(Path.home() / "Radar" / "Processing"),
    "TOMO_DIRS": str(Path.home() / "Radar" / "Tomograms"),
    "SWEPOS_LOGIN": {
        "USERNAME": None,
        "PASSWORD": None
    },
    "FILES": {
        "ANTENNAS": {
            "SATELLITES": None,
        },
        "DEMS": [],
        "CANOPIES": [],
        "MASKS": []
    },
    "RADAR": {
        "POLARIZATIONS": {
            "P-band": ["H-pol"],
            "L-band": ["H-pol", "V-pol"],
            "C-band": ["V-pol"]
        },
        "CENTRAL_FREQUENCIES": {
            "P-band": 412.5e6,
            "L-band": 1.25e9,
            "C-band": 5.3125e9,
        },
        "BANDWIDTHS": {
            "P-band": 25e6,
            "L-band": 50e6,
            "C-band": 125e6
        },
        "BEAMWIDTHS": {
            "P-band": {
                "H-pol": 75.2,
                "V-pol": None,
            },
            "L-band": {
                "H-pol": 78.7,
                "V-pol": 58.8,
            },
            "C-band": {
                "H-pol": None,
                "V-pol": 51.3
            } 
        },
        "DEPRESSION_ANGLES": {
            "P-band": {
                "H-pol": 45.,
                "V-pol": None,
            },
            "L-band": {
                "H-pol": 45.,
                "V-pol": 45.,
            },
            "C-band": {
                "H-pol": None,
                "V-pol": 45.
            } 
        },
    }
}

def save_default() -> None:
    LOCAL.mkdir(exist_ok=True)
    with open(SETTINGS_PATH, "w") as file:
        json.dump(DEFAULT, file, indent=4)