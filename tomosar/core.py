# Imports
from __future__ import annotations
import os
import re
import socket
import shutil
from pathlib import Path
from datetime import datetime, date, time
from typing import Dict, ClassVar
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass, field, fields
import rasterio
from rasterio.profiles import Profile
from rasterio.transform import Affine
from rasterio.features import rasterize
import json
from collections import defaultdict
import copy
from typing import KeysView, ValuesView, ItemsView

from .utils import warn, collect_statistics, estimaterr, apply_variable_descriptions, parse_datetime_string
from .tomogram_processing import multilook, filter
from .apperture import SARModel
from .config import Settings

### Custom classes
@dataclass
class ImageInfo:
    filename: str 
    folder: str = "."
    date: datetime | None = None
    spiral: int | None = None
    band: str | None = None
    width: float | None = None
    res: float | None = None
    linuxTime: int | None = None # timestamp for when the image was processed
    smo: float | None = None
    hoff: float = 0.0
    depth: float = 0.0
    roff: float | None = None
    ham: float = 0.0
    refr: float = 1.0
    lat: float | None = None
    lon: float | None = None
    DC: float = 0.0
    DL: float = 999.0
    HC: float = 120.0
    HV: float = 999.0
    thresh: float | None = 10.0
    squint: float | None = 0.0
    text: str | None = None
    image: np.ndarray | None = None
    profile: Profile | None = None
    _paths: list[Path] = field(default_factory=list, repr=False)
    

    PAIR_PARAMETERS: ClassVar[list[str]] = ['date', 'width', 'res', 'smo', 'ham', 'hoff',
                  'depth', 'refr', 'lat', 'lon', 'DC', 'DL', 'HC', 'HV', 'thresh', 'squint']
    
    @property
    def is_pair(self):
        return self.band in ['phh','cvv']
    
    @property
    def category(self):
        category = []
        if self.hoff == 0 and abs(self.depth) >= 0:
            category.append('sub')
        if self.hoff >= 0 and self.depth == 0 and self.thresh == 10:
            category.append('sup')
        if self.hoff > 0 and abs(self.depth) >= 0 and self.thresh == 10:
            category.append('canopy')
        return category
    
    @property
    def height(self):
        return self.hoff - abs(self.depth)
    
    @property
    def path(self):
        if self.folder and self.filename:
            if self._paths:
                warn("This image seems to have been generated from multiple files, but a " \
                "filename and folder has been set: .path no longer returns the paths to the" \
                "image files used to generate it.")
            return Path(self.folder) / self.filename
        elif self._paths:
            return self._paths
        else:
            return None
        
    def pair(self, other: 'ImageInfo') -> 'ImageInfo':
        if not isinstance(other, ImageInfo):
            raise TypeError
        composed = self.copy()
        composed.filename = ""
        composed.folder = ""
        composed._paths = [self.path, other.path]
        if self.band == 'phh1' and other.band == 'phh0':
            composed.band = 'phh'
        if self.band == 'cvv1' and other.band == 'cvv0':
            composed.band = 'cvv'
        else:
            raise ValueError("Only phh1 and phh0 or cvv1 and cvv0 bands can be paired.")

        key1 = tuple(self.get(k) for k in ImageInfo.PAIR_PARAMETERS)
        key2 = tuple(other.get(k) for k in ImageInfo.PAIR_PARAMETERS)
        if key1 != key2:
            raise ValueError("Image parameters do not match.")

        if self.profile != other.profile:
            raise ValueError("Image profiles do not match.")
        if self.image and other.image:
            r = np.sqrt(np.abs(self.image)**2 + np.abs(other.image)**2)
            phase = np.angle(self.image) - np.angle(other.image)
            composed.image = r * np.exp(phase * 1j)
        else:
            composed.image = None

        composed.linuxTime = max(self.linuxTime, other.linuxTime)
        return composed
    
    def copy(self) -> 'ImageInfo':
        new_info = ImageInfo(self.folder, self.filename)
        new_info.date = self.date
        new_info.spiral = self.spiral
        new_info.band = self.band
        new_info.width = self.width
        new_info.res = self.res
        new_info.linuxTime = self.linuxTime
        new_info.smo = self.smo
        new_info.hoff = self.hoff
        new_info.depth = self.depth
        new_info.roff = self.roff
        new_info.ham = self.ham
        new_info.refr = self.refr
        new_info.lat = self.lat
        new_info.lon = self.lon
        new_info.DC = self.DC
        new_info.DL = self.DL
        new_info.HC = self.HC
        new_info.HV = self.HV
        new_info.thresh = self.thresh
        new_info.squint = self.squint
        new_info.text = self.text
        if isinstance(self.image, np.ndarray):
            new_info.image = self.image.copy()
        else:
            new_info.image = None
        new_info.profile = self.profile.copy()
        return new_info

    def get(self, key: str):
        return getattr(self, key, None)

    def read(self, db0: float = 1):
        import rasterio
        
        db0 = db0 * self.res**2
        if isinstance(self.path, Path):
            try:
                with rasterio.open(self.path) as src:
                    if src.profile.count == 2:
                        real = src.read(1)
                        imag = src.read(2)
                        self.image = (real + 1j * imag) / db0
                        self.profile = src.profile
                    else:
                        self.image = src.read()
                        self.profile = src.profile
            except Exception as e:
                print(f"Error reading {self.filename}: {e}")
        elif len(self.path) == 2:
            try:
                with rasterio.open(self.path[0]) as src:
                    real = src.read(1)
                    imag = src.read(2)
                    image1 = (real + 1j * imag) / db0
                    self.profile = src.profile
            except Exception as e:
                print(f"Error reading {self.filename}: {e}")
            try:
                with rasterio.open(self.path[1]) as src:
                    real = src.read(1)
                    imag = src.read(2)
                    image2 = (real + 1j * imag) / db0
            except Exception as e:
                print(f"Error reading {self.filename}: {e}")
            self.image = (image1, image2)
        
        return self

    def generate_filename(self) -> str:
        name = f"dbr_{self.date.strftime("%Y-%m-%d-%H-%M-%S")}-{self.spiral:02d}_{self.band}_{self.width}m_{self.res}m_{self.linuxTime}"
        if self.text:
            name += f"_{self.text}"
        if self.hoff != 0:
            name += f"_ho{self.hoff}m"
        if self.ham != 0:
            name += f"_ham{self.ham:.2f}"
        name += f"_ro{self.roff}m"
        if self.smo != self.width/10./self.res:
            name += f"_smo{self.smo}"
        if self.refr != 1:
            name += f"_ref{self.refr}"
        if self.depth != 0:
            name += f"_prof{self.depth:+05.1f}m"
        name +- f"_LA={self.lat:.16f}"
        name +- f"_LO={self.lon:.16f}"
        if self.DC != 0:
            name += f"_DC{self.DC:03d}"
        if self.DL != 999:
            name += f"_DL{self.DL:03d}"
        if self.HC != 120:
            name += f"_HC{self.HC:03d}"
        if self.HV != 999:
            name += f"_HV{self.HV:03d}"
        if self.thresh != 10:
            name += f"_TH{self.thresh:.2f}"
        if self.squint != 0:
            name += f"_sq{self.squint}"
        name += f"_C.tif"
        return name

    def save(self, folder: str | Path):
        if self.image is None or self.profile is None:
            return
        
        # Set filename
        if self.filename:
           filename = self.filename
        else:
            filename = self.generate_filename() 

        # Determine output folder
        if folder:
            out_path = Path(folder)
        elif self.folder:
            out_path = Path(self.folder)
        else:
            out_path = Path.cwd()

        # Get path
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / filename

        # Prepare profile for writing complex image as two bands
        profile = self.profile.copy()
        if np.iscomplexobj(self.image):
            profile.update(count=2)
        else:
            profile.update(count=1)

        # Write real and imaginary parts as separate bands
        for path in out_path:
            with rasterio.open(path, 'w', **profile) as dst:
                if np.iscomplexobj(self.image):
                    dst.write(self.image.real, 1)
                    dst.write(self.image.imag, 2)
                else:
                    dst.write(self.image)

    def __eq__(self, other):
        if not isinstance(other, ImageInfo):
            return False
        for f in fields(self):
            val1 = self.get(f.name)
            val2 = other.get(f.name)
            if not val1 or not val2:
                continue  # skip comparison if either is falsy
            # Special comparison for date
            if f.name == 'date':
                if val1.year > 2000 and val2.year > 2000:
                    if val1.date() != val2.date():
                        return False
                if not (val1.time() == time(0,0,0,0) or val2.time() == time(0,0,0,0)):
                    if val1.time() != val2.time():
                        return False
            # Special comparison for band
            elif f.name == 'band':
                if not isinstance(val1, str):
                    if val2 not in val1:
                        return False
                elif not isinstance(val2, str):
                    if val1 not in val2:
                        return False
                else:
                    if val1 != val2:
                        return False    
            # Special for Linux time
            if f.name == 'linuxTime':
                if val1 != val2:
                    warn(f"linuxTime attributes {val1} and {val2} do not match, but the objects may still match.")
            # Compare other attribute
            elif val1 != val2:
                return False
        return True

    def __bool__(self):
        for f in fields(self):
            value = self.get(f.name)
            if f.name == 'date' and value.year < 2000:
                continue # Skip dates before year 2k
            if value:
                return True
        return False

@dataclass
class SliceInfo:
    slices: list[ImageInfo] = field(default_factory=list)

    # Class level constant that defines a tomogram in terms of image parameters
    TOMOGRAM_PARAMETERS: ClassVar[list[str]] = ['date', 'spiral', 'band', 'width', 'res', 'smo', 'ham',
                    'lat', 'lon', 'DC', 'DL', 'HC', 'HV', 'squint']


    def append(self, item: ImageInfo) -> 'SliceInfo':
        self.slices.append(item)
        return self
    
    def read(self, db0, npar: int = os.cpu_count()) -> 'SliceInfo':
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        desc = f"Reading: "
        read_slices = SliceInfo()

        with ThreadPoolExecutor(max_workers=npar) as executor:
            futures = [executor.submit(slice.read, db0) for slice in self.slices if not slice.image]

            for future in tqdm(as_completed(futures), total=len(self), desc=desc, unit='files', leave=False):
                try:
                    read_slices.append(future.result())
                except Exception as e:
                    print(f"Error reading file: {e}")
        return read_slices
    
    def extend(self, other: 'SliceInfo') -> 'SliceInfo':
        if not isinstance(other, SliceInfo):
            raise TypeError("Can only extend with another SliceInfo instance.")
        self.slices.extend(other.slices)
        return self
    
    def copy(self) -> 'SliceInfo':
        new_slice_info = SliceInfo()
        new_slice_info.slices = [slice for slice in self.slices]
        return new_slice_info

    def unique(self) -> 'SliceInfo':
        unique = SliceInfo()
        seen = set()

        for slice in self.slices:
            key = (slice.date, slice.spiral, slice.band, slice.width, slice.res, slice.smo,
                   slice.hoff, slice.depth, slice.refr, slice.lat, slice.lon,
                   slice.DC, slice.DL, slice.HC, slice.HV, slice.thresh, slice.squint)

            if key not in seen:
                seen.add(key)
                unique.append(slice)
        
        return unique

    def group(self, key: str | list[str], list: bool = False) -> Dict[str, 'SliceInfo'] | list['SliceInfo']:
        from collections import defaultdict
        if isinstance(key, str):
            key = [key]
        
        grouped = defaultdict(SliceInfo)
        for slice in self.slices:
            if len(key) == 1:
                group_key = slice.get(key[0])
            else:
                group_key = tuple(getattr(slice, k) for k in key)
            grouped[group_key].append(slice)
        
        if list:
            grouped = [g for g in grouped.values()]
        return grouped
    
    def pair(self, retain: bool = False):
        def make_key(slice):
            return tuple(slice.get(k) for k in ImageInfo.PAIR_PARAMETERS)
        
        band_groups = self.group('band')
        paired = SliceInfo()

        # Pairing logic for each interferometric band
        for band1, band0 in [('phh1', 'phh0'), ('cvv1', 'cvv0')]:
            group_map = defaultdict(lambda: {'1': [], '0': []})

            for s in band_groups.get(band1, SliceInfo()):
                group_map[make_key(s)]['1'].append(s)
            for s in band_groups.get(band0, SliceInfo()):
                group_map[make_key(s)]['0'].append(s)

            for group in group_map.values():
                for s1, s0 in zip(group['1'], group['0']):
                    if s1 and s0:
                        composite_slice = s1.pair_with(s0)
                        paired.append(composite_slice)

        # Include other bands
        for band, slices in band_groups.items():
            if band not in ['phh1', 'phh0', 'cvv1', 'cvv0'] or retain:
                paired.extend(slices)

        return paired

    def tomograms(self) -> list['SliceInfo']:
        tomo_slices = self.group(self.TOMOGRAM_PARAMETERS,list=True)
        tomo_slices = [s for s in tomo_slices if len(s) > 1]
        return tomo_slices
    
    def categorize(self) -> Dict[str, 'SliceInfo']:
        categories = defaultdict(SliceInfo)
        for slice in self:    
            for cat in slice.category:
                    categories[cat].append(slice)
        
        return categories

    def sort(self, key: str) -> 'SliceInfo':
        values = self.get(key)
        if isinstance(values, np.ndarray) and values.ndim == 1:
            idx = np.argsort(values)
            sorted_slices = [self.slices[i] for i in idx]
            self.slices = sorted_slices
            return idx
        else:
            raise Exception(f"Key {key} cannot be sorted.")

    def get(self, key: str):
        values = [slice.get(key) for slice in self.slices]
        if all(isinstance(x,(int,float,np.number,np.ndarray)) for x in values):
            values = np.array(values)
        return values if len(values) > 1 else values[0] if values else None
    
    def save(self, folder: str = ""):
        for slice in self.slices:
            slice.save(folder=folder)

    def filter(self, filter: ImageInfo):
        filtered_slices = [s.copy() for s in self.slices if s == filter]
        print(f"{len(self) - len(filtered_slices)} slices filtered.")
        self.slices = filtered_slices

    @classmethod
    def scan(self, path: str|Path = '.', filter: ImageInfo = None, 
             read: bool = False, npar: int = os.cpu_count) -> 'SliceInfo':
        return sliceinfo(path, filter=filter, read=read, npar=npar)

    def __getitem__(self, index):
        if isinstance(index,(list,np.ndarray)) and np.asarray(index).dtype == bool:
            filtered = [s for s, keep in zip(self.slices, index) if keep]
            return SliceInfo(filtered)
        else:
            return self.slices[index]
        
    def __len__(self):
        return len(self.slices)
        
    def __iter__(self):
        return iter(self.slices)
    
    def __bool__(self):
        return bool(self.slices)

    def __repr__(self):
        return f"SliceInfo({len(self.slices)} slices)"
    
    def __eq__(self, other):
        if not isinstance(other, SliceInfo):
            return False
        return all(slice in other.slices for slice in self.slices) and len(self.slices) == len(other.slices)

@dataclass
class Mask:
    name: str = ""
    id: int = 0
    mask: np.ndarray = field(default=None, repr=False)
    multilooked: np.ndarray = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict)

    def copy(self) -> 'Mask':
        new_mask = Mask(name=self.name)
        new_mask.mask = self.mask.copy() if self.mask is not None else None
        return new_mask
    
    def apply(self, tomogram: np.ndarray, multilooked: bool = False) -> np.ndarray:
        masked_tomogram = tomogram.copy()
        if multilooked:
            mask = np.broadcast_to(self.multilooked[np.newaxis, :, :], tomogram.shape)
        else:
            mask = np.broadcast_to(self.mask[np.newaxis, :, :], tomogram.shape)
        if np.iscomplexobj(tomogram):
            masked_tomogram[~mask] = np.nan + np.nan*1j
        else:
            masked_tomogram[~mask] = np.nan

        return masked_tomogram

class Masks:
    
    def __init__(self, parent: TomoInfo = None, masks: dict[str,list[Mask]] = defaultdict[list]):
        self.parent: TomoInfo = parent
        self.masks: dict[str,list[Mask]] = masks
            
    def keys(self) -> KeysView[str]:
        return self.masks.keys()
    
    def values(self) -> ValuesView[list[Mask]]:
        return self.masks.values()
    
    def items(self) -> ItemsView[str, list[Mask]]:
        return self.masks.items()
    
    def update(self, mask_dir: str = ""):
        if self.parent.tomograms.profile is None:
            raise ValueError("Tomograms profile must be set before updating masks.")
        self.masks = get_masks(raster_profile=self.parent.tomograms.profile,
                                multilooked_profile=self.parent.multilook.profile, mask_dir=mask_dir)
        self.parent.stats.collect('masked')

    def add_masks(self, key: str, masks: Mask | list[Mask]):
        if isinstance(masks, Mask):
            self.masks[key].append(masks)
        else:
            self.masks[key].extend(masks)
    
    def add_key(self, key: str, masks: list[Mask]|Mask):
        if key in masks:
            raise RuntimeError(f"Masks with key {key} already exists, use .add_masks(key, masks) instead.")
        if isinstance(masks, Mask):
            masks = [masks]
        self.masks[key] = masks

    def read(self, path: str|Path) -> Masks:
        self.masks = restore_cache(path)     

    def copy(self) -> Masks:
        new_masks = Masks()
        new_masks.masks = copy.deepcopy(self.masks)
        return new_masks
    
    def __getitem__(self, index):
        return self.masks[index]
    
    def __len__(self):
        return len(self.masks)
    
    def __iter__(self):
        return iter(self.masks)
    
    def __repr__(self):
        s = []
        for key, masks in self.masks.items():
            s.append(f"{key}: {len(masks)} mask{'' if len(masks) == 1 else 's'}")
        return "{" + ", ".join(s) + "}"
    
    def __eq__(self, other):
        if not isinstance(other, Masks):
            return False
        for key, masks in self.masks.items():
            if key not in other.masks:
                return False
            if len(masks) != len(other[key].masks):
                return False
            for i, mask in enumerate(masks):
                if mask != other[key][i]:
                    return False
        return True
    
    def __bool__(self):
        return bool(self.masks)

@dataclass
class Tomograms:
    raw: np.ndarray | None = field(default=None, repr=False)
    multilooked: np.ndarray | None = field(default=None, repr=False)
    filtered: np.ndarray | None = field(default=None, repr=False)
    profile: Profile | None = field(default=None, repr=False)
    height: list[float] = field(default_factory=list)

    def __repr__(self):
        return f"Tomograms(raw={self.raw is not None}, multilooked={self.multilooked is not None}, filtered={self.filtered is not None})"
    
    @classmethod
    def load(cls, path: str|Path) -> 'Tomograms':
        """
        Create a Tomograms instance from a directory containing tomogram files.
        Path validation is delegated to TomoScene.load, and this method should only be called from there.
        """
        path = Path(path)

        raw_path = path / 'raw_tomogram.tif'
        multilooked_path = path / 'multilooked_tomogram.tif'
        filtered_path = path / 'filtered_tomogram.tif'

        # Check if the files exist
        if not raw_path.exists() or not multilooked_path.exists() or not filtered_path.exists():
            raise FileNotFoundError("One or more tomogram files are missing in the directory.")
        
        # Function to read a tomogram file
        def read_tomogram(file_path: Path) -> tuple[np.ndarray,Profile]:
            def is_complex(tags):
                real_slices = set()
                imag_slices = set()
                for t in tags:
                    role = t.get("role")
                    slice_idx = t.get("slice")
                    if role == "real":
                        real_slices.add(slice_idx)
                    elif role == "imag":
                        imag_slices.add(slice_idx)
                    else:
                        return False
                return real_slices == imag_slices and len(real_slices) > 0

            def is_real(tags):
                roles = [t.get("role") for t in tags]
                if 'imag' in roles:
                    return False
                return True

            with rasterio.open(file_path) as src:
                tags = [src.tags(i+1) for i in range(src.count)]
                if is_complex(tags):
                    half = src.count // 2
                    real = np.stack([src.read(i+1) for i in range(half)])
                    imag = np.stack([src.read(i+1+half) for i in range(half)])
                    return real + 1j * imag, src.profile
                elif is_real(tags):
                    return src.read(), src.profile
                else:
                    raise ValueError(f"Tomogram {file_path} contains imaginary slices, \
                                     but they cannot be matched against real slices.\n \
                                     Tags: {tags}")
        
        # Load the tomograms from the files
        with ThreadPoolExecutor() as executor:
            futures = {
                'raw': executor.submit(read_tomogram, raw_path),
                'multilooked': executor.submit(read_tomogram, multilooked_path),
                'filtered': executor.submit(read_tomogram, filtered_path)
            }
            result = futures['raw'].result()
            raw,_ = result if result else (None, None)
            result = futures['multilooked'].result()
            multilooked,_ =  result if result else (None, None)
            result = futures['filtered'].result()
            filtered, profile = result if result else (None, None)

        return cls(raw=raw, multilooked=multilooked, filtered=filtered, profile=profile)
    
    def save(self, tomo_dir: str|Path):
        tomo_dir = Path(tomo_dir)
        def save_tomogram(array, filename):
            if array is not None:
                if array.iscomplexobj():
                    c = True
                    num_slices = array.shape[0]
                    array = np.concatenate([array.real, array.imag], axis=0)
                else:
                    c = False
                    num_slices = array.shape[0]
                profile = self.profile.copy()
                profile.update({
                    'driver': 'GTiff',
                    'count': array.shape[0] ,
                    'height': array.shape[1],
                    'width': array.shape[2],
                    'dtype': array.dtype
                })
                with rasterio.open(tomo_dir/filename, 'w', **profile) as dst:
                    dst.write(array)
                    for i in range(num_slices):
                        dst.update_tags(i+1, role='real', slice=i)
                        if c:
                            dst.update_tags(i+num_slices+1, role='imag', slice=i)


        save_tomogram(self.raw, 'raw_tomogram.tif')
        save_tomogram(self.multilooked, 'multilooked_tomogram.tif')
        save_tomogram(self.filtered, 'filtered_tomogram.tif')

    def copy(self) -> 'Tomograms':
        new_tomograms = Tomograms()
        new_tomograms.raw = self.raw.copy() if self.raw is not None else None
        new_tomograms.multilooked = self.multilooked.copy() if self.multilooked is not None else None
        new_tomograms.filtered = self.filtered.copy() if self.filtered is not None else None
        new_tomograms.profile = self.profile.copy() if self.profile is not None else None
        return new_tomograms

    def get(self, key):
        return getattr(self, key, None)

@dataclass
class Multilook:
    parent: "TomoInfo" = field(repr=False,compare=False)
    factor: int = 1

    @property
    def tomogram(self) -> np.ndarray:
        if self.parent.tomograms.raw is None:
            raise ValueError("No raw tomogram data available.")
        if self.parent.tomograms.multilooked is None:
            self.apply()
        return self.parent.tomograms.multilooked
    
    @property
    def res(self) -> float:
        if self.parent.res is None:
            raise ValueError("No parent resolution available.")
        return self.parent.res * self.factor
    
    @property
    def profile(self) -> Profile:
        if self.parent.tomograms.profile is None:
            raise ValueError("No parent profile available.")
        profile = self.parent.tomograms.profile.copy()
        profile.update({
            'width': (self.parent.tomograms.profile['width'] + self.factor - 1) // self.factor,
            'height': (self.parent.tomograms.profile['height'] + self.factor - 1) // self.factor,
            'transform': self.parent.tomograms.profile['transform'] * Affine.scale(self.factor, self.factor)
        })
        return profile
    
    def apply(self, factor: int = None, npar: int = os.cpu_count(), RR: bool = True):
        if not factor:
            factor = self.factor
        elif int(factor) > 1:
            self.factor = int(factor)
        else:
            raise ValueError("Multilook factor must be a positive integer.")
        
        self.parent.tomograms.multilooked = multilook(np.abs(self.parent.tomograms.raw)**2, ds=self.factor,
                                                      npar=npar)
        self.parent.stats.collect('multilooked', RR=RR)
    
    def copy(self) -> 'Multilook':
        new_multilook = Multilook(parent=self.parent, factor=self.factor)
        return new_multilook

@dataclass
class Filter:
    parent: 'TomoInfo' = field(repr=False)
    sigma_xi: float = 0.9
    size: int = 9
    point_percentile: float = 98.0
    point_threshold: int = 15

    @property
    def tomogram(self) -> np.ndarray:
        if self.parent.tomograms.raw is None:
            raise ValueError("No raw tomogram data available.")
        if self.parent.tomograms.filtered is None:
            self.apply()
        return self.parent.tomograms.filtered
    
    @property
    def res(self) -> float:
        return self.parent.res
    
    @property
    def profile(self) -> Profile:
        return self.parent.tomograms.profile
    
    def apply(self, sigma_xi: float = None, size: int = None, point_percentile: float = None, 
              point_threshold: int = None, npar: int = os.cpu_count(), RR: bool = False):
        if not sigma_xi:
            sigma_xi = self.sigma_xi
        else:
            self.sigma_xi = float(sigma_xi)
        if not size:
            size = self.size
        else:
            self.size = int(size)
        if not point_percentile:
            point_percentile = self.point_percentile
        else:
            self.point_percentile = float(point_percentile)
        if not point_threshold:
            point_threshold = self.point_threshold
        else:
            self.point_threshold = int(point_threshold)
        self.parent.tomograms.filtered = filter(np.abs(self.parent.tomograms.raw)**2, sigma_xi=sigma_xi, 
                                                size=size, point_percentile=point_percentile, point_threshold=point_threshold,
                                                npar=npar)
        
        self.parent.stats.collect('filtered', RR=RR)

    def copy(self) -> 'Filter':
        new_filter = Filter(parent=self.parent, sigma_xi=self.sigma_xi, size=self.size,
                            point_percentile=self.point_percentile, point_threshold=self.point_threshold)
        return new_filter

@dataclass
class TomoStats:
    parent: TomoInfo | SceneStats = field(repr=False,compare=False)
    stats: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def copy(self) -> TomoStats:
        new_stats = TomoStats()
        for key, value in self.stats:
            new_stats[key] = value.copy()
        return new_stats
    
    def save(self, band_dir: str|Path):
         # Save main statistics
        for key in ['raw', 'multilooked', 'filtered']:
            if key in self.stats:
                self.stats[key].to_csv(band_dir / f'{key}_statistics.csv', index=False)

        # Save masked statistics
        masked_stats_dir = band_dir / 'masked_statistics'
        masked_stats_dir.mkdir(exist_ok=True)
        for stat_key, df in self.stats.items():
            if '_' in stat_key:
                df.to_csv(masked_stats_dir / f'{stat_key}_statistics.csv', index=False)

    def collect(self, layers: str | list[str] = ['raw', 'multilooked', 'filtered','masked'], RR: bool = False):
        if isinstance(layers, str):
            layers = [layers]
        if any(layer not in ['raw', 'multilooked', 'filtered','masked'] for layer in layers):
            raise ValueError("The different layers are 'raw', 'multilooked', 'filtered' and 'masked'.")
        for layer in layers:
            if layer != 'masked':
                self[layer] = collect_statistics(self.parent.tomograms.get(layer), height=self.parent.tomograms.height)

            else:
                for mask in self.parent.masks:
                    for l in  ['raw', 'multilooked', 'filtered']:
                        layer_name = mask.name + '_' + l
                        self[layer_name] = collect_statistics(mask.apply(self.parent.tomograms.get(l), multilooked=(l=='multilooked')), 
                                                              height=self.parent.tomograms.height,circ=False)
                
        if RR:
            RR_estimate, cFactor = estimaterr(self.parent.multilook.tomogram,)
            self['multilooked']['RR'] = RR_estimate
            self['multilooked']['cFactor'] = cFactor

            apply_variable_descriptions(self['multilooked'])

    @classmethod
    def load(cls, parent: 'TomoInfo'|'SceneStats', path: Path|str, cached: bool) -> TomoStats:
        path = Path(path)
        stats = cls(parent=parent)
        # Load main statistics files
        for key in ['raw', 'multilooked', 'filtered']:
            stats_file = path / f'{key}_statistics.csv'
            if stats_file.exists():
                df = pd.read_csv(stats_file)
                stats.stats[key] = df

                apply_variable_descriptions(df)

        # Load masked statistics if cached masks are used
        if cached:
            masked_stats_dir = path / 'masked_statistics'
            if masked_stats_dir.is_dir():
                for mask in parent.masks:
                    mask_name = mask.name if hasattr(mask, 'name') else None
                    if mask_name:
                        for key in ['raw', 'multilooked', 'filtered']:
                            masked_file = os.path.join(masked_stats_dir, f'{mask_name}_{key}_statistics.csv')
                            if os.path.exists(masked_file):
                                df = pd.read_csv(masked_file)
                                stat_key = f'{mask_name}_{key}'
                                stats.stats[stat_key] = df

                                apply_variable_descriptions(df)

        return stats
    
    def __getitem__(self, index):
        return self.stats[index]
    
    def __setitem__(self, layer: str, value: pd.DataFrame):
        self.stats[layer] = value

    def __len__(self):
        return len(self.stats)
    
    def __iter__(self):
        return iter(self.stats)
    
    def __repr__(self):
        frames = []
        N = len(self.stats)
        for key, frame in self.stats:
            if frame:
                frames.append(f"{key}")

        return f"{N} data frames: " + ", ".join(frames)

    def __bool__(self):
        return bool(self.stats)

@dataclass
class SceneStats:
    id: str
    stats: dict[str, TomoStats] = field(init=False)

    def keys(self):
        return self.stats.keys()
    
    def values(self):
        return self.stats.values()
    
    def items(self):
        return self.stats.items()
    
    @property
    def bands(self) -> list[str]:
        return list(self.keys())
    
    def get(self, band: str) -> TomoStats:
        return self.tomograms.get(band)
    
    def add(self, stats: TomoStats, band: str, overwrite: bool = False) -> None:
        bands = ['phh','cvv','lhh','lhv','lvh','lvv', 'phh1','phh0','cvv1','cvv0']
        if band not in bands:
            raise ValueError(f"Valid bands are: {bands}")
        if band in self.bands and not overwrite:
            raise RuntimeError(f"The SceneStats {self.id} already contained a {band} band TomoStats.")
        self[band] = stats
        stats.parent = self

    def copy(self) -> 'SceneStats':
        new_scene = SceneStats(id=self.id)
        for band, stats in self.items():
            new_scene[band] = stats.copy()
        return new_scene
    
    
    @classmethod
    def load(cls, path: str|Path, npar: int = os.cpu_count()) -> 'SceneStats':
        path = Path(path)
        # Check if the path exists
        if not path.exists():
            raise FileNotFoundError(f"'{path}' not found. Check the path or file permissions.")
        # Check if the path is a .tomo directory
        if not path.is_dir() or not path.suffix == ".tomo":
            raise ValueError(f"'{path}' is not a valid .tomo directory.")
        
        scene_stats = cls(id=path.stem)

        bands = [band for band in path.iterdir() if band.is_dir() and band.name in 
                 ['phh','cvv','lhh','lhv','lvh','lvv', 'phh1','phh0','cvv1','cvv0']]
        with ThreadPoolExecutor(max_workers = npar) as executor:
            future_stats = {executor.submit(TomoStats.load, parent=scene_stats, path=band, cached=True): band for band in bands}
            for future in as_completed(future_stats):
                band = future_stats[future]
                try:
                    stats = future.result()
                    scene_stats[band] = stats
                except Exception as e:
                    warn(f"Failed to load {band}: {e}")

        return scene_stats
    
@dataclass
class TomoInfo:
    band: str = None
    width: float = None
    res: float = None
    vres: float = None
    bottom: float = None
    top: float = None
    smo: float = None
    ham: float = 0
    refr: float = 1
    lat: float = None
    lon: float = None
    DC: float = 0
    DL: float = 999
    HC: float = 120
    HV: float = 999
    thresh: float = 10
    squint: float = 0
    text: str = None
    category: str = None
    tomograms: Tomograms = field(default_factory=Tomograms)
    masks: Masks = field(init=False)
    multilook: Multilook = field(init=False)
    filter: Filter = field(init=False)
    stats: Dict[str, pd.DataFrame] = field(init=False, default_factory=dict, repr=False)
    _slices: SliceInfo = field(default_factory=SliceInfo, repr=False)
    _scene: 'TomoScene' = field(default=None, repr=False, compare=False)

    TOMOGRAM_PARAMETERS: ClassVar[list[str]] = ['band', 'width', 'res', 'smo', 'ham', 'lat', 'lon', 
                                                'DC', 'DL', 'HC', 'HV', 'squint']

    def __post_init__(self) -> None:
        self.masks = Masks(parent=self)
        self.multilook = Multilook(parent=self)
        self.filter = Filter(parent=self)
        self.stats = TomoStats(parent=self)
    
    @property
    def slices(self) -> SliceInfo:
        return self._slices.copy()
    
    @property
    def parameters(self) -> dict:
        return {'band': self.band,
            'width': self.width, 'res': self.res, 'vres': self.vres,
            'bottom': self.bottom, 'top': self.top, 'smo': self.smo,
            'ham': self.ham, 'refr': self.refr, 'lat': self.lat, 'lon': self.lon,
            'DC': self.DC, 'DL': self.DL, 'HC': self.HC, 'HV': self.HV,
            'thresh': self.thresh, 'squint': self.squint, 'text': self.text,
            'category': self.category, 'height': self.tomograms.height.tolist(),
            'multilook': self.multilook.factor if self.multilook else 1,
            'sigma_xi': self.filter.sigma_xi, 'filter_size': self.filter.size, 
            'point_percentile': self.filter.point_percentile, 'point_threshold': self.filter.point_threshold
        }
    
    @property
    def info(self) -> dict:
        # Get band specific parameters from the parent TomoScene
        if self._scene is None:
            raise AttributeError("TomoInfo object has not been initialized as a part of a TomoScene")
            band = 'C-band'
        info = self._scene._info
        info[self.band] = self.parameters

    def update(self) -> None:
        self.masks.update()

    @classmethod
    def forge(cls, slices: SliceInfo, multilook: int = 1, sigma_xi: float = 0.9, 
              filter_size: int = 9, point_percentile: float = 98.0, point_threshold: int = 9,
              fused: bool = True, sub: bool = True, sup: bool = True, canopy: bool = True, 
              npar: int = os.cpu_count(), RR: bool = True, masks: str = "") -> 'TomoInfo':
        """
        Initializes a TomoInfo instance from a SliceInfo with slices from the same tomogram.
        """
        if not slices:
            return TomoInfo()
        if len(slices) == 1:
            return TomoInfo()
        # Check that all slices belong to the same tomogram
        elif len(slices.tomograms()) > 1:
            raise ValueError("All slices must be from the same tomogram.")
        # Categorize
        categories = slices.categorize()
        tomos = defaultdict(TomoInfo)
        for cat, group in categories.items():
            if len(group) == 1:
                continue # Skip groups with just a single slice
            if cat == 'sub':
                long_category = 'subsurface'
                # Group according to refractive index and threshold parameter
                subgroups = group.group(['refr','thresh'], list=True)
                if len(subgroups) > 1:
                    group = next((g for g in subgroups if len(g) > 1), None)
                    if not group:
                        continue
                    warn("Multiple sub-surface groups with varying refractive index and threshold parameter detected. Selecting first group.")
            elif cat == 'canopy':
                long_category = 'canopy'
                # Group according to refractive index and hoff parameter
                subgroups = group.group(['refr','hoff'], list=True)
                if len(subgroups) > 1:
                    group = next((g for g in subgroups if len(g) > 1), None)
                    if not group:
                        continue
                    warn("Multiple canopy groups with varying refractive index and height offset detected. Selecting first group.")
            else:
                long_category = 'supersurface'
            print(f"Forging {long_category} tomogram from {len(group)} slices ...")
            # Sort according to height
            group.sort('height')
            base = group[0]
            # Calculate bottom and top
            bottom = base.height
            top = group[-1].height
            # Calculate optimal vres
            vres = calculate_vres(group)
            # Get refractive index
            if cat == 'sup':
                refr = 1
            else:
                refr = base.refr
            # Construct tomogram    
            tomogram = np.stack([s.image for s in group], axis=0)
            info = TomoInfo()
            info.band = base.band
            info.width = base.width
            info.res = base.res
            info.vres = vres
            info.bottom = bottom
            info.top = top
            info.smo = base.smo
            info.ham = base.ham
            info.refr = refr
            info.lat = base.lat
            info.lon = base.lon
            info.DC = base.DC
            info.DL = base.DL
            info.HC = base.HC
            info.HV = base.HV
            info.thresh = base.thresh
            info.squint = base.squint
            info.text = base.text
            info.category = cat
            info._slices = group.copy()
            info.tomograms.raw = tomogram
            info.tomograms.profile = base.profile.copy()
            info.tomograms.height = slices.get('height')
            info.multilook.factor = multilook
            info.filter.sigma_xi = sigma_xi
            info.filter.size = filter_size
            info.filter.point_percentile = point_percentile
            info.filter.point_threshold = point_threshold

            # Update slice count
            info.tomograms.profile.update({
                'count': len(info.tomograms.height)
            })

            tomos[cat] = info

        if fused and 'sub' in tomos and 'sup' in tomos:
            # Fuse
            tomo = tomos['sub'].fuse(tomos['sup'])
            print("Subsurface and supersurface tomograms fused. Processing fused tomogram.")
        elif sub and 'sub' in tomos:
            tomo = tomos['sub']
            print("Processing subsurface tomogram.")
        elif sup and 'sup' in tomos:
            tomo = tomos['sup']
            print("Processing supersurface tomogram.")
        elif canopy and 'canopy' in tomos:
            tomo = tomos['canopy']
            print("Processing canopy tomogram.")

        # Collect raw statistics
        tomo.stats.collect('raw', RR=RR)
        print("Raw statistics collected.")
        # Multilook
        tomo.multilook.apply(npar=npar,RR=RR)
        print("Multilooking done.")
        # Filter
        tomo.filter.apply(npar=npar,RR=RR)
        print("Filtering done.")
        # Update masks
        tomo.masks.update(masks)
        print(f"{len(tomo.masks)} mask(s) loaded.")

        return tomo

    @classmethod
    def load(cls, path: str|Path, cached: bool = False) -> 'TomoInfo':
        """
        Create a TomoInfo instance from a band  sub-directory of a .tomo directory.
        Path validation is delegated by TomoScene.load() and this method should only be called from there.
        """

        path = Path(path)
        # Construct the full path to the JSON file
        processing_params_file = path / 'processing_parameters.json'
        tomo_path,band = path.parent, path.name
        if not (processing_params_file.exists() and processing_params_file.is_file()):
            raise FileNotFoundError(f"Processing parameters file '{processing_params_file}' not found in the {band} sub-directory of {tomo_path}.")
        # Load the JSON data
        with open(processing_params_file, 'r') as f:
            data = json.load(f)

        # Redundancy: confirm band
        if band != data.get('band'):
            raise RuntimeError(f"The recorded band {data.get('band')} does not match band directory {band}.")
        
        # Construct the Tomograms instance
        tomograms = Tomograms.load(path)
        tomograms.height = np.array(data.get('height'))

        # Create a TomoInfo instance from the loaded data
        tomo_info = cls(
            date=data.get('date'),
            band=band,
            width=data.get('width'),
            res=data.get('res'),
            vres=data.get('vres'),
            bottom=data.get('bottom'),
            top=data.get('top'),
            smo=data.get('smo'),
            ham=data.get('ham', 0),
            refr=data.get('refr', 1),
            lat=data.get('lat'),
            lon=data.get('lon'),
            DC=data.get('DC', 0),
            DL=data.get('DL', 999),
            HC=data.get('HC', 120),
            HV=data.get('HV', 999),
            thresh=data.get('thresh', 10),
            squint=data.get('squint', 0),
            text=data.get('text'),
            category=data.get('category'),
            tomograms=tomograms,
        )
        # Load slices
        slice_directory = path / '.slices'
        tomo_info._slices = SliceInfo.scan(slice_directory)
        
        # Set masks
        if cached:
            # Construct full path to the cached_masks subdirectory
            masks_dir = path / 'cached_masks'
            if not masks_dir.is_dir():
                raise NotADirectoryError(f"Cached masks directory {masks_dir} is not a directory.")
            # Load the cached directory
            tomo_info.masks.read(masks_dir)
        else:
            # Load masks from the TOMOMASKS folder
            tomo_info.masks.update()
        
        # Set multilook
        tomo_info.multilook = data.get('multilook', 1)

        # Set filter parameters
        tomo_info.filter.sigma_xi = data.get('sigma_xi', 0.9)
        tomo_info.filter.size = data.get('filter_size', 9)
        tomo_info.filter.point_percentile = data.get('point_percentile', 98)
        tomo_info.filter.point_threshold = data.get('point_threshold', 6)

        # Load statistics
        tomo_info.stats = TomoStats.load(parent=tomo_info, path=path, cached=cached)

        return tomo_info

    def save(self, band_dir: str|Path):
        band_dir = Path(band_dir)
        band_dir.mkdir(exist_ok=True)
        # Save processing_parameters.json
        processing_params = {'band': self.band,
            'width': self.width, 'res': self.res, 'vres': self.vres,
            'bottom': self.bottom, 'top': self.top, 'smo': self.smo,
            'ham': self.ham, 'refr': self.refr, 'lat': self.lat, 'lon': self.lon,
            'DC': self.DC, 'DL': self.DL, 'HC': self.HC, 'HV': self.HV,
            'thresh': self.thresh, 'squint': self.squint, 'text': self.text,
            'category': self.category, 'height': self.tomograms.height.tolist(),
            'multilook': self.multilook.factor if self.multilook else 1,
            'sigma_xi': self.filter.sigma_xi, 'filter_size': self.filter.size, 
            'point_percentile': self.filter.point_percentile, 'point_threshold': self.filter.point_threshold
        }
        with open(band_dir / 'processing_parameters.json', 'w') as f:
            json.dump(processing_params, f, indent=4)

        # Save tomograms
        self.tomograms.save(band_dir)

        # Save statistics
        self.stats.save(band_dir)

        # Save cached masks
        if self.masks and hasattr(self.masks, 'masks'):
            cache_masks(self.masks.masks,folder=band_dir/'cached_masks')

        # Save slices
        slice_folder = band_dir/'.slices'
        slice_folder.mkdir(exist_ok=True)
        for s in self._slices.get('path'):
            target_path = slice_folder / s.name

            if not target_path.exists():
                shutil.copy2(s,target_path)

    def fuse(self, other: 'TomoInfo', RR: bool = True) -> 'TomoInfo':
        """
        Fuses a subsurface or supersurface category TomoInfo instance with one of the other category.
        
        """
        if self.category == 'sub':
            if not other.category == 'sup':
                raise ValueError("Only tomograms of subsurface and supersurface categories can be fused.")
            
            sub = self
            sup = other
        elif self.category == 'sup':
            if not other.category == 'sub':
                raise ValueError("Only tomograms of sub and sup categories can be fused.")
            
            sub = other
            sup = self
        else:
            raise ValueError("Only tomograms of sub and sup categories can be fused.")
        
        if sub != sup:
            raise ValueError("Only tomograms with the same processing parameters can be fused.")
        
        fused = sub.copy()
        fused.category = 'fused'
        fused._slices = np.concatenate([sub._slices, sup._slices])
        fused.tomograms.raw = np.concatenate([sub.tomograms.raw, sup.tomograms.raw[1:]], axis=-1)
        fused.tomograms.height = np.concatenate([sub.tomograms.height, sup.tomograms.height[1:]])
        fused.vres = calculate_vres(fused.slices)
        for key, frame in fused.stats.items():
            fused[key] = pd.concat(frame, sup.stats[key][1:], ignore_index=True)

        if sub.tomograms.multilooked and sup.tomograms.multilooked:
            if sub.multilook.factor == sup.multilook.factor:
                fused.tomograms.multilooked = np.concatenate([sub.tomograms.multilooked,sup.tomograms.multilooked], axis=-1)
            else:
                warn("Different multilook factors, selecting subsurface factor and applying.")
                fused.multilook.apply()

        if sub.tomograms.filtered and sup.tomograms.filtered:
            if sub.filter == sup.filter:
                fused.tomograms.filtered = np.concatenate([sub.tomograms.filtered,sup.tomograms.filtered], axis=-1)
            else:
                warn("Different filter parameters, selecting subsurface filter and applying.")
                fused.filter.apply()

        if sub.masks and sup.masks:
            if sub.masks != sup.masks:
                warn("Different masks applied to tomograms, updating masks.")
                fused.masks.update()

    def copy(self) -> 'TomoInfo':
        new_info = TomoInfo(
            band=self.band,
            width=self.width,
            res=self.res,
            vres=self.vres,
            bottom=self.bottom,
            top=self.top,
            smo=self.smo,
            ham=self.ham,
            refr=self.refr,
            lat=self.lat,
            lon=self.lon,
            DC=self.DC,
            DL=self.DL,
            HC=self.HC,
            HV=self.HV,
            thresh=self.thresh,
            squint=self.squint,
            text=self.text,
            category=self.category,
            _slices=self._slices.copy()
        )
        new_info.tomograms = self.tomograms.copy()
        new_info.masks = self.masks.copy()
        new_info.masks.parent = new_info # Update parent reference in masks
        new_info.multilook = self.multilook.copy() if self.multilook else None
        new_info.multilook.parent = new_info  # Update parent reference in multilook
        new_info.filter = self.filter.copy()
        new_info.filter.parent = new_info # Update parent reference in filter
        new_info.stats = self.stats.copy()
        new_info.stats.parent = new_info # Update parent reference in stats
        new_info._scene = None #

        return new_info

    def get(self, key: str):
        return getattr(self, key, None)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, 'TomoInfo'):
            return False
        return all(self.get(key) == other.get(key) for key in TomoInfo.TOMOGRAM_PARAMETERS)

@dataclass
class TomoScene:
    id: str = ""
    date: datetime = None
    spiral: int = None
    tomograms: Dict[str, TomoInfo] = field(default_factory=dict)
    _info: Dict[str, float] = field(default_factory=dict)
    moco: pd.DataFrame = field(default_factory=pd.DataFrame)
    _model: SARModel = None
    
    @property
    def model(self) -> SARModel:
        if self._model is None:
            self._model = SARModel(self.moco)
        return self._model

    def items(self):
        return self.tomograms.items()
    
    def values(self):
        return self.tomograms.values()
    
    def keys(self):
        return self.tomograms.keys()

    def update(self) -> None:
        for tomo in self.values():
            tomo.update()
        
    @property
    def bands(self) -> list[str]:
        return list(self.keys())
    
    @property
    def info(self) -> dict:
        info = self._info
        for band in self.bands:
            info[band] = self[band].parameters
        return info
    
    def get(self, band: str) -> TomoInfo:
        return self.tomograms.get(band)
    
    def add(self, tomo: TomoInfo, overwrite: bool = False):
        if tomo.band in self.bands and not overwrite:
            raise RuntimeError(f"The TomoScene {self.id} already contained a {tomo.band} band TomoInfo.")
        self[tomo.band] = tomo
        tomo._scene = self

    def copy(self) -> 'TomoScene':
        new_scene = TomoScene(id=self.id, date=self.date, spiral=self.spiral)
        for band, tomos in self.items():
            new_scene[band] = tomos.copy()
        new_scene._info = self._info.copy()
        new_scene.moco = self.moco.copy()
        new_scene._model = self._model.copy()
        return new_scene
    
    @classmethod
    def load(cls, path: str|Path = '.', cached: bool = False, npar: int = os.cpu_count()) -> 'TomoScene':
        """
        Create a TomoScene instance from a .tomo directory.
        """
        path = Path(path)
        # Check if the path exists
        if not path.exists():
            raise FileNotFoundError(f"'{path}' not found. Check the path or file permissions.")
        # Check if the path is a .tomo directory
        if not path.is_dir() or not path.suffix == ".tomo":
            raise ValueError(f"'{path}' is not a valid .tomo directory.")
        
        tomo_scene = cls(id=path.stem)
        
        # Construct the full path to the SAR parameters file
        info_file = path / 'flight_info.json'
        if not info_file.exists():
            raise FileNotFoundError(f"Flight info file '{info_file}' not found in the .tomo directory.")
        # Load the SAR parameters
        with open(info_file, 'r') as f:
            flight_info = json.load(f)

        # Store SAR parameters
        tomo_scene.date = datetime.fromisoformat(flight_info['date'])
        tomo_scene.spiral = flight_info['spiral']
        tomo_scene.info = flight_info['info']

        # Construct the full path to the .moco cut CSV file
        moco_file = path / 'moco_cut.csv'
        if not moco_file.exists():
            raise FileNotFoundError(f".moco cut CSV file '{moco_file}' not found in the .tomo directory.")
        # Load the moco data
        tomo_scene.moco = pd.read_csv(moco_file)
        tomo_scene._model = SARModel(tomo_scene.moco)

        # Load the tomograms
        bands = [band for band in path.iterdir()
                        if band.is_dir() and band.name in ['phh','cvv','lhh','lhv','lvh','lvv',
                                                            'phh1','phh0','cvv1','cvv0']]
        with ThreadPoolExecutor(max_workers = npar) as executor:
            future_tomos = {executor.submit(TomoInfo.load, path=band, cached=cached): band for band in bands}
            for future in as_completed(future_tomos):
                band = future_tomos[future]
                try:
                    tomo = future.result()
                    tomo_scene[band] = tomo
                    tomo._scene = tomo_scene
                except Exception as e:
                    warn(f"Failed to load {band}: {e}")
    
    def save(self, folder: str|Path = "."):
        folder = Path(folder)
        tomo_dir = folder  / f"{self.id}.tomo"
        tomo_dir.mkdir(exist_ok=True)

        # Save SAR parameters
        with open(tomo_dir / 'flight_info.json', 'w') as f:
            json.dump({
                'date': self.date.isoformat(timespec='seconds'),
                'spiral': self.spiral,
                'info': self._info
                }, f, indent=4)

        # Save .moco cut explicitly as .csv
        self.moco.to_csv(tomo_dir / 'moco_cut.csv', index=False)

        # Save tomogram data
        for band, tomo in self.tomograms.items():
            band_dir = tomo_dir / band
            tomo.save(band_dir)
    
    def __iter__(self):
        return iter(self.tomograms)
    
    def __getitem__(self, band: str) -> TomoInfo:
        return self.tomograms[band]
    
    def __setitem__(self, band: str, value: TomoInfo):
        self.tomograms[band] = value

@dataclass
class TomoScenes:
    def __init__(self, scenes: list[TomoScene]):
        self.scenes = {}
        for scene in scenes:
            key = (scene.date.date(), scene.date.time(), scene.spiral)
            if key in self.scenes:
                raise ValueError("Only lists of unique TomoScene objects can be used to initialize a TomoScenes object.")
            self.scenes[key] = scene


    KEY_TYPES: ClassVar = datetime|date|tuple[date,time]|tuple[datetime,int]|tuple[date,time,int]

    def items(self):
        return self.scenes.items()
    
    def values(self):
        return self.scenes.values()
    
    def keys(self):
        return self.scenes.keys()

    def copy(self) -> 'TomoScenes':
        new_scenes = TomoScenes()
        new_scenes.scenes = copy.deepcopy(self.scenes)
        return new_scenes
    
    def save(self, folder: str = "."):
        for scene in self.scenes:
            scene.save(folder)

    def list(self):
        print(f"Containing {len(self)} scenes:")
        for i, scene in enumerate(self):
            print(f"\t{i}: {scene.id} with bands {scene.bands}")
    
    def update(self):
        for scene in self.values():
            scene.update()
    
    @classmethod
    def load(self, path: str|Path = ".", cached: bool = False, npar: int = os.cpu_count) -> 'TomoScenes':
        path = Path(path)
        tomo_scenes = TomoScenes()
        if path.is_dir():
            tomo_dirs = [d for d in path.iterdir() if d.is_dir() and d.suffix == '.tomo']

            with ThreadPoolExecutor(max_workers=npar) as executor:
                interior_npar = npar // len(tomo_dirs)
                future_to_path = {executor.submit(TomoScene.load, tomo_path, cached=cached, npar=interior_npar): tomo_path for tomo_path in tomo_dirs}
                for future in as_completed(future_to_path):
                    tomo_path = future_to_path[future]
                    try:
                        scene = future.result()
                        tomo_scenes.append(scene)
                    except Exception as e:
                        print(f"Warning: Failed to load {tomo_path}: {e}")

        return tomo_scenes
  
    @staticmethod
    def _is_date_time_tuple(key):
        return (
            isinstance(key, tuple) and
            len(key) == 2 and
            isinstance(key[0], date) and
            isinstance(key[1], time)
        )
    
    @staticmethod
    def _is_datetime_int_tuple(key):
        return (
            isinstance(key, tuple) and
            len(key) == 2 and
            isinstance(key[0], datetime) and
            isinstance(key[1], int)
        )
    
    @staticmethod
    def _is_date_time_int_tuple(key):
        return (
            isinstance(key, tuple) and
            len(key) == 3 and
            isinstance(key[0], date) and
            isinstance(key[1], time) and
            isinstance(key[2], int)
        )
    
    def __getitem__(self, key):
        # Check whether a list was passed as key
        if isinstance(key, list):
            key = tuple(key)
        # If a single-element tuple was passed, get the first element
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        # If a string was passed, parse as datetime (or date or time)
        if isinstance(key,str):
            key = parse_datetime_string(key)
        # Check for strings in tuple, and parse as datetime (or date or time)
        if isinstance(key,tuple):
            for i, k in enumerate(key):
                if isinstance(k, str):
                    key[i] = parse_datetime_string(k)
        ## Start handling valid key types (listed in self.KEY_TYPES)
        # These types return a new TomoScenes object or a single TomoScene
        if isinstance(key, datetime):
            key = (key.date, key.time)
        if isinstance(key, date):
            scenes = [s for k, s in self.items() if k[0] == key]
            if len(scenes) == 1:
                return scenes[0]
            else:
                return TomoScenes(scenes)
        if self._is_date_time_tuple(key):
            scenes = [s for k, s in self.items() if k[0] == key[0] and k[1] == key[1]]
            if len(scenes) == 1:
                return scenes[0]
            else:
                return TomoScenes(scenes)
        # These types returns a single TomoScene object or raises KeyError
        if self._is_datetime_int_tuple(key):
            key = (key[0].date, key[0].time, key[1])
            if key in self.scenes:
                return self[key]
            else:
                raise KeyError(f"No scene found for key: {key}")
        if self._is_date_time_int_tuple(key):
            if key in self.scenes:
                return self[key]
            else:
                raise KeyError(f"No scene found for key: {key}")
        # Not a valid key type
        raise TypeError(f"Unsupported key structure: {key}")

    @staticmethod
    def _sort_and_validate_key(key):
        sorted_key =  (None, None, None)
        for k in key:
            if isinstance(k, date):
                sorted_key[0] = k
            elif isinstance(k, time):
                sorted_key[1] = k
            elif isinstance(k, int):
                sorted_key[2] = k
            else:
                raise KeyError(f"Invalid key element: {k} of type {type(k)}")
        if sorted_key[0] is None:
            raise KeyError(f"No date was passed in key parsed as: {key}")
        if sorted_key[1] is None:
            raise KeyError(f"No time was passed in key parsed as: {key}")
        if sorted_key[2] is None:
            raise KeyError(f"No spiral id was passed in key parsed as: {key}")
        return sorted_key
    
    def __setitem__(self, key, item: TomoScene):
        # Validate item
        if not isinstance(item, TomoScene):
            raise TypeError(f"Only TomoScene objects can be set, but you tried to set a {type(item)} object.")
        # Check whether a list was passed as key
        if isinstance(key, list):
            key = tuple(key)
        if isinstance(key,tuple):
            # Check for strings in tuple, and parse as datetime (or date or time)
            for i, k in enumerate(key):
                if isinstance(k, str):
                    key[i] = parse_datetime_string(k)
                # Split datetime objects into date and time
                if isinstance(key[i], datetime):
                    d = key[i].date()
                    t = key[i].time()
                    key[i] = (d, t)
            # Flatten tuple if needed
            flattened = []
            for item in key:
                if isinstance(item, tuple):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            key = tuple(flattened)
            if len(key) != 3:
                raise KeyError(f"Keys must contain date, time and spiral id, but the passed key was parsed as: {key}")
            key = self._sort_and_validate_key(key)
            self.scenes[key] = item            
        else:
            raise KeyError(f"Invalid key type: {key} of type {type(key)}")
        
    def __len__(self):
        return len(self.scenes)
        
    def __iter__(self):
        return iter(self.scenes)
    
    def __bool__(self):
        return bool(self.scenes)
    
    def __repr__(self):
        return f"TomoScenes({len(self.scenes)} scenes)"
    
    def __eq__(self, other):
        if not isinstance(other, TomoScenes):
            return False
        return all(scene in other.scenes for scene in self.scenes) and len(self) == len(other)

# Helper functions
## Regrouping a grouped SliceInfo
def regroup(grouped_dict: Dict[str,SliceInfo], keys: str | list[str], list: bool = False):
    regrouped = defaultdict(lambda: defaultdict(SliceInfo))

    if isinstance(keys,str):
        keys = [keys]

    # Group
    for outer_key, sliceinfo in grouped_dict.items():
        for s in sliceinfo:
            key = tuple(s.get(k) for k in keys)
            regrouped[key][outer_key].append(s)

    # Convert to list if list is set to True
    if list:
        result = [s for key, s in regrouped.items()]
    else:
        result = regrouped
    return result

## Calculate vres from SliceInfo
def calculate_vres(slices: SliceInfo) -> float | None:
    height = slices.get('height')
    bottom = height[0]
    top = height[-1]
    dz = np.diff(height)
    candidates = []
    for v in dz:
        N = (top - bottom)/v
        if np.isclose(N, round(N), atol=1e-6):
            retained = [s for s in slices if np.isclose((s.height - bottom)/v, round((s.height - bottom)/v), atol=1e-6)]
            if len(retained) == round(N) + 1:
                candidates.append(v)
    if candidates:
        vres = min(candidates)
    else:
        vres = None
        warn("No uniform vertical sampling frequency could be found.")

## Parsing a filename for ImageInfo
def parse_filename(path: str) -> ImageInfo:
    p = Path(path)
    default = ImageInfo(filename=p.name, folder=p.parent)

    parts = p.name.split('_')
    if len(parts) < 6:
        return default
    dt = datetime.strptime(parts[1], "%Y-%m-%d-%H-%M-%S-%f")
    default.spiral = int(dt.microsecond / 10**4)
    default.date = dt.replace(microsecond=0)
    default.band = parts[2]
    default.width = float(re.findall(r"[-0-9.]+", parts[3])[0])
    default.res = float(re.findall(r"[-0-9.]+", parts[4])[0])
    default.linuxTime = int(re.findall(r"\d+", parts[5])[0])
    default.smo = default.width / 10 / default.res

    text_found = 0
    for part in parts[6:-1]:
        flag = re.findall(r"^[a-zA-Z]+", part)
        str_val = re.findall(r"[-0-9.]+", part)
        val = float(str_val[0]) if str_val else None

        if flag:
            key = flag[0]
            if key == 'ho':
                default.hoff = val
            elif key == 'ham':
                default.ham = val
            elif key == 'ro':
                default.roff = val
            elif key == 'smo':
                default.smo = val
            elif key == 'ref':
                default.refr = val
            elif key == 'prof':
                default.depth = val
            elif key == 'LA':
                default.lat = val
            elif key == 'LO':
                default.lon = val
            elif key == 'DC':
                default.DC = val
            elif key == 'DL':
                default.DL = val
            elif key == 'HC':
                default.HC = val
            elif key == 'HV':
                default.HV = val
            elif key == 'TH':
                default.thresh = val
            elif key == 'sq':
                default.squint = val
            elif key == 'shm':
                pass
            else:
                text_found += 1
                default.text = part + ("_" + default.text if text_found > 1 else default.text)

    return default

## Masks helpers
def get_masks(raster_profile: Profile, multilooked_profile: Profile, 
              user_mask: str | Path = "") -> dict[str,list[Mask]]:
    """
    Generate binary masks from shapefiles using rasterio and geopandas.

    Parameters:
    - mask_dirs: list of directories containing .shp files
    - raster_profile: rasterio profile dictionary (contains raster size, dtype, etc.)
    - raster_transform: rasterio transform object

    Returns:
    - List of dictionaries with keys 'mask' (binary np.ndarray) and 'name' (shapename)
    """
    mask_paths = Settings().MASKS
    if user_mask:
        mask_paths.append(user_mask)
    masks = defaultdict(list)

    for path in mask_paths:
        # Find all .shp files in the directory
        if path.is_dir():
            shapefiles = path.rglob("*.shp")
        elif path.suffix == ".shp":
            shapefiles = [path]
        else:
            continue

        for shp_path in shapefiles:
            # Read shapefile using geopandas
            gdf = gpd.read_file(shp_path)
            shapename = shp_path.stem

            # Loop through each shape in the shapefile
            for idx, row in gdf.iterrows():
                geometry = row.geometry

                # Create a binary mask using rasterio.features.rasterize
                mask = rasterize(
                    [(geometry, 1)],
                    out_shape=(raster_profile['height'], raster_profile['width']),
                    transform=raster_profile['transform'],
                    fill=0,
                    dtype='uint8'
                ).astype(bool)

                if not np.any(mask):
                    continue        # This shape does not intersect raster

                # Create multilooked binary mask
                multilooked = rasterize(
                    [(geometry, 1)],
                    out_shape=(multilooked_profile['height'], multilooked_profile['width']),
                    transform=multilooked_profile['transform'],
                    fill=0,
                    dtype='uint8'
                ).astype(bool)

                # Generate a shapename
                shape_id = row.get('id', idx)

                # Generate metadata
                metadata = {
                    'source': shp_path,
                    'shape_id': shape_id,
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'bounding_box': geometry.bounds,
                    'generated_on': socket.gethostname(),
                    'profile': str(raster_profile),
                    'multilooked': str(multilooked_profile)
                }

                # Append the mask and name to the list
                masks[shapename].append(Mask(name=shapename, id=shape_id, mask=mask, 
                                             multilooked=multilooked, metadata=metadata))

    return masks

def cache_masks(masks: dict[str,list[Mask]], folder: str|Path = 'cached_masks'):
    """
    Save each mask's numpy array and metadata to a folder.
    Each mask is saved as <name>.npy and <name>.json using the Mask.name attribute.
    If the folder exists, it is deleted and recreated.
    """
    folder = Path(folder)
    # Remove existing folder contents
    if folder.exists():
        for file_path in folder.iterdir():
            if file_path.is_file():
                os.remove(file_path)
            elif file_path.is_dir():
                shutil.rmtree(file_path)
    else:
        folder.mkdir(exist_ok=True)

    # Save each mask and metadata
    for shapename, mask_list in masks:
        mask_folder = folder / f"{shapename}"
        mask_folder.mkdir(exist_ok=True)
        for mask_obj in mask_list:
            mask_path = mask_folder / f"shape_id_{mask_obj.id}.npy"
            np.save(mask_path, mask_obj.mask)

            multilooked_path = mask_folder / f"multilooked_id_{mask_obj.id}.npy"
            np.save(multilooked_path,mask_obj.multilooked)

            metadata_path = mask_folder / f"metadata_id_{mask_obj.id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(mask_obj.metadata, f, indent=4)

def restore_cache(folder: str|Path = 'cached_masks') -> defaultdict[str,list[Mask]]:
    """
    Load cached masks from a folder containing .npy and .json files.
    Returns a list of Mask objects with mask array and metadata restored.
    """
    masks = defaultdict(list)
    folder = Path(folder)
    if not folder.exists():
        warn(f"Folder '{folder}' does not exist.")
        return masks
    if not folder.is_dir():
        warn(f"Folder '{folder}' is not a folder.")
        return masks

    # Collect all .json files and match with corresponding .npy files
    json_files = folder.rglob("*.json")

    for json_file in json_files:
        # Load metadata 
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        name = metadata.get('name')
        id = metadata.get('id')
        
        npy_file = json_file.parent / f"shape_id_{id}.npy"
        ml_file = json_file.parent / f"multilooked_id_{id}.npy"

        if not npy_file.exists() or not ml_file.exists():
            warn(f"Missing .npy file for {json_file}")
            continue

        # Load masks
        mask_array = np.load(npy_file)
        multilooked_array = np.load(ml_file)

        # Create Mask object
        mask_obj = Mask(name=name, id=id, mask=mask_array, multilooked=multilooked_array, metadata=metadata)
        masks[name].append(mask_obj)

    return masks

# Orchestrating functions
def sliceinfo(path: str|Path = '.', filter: ImageInfo = None, read: bool = False,
              npar: int = os.cpu_count()) -> SliceInfo:
    p = Path(path)
    if not p.exists:
        raise FileNotFoundError(f"'{path}' not found. Check the path or file permissions.")

    slice_info = SliceInfo()

    # Ensure path is the full path
    if not p.is_absolute:
        p = p.resolve()

    if p.is_file():
        # If path is a file, parse the filename directly
        if re.search(r'db.*C(?=\.tif$)', p.name):
            try:
                slice_info.append(parse_filename(path))
            except Exception as e:
                print(f"Error parsing file {p.name}: {e}")
        else:
            raise ValueError(f"File '{p.name}' does not match expected pattern for slice info.")
    else:
        tif_files = list(p.glob("dbr*C.tif"))


        for f in tif_files:
            try:
                slice_info.append(parse_filename(f.resove()))
            except Exception as e:
                print(f"Error parsing file {f}: {e}")

    # Apply filter
    if filter:
        slice_info.filter(filter)

    # Read images and georeferences
    if read:
        slice_info.read(npar)
    
    return slice_info

def tomoinfo(path: str|Path) -> dict:
    """Prints info about a .tomo folder"""
    path = Path(path)
    if not path.is_dir() or path.resolve().suffix != ".tomo":
        raise ValueError(f"Invalid path {path}")
    
    with open(path / "flight_info.json", 'r') as info_file:
        info = json.load(info_file)

    for dir in path.iterdir():
        if dir.is_dir():
            with open(dir / "processing_parameters.json", 'r') as band_file:
                info[dir.name] = json.load(band_file)
    
    return info

def tomoload(path: str = '.', cached: bool = True, npar: int = os.cpu_count()) -> TomoScene | TomoScenes:
    """
    Loads TomoScene instances from .tomo directories, collecting them into a TomoScenes if multple are found.
    """
    # yyyy-mm-dd-HH-MM-SS-filename_processing-time.tomo/
    #   |-- flight_info.json
    #   |-- moco_cut.csv
    #   |-- phh
    #   |    |-- processing_parameters.json
    #   |    |-- raw_tomogram.tif
    #   |    |-- multilooked_tomogram.tif
    #   |    |-- filtered_tomogram.tif
    #   |    |-- raw_statistics.csv
    #   |    |-- multilooked_statistics.csv
    #   |    |-- filtered_statistics.csv
    #   |    |-- masked_statistics/
    #   |    |       |-- <mask1>_raw_statistics.csv
    #   |    |       |-- <mask1>_multilooked_statistics.csv
    #   |    |       |-- <mask1>_filtered_statistics.csv
    #   |    |       |-- <mask2>_raw_statistics.csv
    #   |    |       |-- ...
    #   |    |-- cached_masks/
    #   |    |       |-- <mask1>.npy
    #   |    |       |-- <mask1>.json
    #   |    |       |-- <mask2>.npy
    #   |    |       |-- ...
    #   |    |-- .slices/
    #   |    |       |-- ...
    #   |-- cvv
    #   |    |-- ...
    #   |-- lhh
    #   |    |-- ...
    #   |-- ...

    # Ensure path is the full path
    path = Path(path)

    # If path is a single .tomo directory
    if path.suffix == '.tomo':
        print("Returning single TomoScene.")
        return TomoScene.load(path,cached=cached, npar=npar)
    # If path is a folder containing multiple .tomo directories
    tomo_scenes = TomoScenes.load(path=path, cached=cached, npar=npar)

    return tomo_scenes if tomo_scenes else None