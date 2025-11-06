from pyproj import Transformer

ecef_to_geo = Transformer.from_crs("epsg:4978", "epsg:4979", always_xy=True)
geo_to_ecef = Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)