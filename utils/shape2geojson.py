import shapefile
from json import dumps

# read the shapefile
shape_file = './data/RPG_2-0__SHP_LAMB93_FR-2017_2017-01-01/RPG/1_DONNEES_LIVRAISON_2017/RPG_2-0_SHP_LAMB93_FR-2017/PARCELLES_GRAPHIQUES.shp'


import geopandas

shp_file = geopandas.read_file(shape_file)
shp_file.to_file('pyshp-all-2000-sentinel.geojson', driver='GeoJSON')