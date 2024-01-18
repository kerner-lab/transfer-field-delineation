import os
from pyunpack import Archive
import geopandas as gpd

Archive('france_data.7z').extractall("./france/")