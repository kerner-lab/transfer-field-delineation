import requests
from gee_helpers import order_imagery
import shapely.geometry
from shapely.geometry import MultiPolygon, Polygon
from requests.auth import HTTPBasicAuth

geojson = {
  	"type": "Polygon",
  	"coordinates": [
           [
            [
              20.517654418945312,
              -34.00087355538849
            ],
            [
              20.523147583007812,
              -34.2004447595411
            ],
            [
              20.789566040039062,
              -34.2004447595411
            ],
            [
              20.77789306640625,
              -33.99233439016673
            ],
            [
              20.517654418945312,
              -34.00087355538849
            ]
          ]
    	]
	}
def getPolygon():
	
	poly = Polygon([tuple(l) for l in geo['coordinates'][0]])
	polygons = [poly]
	return shapely.geometry.MultiPolygon(polygons)
	
def bbox(long0, lat0, lat1, long1):
        return Polygon([[long0, lat0],
                        [long1,lat0],
                        [long1,lat1],
                        [long0, lat1]])


if __name__ == "__main__":
	PLANET_API_KEY = os.getenv('PL_API_KEY')
	session = requests.Session()
	session.auth = HTTPBasicAuth(PLANET_API_KEY, '')
	#poly = bbox(-0.89,46.08,47.33,0.77)
	#polygons = [poly]
	order_imagery("test order","ee-2019","ps2-southafrica",geojson,'2018-07-01','2018-09-30',session)