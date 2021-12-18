import requests
import sys
import shapely.geometry
from shapely.geometry import MultiPolygon
from datetime import datetime, date, timedelta
from datetime import timezone
from time import mktime
from planet.api.utils import strp_lenient
from typing import List, Generator, Tuple

# PLANET ORDERING


def time2utc(st: str) -> str:
    '''Helper function to convert to utc'''
    st_time = strp_lenient(st)
    if st_time is not None:
        dt_ts = datetime.fromtimestamp(time2epoch(st_time), tz=timezone.utc)
        return dt_ts.isoformat().replace('+00:00', 'Z')
    else:
        sys.exit('Could not parse time {}: check and retry'.format(st))


def time2epoch(st: datetime) -> float:
    '''Helper function to convert to epoch time'''
    str_time = datetime.strptime(st.isoformat(), '%Y-%m-%dT%H:%M:%S')
    str_tuple = str_time.timetuple()
    epoch_time = mktime(str_tuple)
    return epoch_time


def search_payload(geojson: str, start: str, end: str) -> dict:
    '''Creates the planet search request json object'''
    geometry_filter = {
      'type': 'GeometryFilter',
      'field_name': 'geometry',
      'config': geojson
    }

    date_range_filter = {
      'type': 'DateRangeFilter',
      'field_name': 'acquired',
      'config': {
        'gte': time2utc(start),
        'lte': time2utc(end)
      }
    }

    asset_filter = {
        'type': 'AndFilter',
        'config': [{
            'type': 'AssetFilter',
            'config': ['analytic_sr']
        }, {
            'type': 'AssetFilter',
            'config': ['udm2']
        }]
    }

    combined_filter = {
      'type': 'AndFilter',
      'config': [geometry_filter, date_range_filter, asset_filter]
    }

    search_request = {
      'item_types': ['PSScene4Band'],
      'filter': combined_filter
    }

    return search_request


def yield_features(url: str, sess: requests.Session,
                   payload: dict) -> Generator[dict, None, None]:
    '''Helper to create a generator from matching planet imagery'''
    page = sess.post(url, json=payload)
    for feature in page.json()['features']:
        yield feature
    while True:
        url = page.json()['_links']['_next']
        page = sess.get(url)

        for feature in page.json()['features']:
            yield feature

        if page.json()['_links'].get('_next') is None:
            break


def get_image_ids(geom: MultiPolygon, start: str, end: str,
                  sess: requests.Session) -> List[str]:
    '''Searches planet for all imagery in AOI'''
    search_json = search_payload(geom, start, end)
    all_features = list(
        yield_features('https://api.planet.com/data/v1/quick-search', sess,
                       search_json))

    image_ids = [x['id'] for x in all_features]
    print(f'Processed a total of {len(image_ids)}')

    return image_ids


def order_payload(name: str, image_ids: List[str], geojson: str,
                  project: str, collection: str) -> dict:
    order_request = {
		'name': name,
        'order_type': 'partial',
        'products': [{
            'item_ids': image_ids,
            'item_type': 'PSScene4Band',
            'product_bundle': 'analytic_sr_udm2'
        }],
        'tools': [{
            'clip': {
                'aoi': geojson
            }
        }],
        'delivery': {
            'google_earth_engine': {
                'project': project,
                'collection': collection
            }
        }
    }
    return order_request


def place_order(payload: dict, order_size: int, sess: requests.Session):
    orders_url = 'https://api.planet.com/compute/ops/orders/v2'
    response = sess.post(orders_url, json=payload)
    if response.status_code == 202:
        order_id = response.json()['id']
        url = f'https://api.planet.com/compute/ops/orders/v2/{order_id}'
        feature_check = sess.get(url)
        if feature_check.status_code == 200:
            num_accepted = len(feature_check.json()['products'][0]['item_ids'])
            print(f'Submitted a total of {order_size} image ids: accepted '
                  f'a total of {num_accepted} ids')
            print(f'Order URL: {orders_url}/{order_id}')
    else:
        try:
            error_text = response.json()['general'][0]['message']
        except Exception:
            error_text = 'No error message'
        print(f'Failed with exception code : {response.status_code} '
              f'{error_text}')


def order_imagery(name: str, project: str, collection: str, geom: MultiPolygon,
                  start: str, end: str, sess: requests.Session):
    image_ids = get_image_ids(geom, start, end, sess)
    # Max number of assets is 500, so have to chunk into groups of 500
    chunks = [image_ids[i:i+500] for i in range(0, len(image_ids), 500)]
    for chunk in chunks:
        payload = order_payload(name, chunk, geom, project, collection)
        place_order(payload, len(chunk), sess)


# GENERATING IMAGERY


def get_date_ranges(start: str, end: str) -> List[Tuple[str, str]]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    freq = timedelta(weeks=2)
    date_ranges: List[Tuple[date, date]] = []
    while end_date > start_date:
        date_ranges.append((end_date - freq, end_date))
        end_date = end_date - freq
    fmt = '%Y-%m-%d'
    date_strings = [(start.strftime(fmt), end.strftime(fmt)) for start, end
                    in date_ranges]

    return date_strings

