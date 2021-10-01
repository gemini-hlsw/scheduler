from collector import *
from selector import *
from resource_mock import Resource

import sys
import logging

# *** FILE LOGGING ***
# logging.basicConfig(level=logging.DEBUG, filename=f'{__name__}.log', filemode='w')
# logger = logging.getLogger(__name__)
# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(logging.INFO)
# logger.addHandler(stream_handler)

# *** CONSOLE LOGGING ***
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

if __name__ == '__main__':
    time_range = Time(["2019-02-02 08:00:00", "2019-05-01 08:00:00"], format='iso', scale='utc')
    night_date = '2019-02-01'

    # The observation classes, program types, sites, and semesters to consider for the plan.
    obs_classes = ['SCIENCE', 'PROG_CAL', 'PARTNER_CAL']
    prog_types = ['Q', 'LP', 'FT', 'DD']
    sites = [Site.GS, Site.GN]
    semesters = ['2018B', '2019A']

    path = os.path.join('collector', 'data')
    collector = Collector(sites, semesters, prog_types, obs_classes, time_range=time_range)
    collector.load(path)

    # Resource API mock
    resource_path = os.path.join('resource_mock', 'data')
    resource = Resource(resource_path)
    resource.connect()
    resources = resource.get_night_resources(sites, night_date)

    # Ephemerides path
    ephem_path = os.path.join(path, 'ephem')
    if not os.path.exists(ephem_path):
        os.makedirs(ephem_path)

    selector = Selector(collector, sites, time_range=time_range)

    actual_conditions = collector.get_actual_conditions()
    
    visits = selector.create_pool()

    for site in sites:
        # Check if save visibility exists: if not, calculate
        selector.visibility(site, ephem_path=ephem_path)

    night_period = 0 

    selection = {site: selector.select(visits[site],
                                       night_period,
                                       site,
                                       actual_conditions,
                                       resources,
                                       ephem_path) for site in sites}
    selector.selection_summary()
