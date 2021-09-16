from time import time
from collector import * 
from selector import *
from resource_mock import Resource

import logging
logging.basicConfig(level=logging.DEBUG,filename=f'{__name__}.log', filemode='w')
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)


if __name__ == '__main__':
    
    time_range = Time(["2019-02-02 08:00:00", "2019-05-01 08:00:00"], format='iso', scale='utc')
    night_date = '2019-01-03'

    # The semesters to consider for the plan
    obs_classes = ['SCIENCE', 'PROG_CAL', 'PARTNER_CAL']
    prog_types = ['Q', 'LP', 'FT', 'DD']
    sites = [Site('gs')]
    semesters = ['2018B','2019A']

    path = './collector/data'
    collector = Collector(sites, semesters, prog_types, obs_classes, time_range=time_range)
    collector.load(path)

    # Resource API mock 
    resource = Resource('/resource_mock/data')
    resource.connect()
    resources = resource.get_night_resources(sites,night_date)

    # ephemid directory; should be file?

    ephem_dir = path + '/ephem'
    if not os.path.exists(ephem_dir):
        os.makedirs(ephem_dir)

    selector = Selector(collector, sites, time_range=time_range)

    actual_conditions = collector.get_actual_conditions()
    
    visits = selector.create_pool()

    for site in sites:
        #check if save visibility exists if not calculate
        selector.visibility(site, ephem_dir=ephem_dir)

    night_period = 0 
    selection = { site: selector.select(visits[site], 
                                        night_period, 
                                        site, 
                                        actual_conditions, 
                                        resources, 
                                        ephem_dir) for site in sites}
    selector.selection_summary()