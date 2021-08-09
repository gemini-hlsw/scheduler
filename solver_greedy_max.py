from time import time
from collector import * 
#from selector import *



if __name__ == '__main__':
    
    time_range = Time(["2019-02-02 08:00:00", "2019-05-01 08:00:00"], format='iso', scale='utc')

    # The semesters to consider for the plan
    obs_classes = ['SCIENCE', 'PROG_CAL', 'PARTNER_CAL']
    prog_types = ['Q', 'LP', 'FT', 'DD']
    sites = ['gs']
    semesters = ['2018B','2019A']

    path = './collector/data'
    collector = Collector(sites, semesters, prog_types, obs_classes, time_range=time_range)
    collector.load(path)

    # ephemid directory 

    ephem_dir = path + '/ephem/'
    if not os.path.exists(ephem_dir):
        os.makedirs(ephem_dir)

    #selector = Selector(collector,sites,)