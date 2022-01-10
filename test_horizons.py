from horizons import horizons_session, HorizonsClient
from common.minimodel import Site, NonsiderealTarget, TargetTag, TargetType
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar


if __name__ == '__main__':

    site = Site.GS
    today = datetime.today()
    target = NonsiderealTarget('Jupiter', None, type=TargetType.BASE,
                                tag=TargetTag.MAJOR_BODY, des='jupiter', ra=None, dec=None)

    program_start = datetime(2019, 2, 1)
    program_end = datetime(2019, 7, 31)
    fuzzy_boundary = relativedelta(months=2)
    cal = calendar.Calendar()
    
    for month in range(program_start.month, (program_end+fuzzy_boundary).month+1):
        for date in cal.itermonthdates(program_start.year, month):
            if date.month >= program_start.month and date.month <= (program_end+fuzzy_boundary).month:
                
                with horizons_session(site, date, date+timedelta(days=1), 300) as session:
                    eph = session.get_ephemerides(target, date)
                    target.ra = eph.ra
                    target.dec = eph.dec
                    print(target)

                    ra, dec = HorizonsClient.interpolate_ephemeris(eph, datetime(2019, 2, 1, 10, 10, 10, 10))
                    print(ra)
                    input() # remove input to test the whole semester
