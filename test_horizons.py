from horizons import horizons_session
from common.minimodel import Site, NonsiderealTarget, TargetTag, TargetType
from datetime import datetime, timedelta

if __name__ == '__main__':

    site = Site.GS
    today = datetime.today()
    target = NonsiderealTarget('Beer', None, type=TargetType.BASE, 
                                tag=TargetTag.COMET, des='1971 UC1', ra=None, dec=None)


    with horizons_session(site, today-timedelta(days=1), today, 300) as session:
        eph = session.get_ephemerides(target)
        target.ra = eph.ra
        target.dec = eph.dec
        print(target)
