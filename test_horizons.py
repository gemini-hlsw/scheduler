from horizons import HorizonsClient
from common.minimodel import Site, NonsiderealTarget, TargetTag, TargetType
from datetime import datetime, timedelta

if __name__ == '__main__':

    site = Site.GS
    today = datetime.today()
    client = HorizonsClient(Site.GS, airmass=300)
    target = NonsiderealTarget('Beer', None, type=TargetType.BASE, tag=TargetTag.COMET, des='1971 UC1', ra=None, dec=None)
    eph = client.get_ephemerides(target,start= today-timedelta(days=1), end= today)
    target.ra = eph.ra
    target.dec = eph.dec
    print(target)
