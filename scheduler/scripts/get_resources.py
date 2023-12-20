# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import date

from lucupy.minimodel import Site

from scheduler.services.resource import OcsResourceService

# For Bryan and Kristin: testing instructions
if __name__ == '__main__':
    # To get the Resources for a specific site on a specific local date, modify the following.
    site = Site.GN
    day = date(year=2018, month=11, day=8)

    resources_available = OcsResourceService().get_resources(site, day)

    print(f'*** Resources for site {site.name} for {day} ***')
    for resource in sorted(resources_available, key=lambda x: x.id):
        print(resource)
