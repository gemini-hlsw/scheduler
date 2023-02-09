# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import pytest

from datetime import date, timedelta

from lucupy.minimodel.resource import Resource
from lucupy.minimodel.site import Site
from scheduler.services.resource import OcsResourceService


@pytest.fixture
def year():
    return timedelta(days=365)


def test_specific_date_gn():
    expected = frozenset(
        [Resource(id='10000007'), Resource(id='10005351'), Resource(id='10005352'), Resource(id='10005353'),
         Resource(id='10005354'), Resource(id='10005355'), Resource(id='10005356'), Resource(id='10005357'),
         Resource(id='10005360'), Resource(id='10005367'), Resource(id='10005369'), Resource(id='11002801'),
         Resource(id='11004601'), Resource(id='11090301'), Resource(id='11100301'), Resource(id='11200401'),
         Resource(id='11200402'), Resource(id='11200403'), Resource(id='11200404'), Resource(id='11200499'),
         Resource(id='11201501'), Resource(id='Altair'), Resource(id='B1200'), Resource(id='GCAL'),
         Resource(id='GMOS-N'), Resource(id='GNIRS'), Resource(id='Mirror'), Resource(id='NIRI'),
         Resource(id='R150'), Resource(id='R400')])
    resources = OcsResourceService().get_resources(Site.GN, date(year=2018, month=11, day=8))
    assert resources == expected


def test_specific_date_gs():
    expected = frozenset(
        [Resource(id='10000005'), Resource(id='10000009'), Resource(id='10005372'), Resource(id='10005373'),
         Resource(id='10005374'), Resource(id='10005376'), Resource(id='10005377'), Resource(id='10005381'),
         Resource(id='10005390'), Resource(id='11003908'), Resource(id='11003910'), Resource(id='11003912'),
         Resource(id='11005902'), Resource(id='11021602'), Resource(id='11022001'), Resource(id='11023303'),
         Resource(id='11023304'), Resource(id='11023332'), Resource(id='11023336'), Resource(id='11100101'),
         Resource(id='Mirror'), Resource(id='B600'), Resource(id='R150'), Resource(id='R400'),
         Resource(id='Canopus'), Resource(id='Flamingos2'), Resource(id='GCAL'), Resource(id='GMOS-S'),
         Resource(id='GPI')])
    resources = OcsResourceService().get_resources(Site.GS, date(year=2018, month=12, day=30))
    assert resources == expected


def test_early_date(year):
    earliest_date = OcsResourceService().date_range_for_site(Site.GN)[0]
    expected = frozenset()
    resources = OcsResourceService().get_resources(Site.GN, earliest_date - year)
    assert resources == expected


def test_late_date(year):
    latest_date = OcsResourceService().date_range_for_site(Site.GN)[1]
    expected = frozenset()
    resources = OcsResourceService().get_resources(Site.GN, latest_date + year)
    assert resources == expected
