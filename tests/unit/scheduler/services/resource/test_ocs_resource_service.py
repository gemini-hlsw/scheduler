# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import pytest

from datetime import date, timedelta

from lucupy.minimodel.site import Site
from lucupy.resource_manager import ResourceManager

from scheduler.services.resource import OcsResourceService


# Sets are notoriously difficult to compare since the order changes drastically if there is a small difference.
# Because of this, we sort them.
@pytest.fixture
def year():
    return timedelta(days=365)


def test_specific_date_gn():
    rm = ResourceManager()
    expected = sorted([
        Site.GN.resource,
        rm.lookup_resource(rid='GCAL'),
        rm.lookup_resource(rid='Altair'),
        rm.lookup_resource(rid='NIRI'),
        rm.lookup_resource(rid='GMOS OIWFS'),
        rm.lookup_resource(rid='PWFS1'),
        rm.lookup_resource(rid='GMOS-N'),
        rm.lookup_resource(rid='PWFS2'),
        rm.lookup_resource(rid='GNIRS'),
        rm.lookup_resource(rid='10000007'),
        rm.lookup_resource(rid='11100301'),
        rm.lookup_resource(rid='11201501'),
        rm.lookup_resource(rid='11004601'),
        rm.lookup_resource(rid='Mirror'),
        rm.lookup_resource(rid='R400'),
        rm.lookup_resource(rid='B1200'),
        rm.lookup_resource(rid='R150'),
        rm.lookup_resource(rid='10005360'),
        rm.lookup_resource(rid='10005351'),
        rm.lookup_resource(rid='10005352'),
        rm.lookup_resource(rid='10005353'),
        rm.lookup_resource(rid='10005354'),
        rm.lookup_resource(rid='10005355'),
        rm.lookup_resource(rid='10005356'),
        rm.lookup_resource(rid='10005357'),
        rm.lookup_resource(rid='10005367'),
        rm.lookup_resource(rid='10005369'),
        rm.lookup_resource(rid='11002801'),
        rm.lookup_resource(rid='11200499'),
        rm.lookup_resource(rid='11200403'),
        rm.lookup_resource(rid='11200404'),
        rm.lookup_resource(rid='11200402'),
        rm.lookup_resource(rid='11200401'),
        rm.lookup_resource(rid='11090301')], key=lambda x: x.id)
    resources = sorted(list(OcsResourceService().get_resources(Site.GN, date(year=2018, month=11, day=8))),
                       key=lambda x: x.id)
    assert expected == resources


def test_specific_date_gs():
    rm = ResourceManager()
    expected = sorted([
        Site.GS.resource,
        rm.lookup_resource(rid='GCAL'),
        rm.lookup_resource(rid='B600'),
        rm.lookup_resource(rid='R150'),
        rm.lookup_resource(rid='R400'),
        rm.lookup_resource(rid='Canopus'),
        rm.lookup_resource(rid='Flamingos2'),
        rm.lookup_resource(rid='GMOS-S'),
        rm.lookup_resource(rid='GPI'),
        rm.lookup_resource(rid='Mirror'),
        rm.lookup_resource(rid='PWFS1'),
        rm.lookup_resource(rid='PWFS2'),
        rm.lookup_resource(rid='GMOS OIWFS'),
        rm.lookup_resource(rid='FII OIWFS'),
        rm.lookup_resource(rid='10000005'),
        rm.lookup_resource(rid='10000009'),
        rm.lookup_resource(rid='10005372'),
        rm.lookup_resource(rid='10005373'),
        rm.lookup_resource(rid='10005374'),
        rm.lookup_resource(rid='10005376'),
        rm.lookup_resource(rid='10005377'),
        rm.lookup_resource(rid='10005381'),
        rm.lookup_resource(rid='10005390'),
        rm.lookup_resource(rid='11005902'),
        rm.lookup_resource(rid='11013104'),
        rm.lookup_resource(rid='11020601'),
        rm.lookup_resource(rid='11021602'),
        rm.lookup_resource(rid='11022001'),
        rm.lookup_resource(rid='11023303'),
        rm.lookup_resource(rid='11023304'),
        rm.lookup_resource(rid='11023331'),
        rm.lookup_resource(rid='11023332'),
        rm.lookup_resource(rid='11023335'),
        rm.lookup_resource(rid='11023336')], key=lambda x: x.id)
    resources = sorted(list(OcsResourceService().get_resources(Site.GS, date(year=2018, month=12, day=30))),
                       key=lambda x: x.id)
    assert expected == resources


def test_early_date(year):
    earliest_date = OcsResourceService().date_range_for_site(Site.GN)[0]
    expected = frozenset()
    resources = OcsResourceService().get_resources(Site.GN, earliest_date - year)
    assert expected == resources


def test_late_date(year):
    latest_date = OcsResourceService().date_range_for_site(Site.GN)[1]
    expected = frozenset()
    resources = OcsResourceService().get_resources(Site.GN, latest_date + year)
    assert expected == resources
