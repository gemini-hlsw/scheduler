# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import pytest

from datetime import date

from lucupy.minimodel.resource import Resource
from lucupy.minimodel.site import Site
from mock.resource import ResourceMock


@pytest.fixture
def rm():
    return ResourceMock()


def test_specific_date_gs():
    expected = frozenset(
        [Resource(id='10000009'), Resource(id='10000705'), Resource(id='10003601'), Resource(id='10003901'),
         Resource(id='10005372'), Resource(id='10005373'), Resource(id='10005374'), Resource(id='10005375'),
         Resource(id='10005376'), Resource(id='10005377'), Resource(id='10005381'), Resource(id='10005388'),
         Resource(id='10005390'), Resource(id='10008001'), Resource(id='10200546'), Resource(id='10200547'),
         Resource(id='10300901'), Resource(id='10900201'),
         Resource(id='B600'), Resource(id='Mirror'), Resource(id='R400'), Resource(id='R831')])
    resources = rm().get_resources(Site.GS, date(year=2017, month=5, day=7))
    assert resources == expected

