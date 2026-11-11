# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""A resource first seen without a name gets the one a later lookup gives it, so the UI can show it."""

from lucupy.minimodel import ResourceType

from scheduler.services.resource.resource_service import ResourceService

# Unlikely to clash with anything another test put in lucupy's singleton cache.
_BARCODE = '10999999'


def test_a_later_description_fills_in_a_missing_one():
    # As the availability files load it: a bare barcode.
    bare = ResourceService.lookup_resource(_BARCODE, resource_type=ResourceType.FPU)
    assert bare.description is None

    # As the MDF lookup of an observation does.
    named = ResourceService.lookup_resource(_BARCODE, description='GN2026BQ999-01', resource_type=ResourceType.FPU)

    assert named.description == 'GN2026BQ999-01'
    assert named.type == ResourceType.FPU
    # Still the same resource for availability checks.
    assert named == bare and hash(named) == hash(bare)
    assert named in frozenset({bare})
    assert ResourceService.lookup_resource(_BARCODE).description == 'GN2026BQ999-01'


def test_an_existing_description_is_kept():
    ResourceService.lookup_resource(_BARCODE + '1', description='first', resource_type=ResourceType.FPU)

    assert ResourceService.lookup_resource(_BARCODE + '1', description='second').description == 'first'
