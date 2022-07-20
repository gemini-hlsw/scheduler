import pytest
from common.helpers import mask_to_barcode


@pytest.mark.parametrize('mask, inst, expected', [('GS2017BLP005-34', 'GMOS', '11200534'),
                                                  ('0.75arcsec', None, '10005373')])
def test_mask_to_barcode(mask, inst, expected):
    assert mask_to_barcode(mask, inst) == expected
