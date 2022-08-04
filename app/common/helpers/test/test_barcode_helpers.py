import pytest
from app.common.helpers import mask_to_barcode, barcode_to_mask


@pytest.mark.parametrize('mask, inst, expected', [('GS2017BLP005-34', 'GMOS', '11200534'),
                                                  ('0.75arcsec', None, '10005373')])
def test_mask_to_barcode(mask, inst, expected):
    assert mask_to_barcode(mask, inst) == expected

@pytest.mark.parametrize('barcode, rootname, expected', [('10005381','GS2017', 'PinholeC'),
                                                         ('11310101', 'GS2017', 'GS2017BFT101-01'),])
def test_barcode_to_mask(barcode, rootname, expected):
    assert barcode_to_mask(barcode, rootname) == expected

