import numpy as np

from app.common.minimodel import CloudCover, Conditions, ImageQuality, SkyBackground, WaterVapor


def test_most_restrictive_conditions1():
    """
    Test an empty set of Conditions to make sure that it returns the least
    restrictive conditions.
    """
    mrc = Conditions.most_restrictive_conditions(())
    c = Conditions(cc=CloudCover.CCANY, iq=ImageQuality.IQANY, sb=SkyBackground.SBANY, wv=WaterVapor.WVANY)
    assert mrc == c


def test_most_restrictive_conditions2():
    """
    Test a mixture of various conditions.
    """
    cc1 = Conditions(cc=np.array([CloudCover.CC70, CloudCover.CC80]),
                     iq=np.array([ImageQuality.IQ85, ImageQuality.IQANY]),
                     sb=np.array([SkyBackground.SB50, SkyBackground.SBANY]),
                     wv=np.array([WaterVapor.WV80, WaterVapor.WV80]))
    cc2 = Conditions(cc=CloudCover.CCANY,
                     iq=ImageQuality.IQANY,
                     sb=SkyBackground.SB80,
                     wv=WaterVapor.WV20)

    mrc = Conditions.most_restrictive_conditions((cc1, cc2))
    exp = Conditions(cc=CloudCover.CC70, iq=ImageQuality.IQ85, sb=SkyBackground.SB50, wv=WaterVapor.WV20)
    assert mrc == exp


# def test_most_restrictive_conditions3():
#     """
#     np.asarray causes problems due to 0-dim array.
#     """
#     cc1 = Conditions(cc=np.asarray(CloudCover.CCANY),
#                      iq=np.asarray(ImageQuality.IQANY),
#                      sb=np.asarray(SkyBackground.SB80),
#                      wv=np.asarray(WaterVapor.WV20))
#     cc2 = Conditions(cc=np.array([CloudCover.CC70, CloudCover.CC80]),
#                      iq=np.array([ImageQuality.IQ85, ImageQuality.IQANY]),
#                      sb=np.array([SkyBackground.SB50, SkyBackground.SBANY]),
#                      wv=np.array([WaterVapor.WV80, WaterVapor.WV80]))
#
#     mrc = Conditions.most_restrictive_conditions((cc1, cc2))
#     exp = Conditions(cc=CloudCover.CC70, iq=ImageQuality.IQ85, sb=SkyBackground.SB50, wv=WaterVapor.WV20)
#     assert mrc == exp


def test_most_restrictive_conditions4():
    cc1 = Conditions(cc=CloudCover.CC70, iq=ImageQuality.IQ70, sb=SkyBackground.SBANY, wv=WaterVapor.WV80)
    mrc = Conditions.most_restrictive_conditions((cc1,))
    assert mrc == cc1
