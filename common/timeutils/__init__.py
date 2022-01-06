# Time and coordinate-related python libraries
# Bryan Miller

from __future__ import print_function
import sys


def sex2dec(stime, todegree=False, sep=':'):
    # stime is a string of format "HR:MIN:SEC"
    # returns the decimal equivalent
    # Bryan Miller
    # From sex2dec.pro

    l_stime = str(stime).replace("+", "")
    if sep not in l_stime:
        print('Separator not found. Input must be in the format "HR<sep>MIN<sep>SEC"')
        return -1

    f = 1.0
    if todegree:
        f = 15.0

    result = 0.0
    exp = 0
    sign = 1.
    for val in l_stime.split(sep):
        # tmp = val.lstrip('0')
        # if tmp == '':
        #    tmp = '0'
        x = float(val)
        if x < 0.0:
            sign = -1.
        result = result + abs(x) / 60. ** exp
        exp = exp + 1
    return sign * f * result


def dtsex2dec(datetime, todegree=False):
    # input is a datetime object
    # Bryan Miller

    f = 1.0
    if todegree:
        f = 15.0

    sign = 1.
    if datetime.hour < 0:
        sign = -1.
    return sign * f * (abs(datetime.hour) + datetime.minute / 60. + datetime.second / 3600.)


def sixty(dd):
    # http://stackoverflow.com/questions/2579535/how-to-convert-dd-to-dms-in-python
    # Equivalent to sixty.pro
    # Bryan Miller
    is_positive = dd >= 0
    l_dd = abs(dd)
    minutes, seconds = divmod(l_dd * 3600, 60)
    degrees, minutes = divmod(minutes, 60)
    if degrees > 0.:
        degrees = degrees if is_positive else -degrees
    elif minutes > 0.:
        minutes = minutes if is_positive else -minutes
    else:
        seconds = seconds if is_positive else -seconds
    return (degrees, minutes, seconds)


def dec2sex(d, p=3, cutsec=False, hour=False, tohour=False, sep=':', leadzero=0, round=False):
    """
    Convert decimal degrees/hours to a formatted sexigesimal string
    From dec2sex.pro
    Bryan Miller

    Parameters
    :param d: input in degrees
    :param p: digits for seconds
    :param cutsec: Cut seconds, just display, e.g. DD:MM
    :param hour: d is decimal hours, so must be <=24
    :param tohour: convert from degress to hours (divide by 15.)
    :param sep: Separator string
    :param leadzero: if >0 display leading 0's, e.g. -05:25. The value is the number of digits for the DD or HR field.
    :param round: when cutsec, round to the nearest minute rather than truncate
    :return: string
    """
    l_d = float(d)
    sign = ''
    sd = ''
    dg = 0
    mn = 0
    sc = 0.0
    maxdg = 360.

    if tohour:
        l_d = l_d / 15.0
        hour = True

    if hour and (abs(l_d) > 24.):
        print('Input in hours must be less than or equal to 24.0.')
        sys.exit(1)

    if hour:
        maxdg = 24.0

    n = 2 if p == 0 else 3
    secstr = '{:0' + '{:1d}'.format(n + p) + '.' + '{:1d}'.format(p) + 'f}'

    six = sixty(l_d)

    dg = abs(int(six[0]))
    if (six[0] < 0):
        sign = '-'
    # if dg >= 100:
    #     ldg = 3
    # elif dg >= 10:
    #     ldg = 2
    # else:
    #     ldg = 1

    if leadzero > 0:
        sldg = '0' + str(leadzero)
    else:
        sldg = str(len(str(dg)))

    mn = abs(int(six[1]))
    if (six[1] < 0):
        sign = '-'

    sc = float(secstr.format(abs(six[2])))
    if (six[2] < 0.0):
        sign = '-'
    #    print sign,dg,mn,sc

    if sc >= 60.0:
        sc -= 60.0
        mn = mn + 1

    if cutsec:
        if round and sc >= 30.:
            # Round to the nearest minute, otherwise truncate
            mn += 1
        sc = 0.0

    if mn >= 60:
        mn -= 60
        dg = dg + 1

    if dg >= int(maxdg):
        dg = dg - int(maxdg)

    if cutsec and sc == 0.0:
        fmt = '{:1s}{:' + sldg + 'd}' + sep + '{:02d}'
        s = fmt.format(sign, dg, mn)
    else:
        fmt = '{:1s}{:' + sldg + 'd}' + sep + '{:02d}' + sep + secstr
        s = fmt.format(sign, dg, mn, sc)

    return s.strip()

