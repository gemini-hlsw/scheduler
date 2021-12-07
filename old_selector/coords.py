#!/usr/bin/env python3

# 2014 Dec 20 - Andrew W Stephens
# 2015 Nov 18 - AWS, Parse now returns a sign.  dms2deg now requires a sign

#-----------------------------------------------------------------------

import logging
import numpy

#-----------------------------------------------------------------------

def rad2deg(rad):
    return rad / numpy.pi * 180.

#-----------------------------------------------------------------------

def arcsec2rad(arcsec):
    return arcsec / 3600. / 180. * numpy.pi

#-----------------------------------------------------------------------

def rad2arcsec(rad):
    return rad / numpy.pi * 180. * 3600.

#-----------------------------------------------------------------------

def deg2rad(deg):
    return deg / 180. * numpy.pi

#-----------------------------------------------------------------------

def hrs2rad(hrs):
    return hrs / 12. * numpy.pi

#-----------------------------------------------------------------------

def dms2deg(dms):
    logger = logging.getLogger('coords.dms2deg')

    if dms is None:
        return None

    d = float(dms[0])
    m = float(dms[1])
    s = float(dms[2])
    sign = dms[3]

    #logger.debug('%s %s %s %s', sign, d, m, s)

    dd = d + m/60. + s/3600.

    if sign == '-':
        dd *= -1.
        
    logger.debug('%s deg -> %s deg', dms, dd)

    return dd

#-----------------------------------------------------------------------

def dms2rad(dms):
    logger = logging.getLogger('coords.dms2rad')

    if dms is None:
        return None
    
    dd = dms2deg(dms)
    rad = dd * numpy.pi / 180.
    logger.debug('%s deg -> %s rad', dd, rad)
    return rad

#-----------------------------------------------------------------------
# Convert from HMS to radians

def hms2rad(hms):
    logger = logging.getLogger('coords.hms2rad')

    if hms is None:
        return None

    h = float(hms[0])
    m = float(hms[1])
    s = float(hms[2])
    hours = h + m/60. + s/3600.
    rad = hours * numpy.pi / 12.

    logger.debug('%s hrs -> %s rad', hms, rad)

    return rad

#-----------------------------------------------------------------------

def hms2deg(hms):
    rad = hms2rad(hms)
    return rad / numpy.pi * 180.

#-----------------------------------------------------------------------
# Convert from radians to DMS

def rad2dms(rad):

    if rad is None:
        return (None, None, None, None)

    deg = rad * 180. / numpy.pi
    dec = abs(deg)
    d = int(dec)
    tmp = (dec - d) * 60.
    m = int(tmp)
    s = (tmp - m) * 60.

    if deg < 0:
        sign = '-'
    else:
        sign = '+'

    return d,m,s,sign

#-----------------------------------------------------------------------

def deg2dms(deg):

    if deg is None:
        return (None, None, None, None)

    dec = abs(deg)
    d = int(dec)
    tmp = (dec - d) * 60.
    m = int(tmp)
    s = (tmp - m) * 60.
    if deg < 0:
        sign = '-'
    else:
        sign = '+'

    return d,m,s,sign

#-----------------------------------------------------------------------
# Convert from hours to HMS

def hrs2hms(hrs):
    logger = logging.getLogger('coords.hrs2hms')

    if hrs is None:
        return (None, None, None)
 
    h = int(hrs)
    tmp = (hrs - h) * 60.
    m = int(tmp)
    s = (tmp - m) * 60.

    logger.debug('%s hrs -> %s, %s, %s', hrs, h,m,s)
    
    return h,m,s

#-----------------------------------------------------------------------
# Convert from radians to HMS

def rad2hms(rad):

    if rad is None:
        return (None, None, None)
    
    hours = rad * 12. / numpy.pi
    ra = abs(hours)
    h = int(ra)
    tmp = (ra - h) * 60.
    m = int(tmp)
    s = (tmp - m) * 60.
    return  h,m,s

#-----------------------------------------------------------------------

def deg2hms(deg):

    if deg is None:
        return (None, None, None)
    
    rad = deg / 180. * numpy.pi
    h,m,s = rad2hms(rad)
    return h,m,s

#-----------------------------------------------------------------------
# Calculate the angular separation between two points whose coordinates are given in RA and Dec
# From angsep.py Written by Enno Middelberg 2001
# http://www.stsci.edu/~ferguson/software/pygoodsdist/pygoods/angsep.py

def angsep(ra1rad,dec1rad,ra2rad,dec2rad):
    logger = logging.getLogger('coords.angsep')

    """ Determine separation in degrees between two celestial objects.
        Arguments are RA and Dec in radians.  Output in radians.
    """

    if abs(ra1rad - ra2rad) < 1e-8 and abs(dec1rad - dec2rad) < 1e-8: # to avoid arccos errors
        logger.debug('Skipping arccos')
        sep = numpy.sqrt( (numpy.cos((dec1rad+dec2rad)/2.)*(ra1rad-ra2rad))**2 + (dec1rad-dec2rad)**2 )

    else:
        # calculate scalar product for determination of angular separation
        x=numpy.cos(ra1rad)*numpy.cos(dec1rad)*numpy.cos(ra2rad)*numpy.cos(dec2rad)
        y=numpy.sin(ra1rad)*numpy.cos(dec1rad)*numpy.sin(ra2rad)*numpy.cos(dec2rad)
        z=numpy.sin(dec1rad)*numpy.sin(dec2rad)
        rad=numpy.arccos(x+y+z) # Sometimes gives warnings when coords match
        #logger.debug('rad = %s', rad)

        if rad < 0.000004848: # Use Pythargoras approximation if rad < 1 arcsec (= 4.8e-6 radians)
            logger.debug('Using Pythargoras approximation for small angles')
            sep = numpy.sqrt( (numpy.cos((dec1rad+dec2rad)/2.)*(ra1rad-ra2rad))**2 + (dec1rad-dec2rad)**2 )
        else:
            sep = rad

    logger.debug('Separation = %s radians = %s arcsec', sep, rad2arcsec(sep))

    return sep

#-----------------------------------------------------------------------
# Calculate the angular separation between two points whose coordinates are given in RA and Dec
# From Astronomical Algorithms by Jean Meeus, Chapter 16
# cos(separation) = sin(d1)sin(d2) + cos(d1)cos(d2)cos(ra1-ra2)

def angsep2(ra1rad,dec1rad,ra2rad,dec2rad):
    logger = logging.getLogger('coords.angsep')

    """ Determine separation in degrees between two celestial objects.
        Arguments are RA and Dec in radians.  Output in radians.
    """

    t1 = numpy.sin(dec1rad) * numpy.sin(dec2rad)
    t2 = numpy.cos(dec1rad) * numpy.cos(dec2rad) * numpy.cos(ra1rad - ra2rad)
    sep = numpy.arccos(t1 + t2)
    logger.debug('sep = %s', sep)

    if sep < 0.0029: # (10 arcmin) then use the Pythargoras approximation
        logger.debug('Using Pythargoras approximation for small angles')
        sep = numpy.sqrt( ((ra1rad-ra2rad) * numpy.cos((dec1rad+dec2rad)/2.))**2 + (dec1rad-dec2rad)**2 )

    logger.debug('Separation = %s radians', sep)

    return sep

# --------------------------------------------------------------------------------------------------
# Calculate the angular separation between two points whose coordinates are given in RA and Dec
# From Astronomical Algorithms by Jean Meeus, Chapter 16
# cos(separation) = sin(d1)sin(d2) + cos(d1)cos(d2)cos(ra1-ra2)
# This is the same as angsep2, but converted to handle input numpy arrays

def angsep3(ra1rad,dec1rad,ra2rad,dec2rad):
    logger = logging.getLogger('coords.angsep')

    """ Determine separation in degrees between two celestial objects.
        Arguments are RA and Dec in radians.  Output in radians.
        REQUIRES numpy arrays.
    """

    t1 = numpy.sin(dec1rad) * numpy.sin(dec2rad)
    t2 = numpy.cos(dec1rad) * numpy.cos(dec2rad) * numpy.cos(ra1rad - ra2rad)
    sep = numpy.arccos(t1 + t2)

    # If < 10 arcmin use the Pythargoras approximation:
    sep[sep<0.0029] = \
            numpy.sqrt( ((ra1rad[sep<0.0029]-ra2rad[sep<0.0029]) *
                         numpy.cos((dec1rad[sep<0.0029]+dec2rad[sep<0.0029])/2.))**2 +
                        (dec1rad[sep<0.0029]-dec2rad[sep<0.0029])**2 )

    logger.debug('Separation = %s radians', sep)

    return sep

#-----------------------------------------------------------------------
# Calculate the angular separation between two points whose coordinates are given in RA and Dec (radians)
# For SMALL separations only (<10 arcmin)

def smallangsep(ra1rad, dec1rad, ra2rad, dec2rad):
    logger = logging.getLogger('coords.smallangsep')
    sep = numpy.sqrt( ((ra1rad-ra2rad) * numpy.cos((dec1rad+dec2rad)/2.))**2 + (dec1rad-dec2rad)**2 )
    logger.debug('Separation = %s radians', sep)
    return sep

# --------------------------------------------------------------------------------------------------
# Calculate the angular separation between two nearby (<10 arcmin) coordinates in degrees

def smallangsepdeg(ra1deg, dec1deg, ra2deg, dec2deg):
    logger = logging.getLogger('coords.smallangsepdeg')
    if ra1deg is None or dec1deg is None or ra2deg is None or dec2deg is None:
        logger.error('None value')
        return None
    sep = rad2deg(smallangsep(deg2rad(ra1deg), deg2rad(dec1deg), deg2rad(ra2deg), deg2rad(dec2deg)))
    logger.debug('Separation = %s degrees', sep)
    return sep

# --------------------------------------------------------------------------------------------------
# Parse a coordinate triple, e.g. "HH:MM:SS.s" or "-DD MM SS.s"

def Parse(coordstring):
    logger = logging.getLogger('coords.Parse')

    logger.debug('Parsing %s', coordstring)

    if coordstring is None:
        return (None, None, None)

    if ':' in coordstring:
        c = coordstring.split(':')
    else:
        c = coordstring.split()

    if c[0][0] == '-':
        s  = '-'
    else:
        s = '+'
    
    c1 = abs(int(c[0]))
    c2 = int(c[1])
    c3 = float(c[2])

    logger.debug('%s -> %s, %s, %s, %s', coordstring, s, c1, c2, c3)

    # Return the sign last so that it may be ignored for HMS triples:
    return c1, c2, c3, s

#-----------------------------------------------------------------------

def hmsstr2deg(coordstring):
    h,m,s,sign = Parse(coordstring)
    return hms2deg([h,m,s])

#-----------------------------------------------------------------------

def dmsstr2deg(coordstring):
    d,m,s,sign = Parse(coordstring)
    return dms2deg([d,m,s,sign])

#-----------------------------------------------------------------------

def hmsstr2rad(coordstring):
    h,m,s,sign = Parse(coordstring)
    return hms2rad([h,m,s])

#-----------------------------------------------------------------------

def dmsstr2rad(coordstring):
    d,m,s,sign = Parse(coordstring)
    return dms2rad([d,m,s,sign])

#-----------------------------------------------------------------------

def deg2dmsstr(deg):
    d,m,s,sign = deg2dms(deg)
    return dms2str(d,m,s,sign)

#-----------------------------------------------------------------------

def deg2str(deg):
    return deg2dmsstr(deg)

#-----------------------------------------------------------------------

def deg2hmsstr(deg):
    h,m,s = deg2hms(deg)
    return hms2str(h,m,s)

#-----------------------------------------------------------------------

def dms2str(d,m,s,sign):
    if d is None and m is None and s is None:
        return 'None'
    return '%1s%02i:%02i:%05.2f' % (sign,d,m,s)

#-----------------------------------------------------------------------

def hrs2str(hrs):
    h,m,s = hrs2hms(hrs)
    return hms2str(h,m,s)

#-----------------------------------------------------------------------

def hms2str(h,m,s):

    if h is None and m is None and s is None:
        return 'None'

    return '%2i:%02i:%06.3f' % (h,m,s)

#-----------------------------------------------------------------------
