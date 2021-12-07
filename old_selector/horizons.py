#!/usr/bin/env python3

# Horizons API using URLs

import argparse
import old_selector.coords as coords
import datetime
import dateutil.parser
import glob
import logging
import numpy
#import odb
import os
import requests
import sys

version = '2015-Jun-17' # Andrew Stephens, beta version
version = '2016-Aug-25' # AWS, Coordinates now returns a list of dates & coordinates
version = '2016-Dec-18' # AWS, add Separation query
version = '2017-Jan-13' # AWS, add AngularSize query
version = '2017-Jan-21' # astephens, add AngularSizeLatLong query
version = '2019-Jan-22' # astephens, update to https URL
version = '2019-Sep-17' # astephens, add airmass to AngularSizeLatLong
version = '2020-Oct-20' # bmiller, fix Coordinates parsing, add site parameter argument
version = '2020-Oct-21' # bmiller, add file i/o option to Coordinates
# --------------------------------------------------------------------------------------------------

def main(args):

    #if args.debug:
    #    logger = odb.ConfigureLogging(screenlevel='DEBUG')
    #else:
    #    logger = odb.ConfigureLogging(screenlevel='INFO')
    #logger.debug('odb-new-horizons.py version %s', version)

    #logger.debug('Target = %s', args.target)
   
    horizons = Horizons(args.site, airmass=args.airmass, daytime=args.daytime)

    if 'today' in args.start:
        start = datetime.datetime.today()
    else:
        start = dateutil.parser.parse(args.start)

    if 'tomorrow' in args.end:
        end = datetime.datetime.today() + datetime.timedelta(days=1)
    else:
        end = dateutil.parser.parse(args.end)


    if args.query:
        horizons.Search(args.target)

    elif args.common:
        common = horizons.GetCommonName(args.target)
        print(common)
        
    elif args.coords:
        if args.target == 'asteroids':
            targets = ReadTargetList('new-horizons.asteroids.dat')
            for t in targets:
                #logger.debug('checking %s', t)
                time, ra, dec = horizons.Coordinates(t, start, end)
        else:
            time, ra, dec = horizons.Coordinates(args.target, start, end, step=args.step)

    elif args.ephemeris:
        horizons.Ephemeris(args.target, start, end, step=args.step)

    return

# --------------------------------------------------------------------------------------------------

class Horizons():

    def __init__(self, location, airmass=3, daytime=False):
        logger = logging.getLogger('init')

        self.location = location
        logger.debug('Location = %s', self.location)
        if self.location == 'GN':
            self.center = '568@399'
        elif self.location == 'GS':
            self.center = 'I11@399'
        else:
            logger.error('Unknown location')
            raise SystemExit
        logger.debug('Center = %s', self.center)

        self.airmass = airmass

        if daytime:
            self.skip_day = 'NO'
        else:
            self.skip_day = 'YES'
        logger.debug('Skip Day = %s', self.skip_day)

        self.start = datetime.datetime.today().date()
        self.end = (datetime.datetime.today() + datetime.timedelta(days=1)).date()

        self.senddaterange = False
        self.cal_format = None
        self.obj_data   = None
        self.quantities = None
        self.timestep   = None
        self.csvformat  = 'NO' # This should be changed to YES for simpler parsing

        return
    
    # ----------------------------------------------------------------------------------------------

    def GetCommonName(self, target):
        logger = logging.getLogger('GetCommonName')

        common = None
        self.target = target
        self.make_ephem = 'NO'
        response = self.Query()
        lines = response.text.splitlines()
        for line in lines:
            if 'JPL/HORIZONS' in line:
                logger.debug('The ID is in here: %s', line)
                start = 12 # after JPL/HORIZONS
                end = line.find('(')
                common = line[start:end].lstrip().strip()
                if common == '':
                    common = None
                break

        logger.debug('Common Name = %s', common)
        return common

    # ----------------------------------------------------------------------------------------------
    # Search the Horizons database for a target and return either:
    # the list of results
    # or the number of matches

    def Search(self, target):
        logger = logging.getLogger('Search')

        self.target = target
        self.make_ephem = 'NO'

        response = self.Query()

        # self.ParseHeader(response)

        lines = response.text.splitlines()
        for line in lines:
            print(line)

        #match = []
        #nmatches = 1
        #lines = response.text.splitlines()
        #for line in lines:
        #    logger.debug('%s', line)
        #    if 'JPL/HORIZONS' in line:  # This catches things like: JPL/HORIZONS 1 Ceres
        #        match.append(line)
#
#            if 'Number of matches' in line:
#                nmatches = int(float(line.split()[4]))
#                break
#            elif 'No matches found' in line:
#                matches = 0
#                logger.error('No maches found')
#                raise SystemExit
#                
#        logger.debug('Number of matches = %s', nmatches)
#
#        logger.info('Found %s matches:', len(match))
#        for m in match:
#            print m

        return

    # ----------------------------------------------------------------------------------------------

    def ParseHeader(self, response):
        '''
        Parse the Horizons header.
        '''
        logger = logging.getLogger('ParseHeader')

        lines = response.text.splitlines()

        start = FindSubstringInList(lines,'JPL/HORIZONS')
        logger.debug('Matching "JPL/HORIZONS" start = %s', start)
        if start > -1:
            logger.debug('%s', lines[start:start+1])

        start = FindSubstringInList(lines,'Matching small-bodies')
        logger.debug('Matching "small-bodies" start = %s', start)
        if start > -1:
            end = FindSubstringInList(lines,'To SELECT, enter record # (integer), followed by semi-colon.')
            logger.debug('end = %s', end)
            for i in range(start+2, end-1):
                logger.debug('%s', lines[i])

        start = FindSubstringInList(lines,'Multiple major-bodies match')
        logger.debug('"Multiple major-bodies" start = %s', start)
        if start > -1:
            end = FindSubstringInList(lines,'Use ID# to make unique selection.')
            logger.debug('end = %s', end)
            for i in range(start+2, end-1):
                logger.debug('%s', lines[i])

        start = FindSubstringInList(lines,'No matches found')
        logger.debug('Matching "No matches found" start = %s', start)
        if start > -1:
            logger.error('No matches found')
            return

        return

    # ----------------------------------------------------------------------------------------------
    
    def Coordinates(self, target, start, end, step='1m', file='None', overwrite=False):
        '''
        Query Horizons for the coordinates of a target over a specified range of time.
        Start and End must be datetime objects.
        '''

        if step is None or step == 'None':
            step = '1m'

        logger = logging.getLogger('Coordinates')

        self.target = target
        self.senddaterange = True
        self.start  = start.strftime("'%Y-%b-%d %H:%M'")
        self.end    = end.strftime("'%Y-%b-%d %H:%M'")
        self.cal_format = 'CAL'
        self.make_ephem = 'YES'
        self.obj_data   = 'NO'
        self.quantities = '1'
        self.timestep   = step

        lines = []
        if not overwrite and os.path.exists(file):
            input = ''
            with open(file, 'r') as f:
                input += f.read()
            lines = input.splitlines()
        else:
            response = self.Query()
            lines = response.text.splitlines()
            if file != None:
                with open(file, 'w') as f:
                    f.write(response.text)

        #logger.debug('Lines = %s', lines)
        firstline = lines.index('$$SOE') + 1
        lastline  = lines.index('$$EOE') - 1
        logger.debug('First line = %s', firstline)
        logger.debug('Last line = %s', lastline)
        numlines = lastline - firstline + 1
        logger.debug('Found %s lines of data:', numlines)

        #          1         2         3         4         5         6         7
        #0123456789012345678901234567890123456789012345678901234567890123456789012
        # 2015-Jun-29 20:33 *   15 48 43.7952 -17 51 48.480
        #>..... Daylight Cut-off Requested .....<
        #>..... Airmass Cut-off Requested .....<
        # $$EOE
        # There are a variable number of elements (e.g. the solar and lunar presence columns),
        # so extract using character numbers instead of column numbers:
    	# Shoule I use csv out which might be easier to parse ???

        time = numpy.array([])
        ra   = numpy.array([])
        dec  = numpy.array([])
        for i in range(firstline, lastline + 1):
            d = lines[i]
            logger.debug('Data[%s] = %s', i, d)
            if d and d[7:15] != 'Daylight' and d[7:14] != 'Airmass':
                values = d.split(' ')
                rah  =   int(values[-6])
                ram  =   int(values[-5])
                ras  = float(values[-4])
                decg =       values[-3][0] # sign
                decd =   int(values[-3][1:3])
                decm =   int(values[-2])
                decs = float(values[-1])
                # the following is not general, the length changes depending on the type of query
                # rah  =   int(d[23:25])
                # ram  =   int(d[26:28])
                # ras  = float(d[29:38])
                # decg =       d[39:40] # sign
                # decd =   int(d[40:42])
                # decm =   int(d[43:45])
                # decs = float(d[46:54])
                time = numpy.append(time, dateutil.parser.parse(d[1:18]))
                ra   = numpy.append(ra, coords.hms2rad([rah, ram, ras]))
                dec  = numpy.append(dec, coords.dms2rad([decd, decm, decs, decg]))

        logger.debug('Time = %s', time)
        logger.debug('RA = %s radians', ra)
        logger.debug('Dec = %s radians', dec)

        return time, ra, dec

    # ----------------------------------------------------------------------------------------------
    
    def AngularSize(self, target, start, end, step='1m'):
        '''
        Query Horizons for the angular size of an object

        Ang-diam = The equatorial angular width of the target body full disk, if it were fully
        visible to the observer.  Units: ARCSECONDS

        SOLAR PRESENCE (OBSERVING SITE)
        Time tag is followed by a blank, then a solar-presence symbol:
        *  Daylight (refracted solar upper-limb on or above apparent horizon)
        C  Civil twilight/dawn
        N  Nautical twilight/dawn
        A  Astronomical twilight/dawn
           Night OR geocentric ephemeris

        LUNAR PRESENCE WITH TARGET RISE/TRANSIT/SET MARKER (OBSERVING SITE)
        The solar-presence symbol is immediately followed by another marker symbol:
        m  Refracted upper-limb of Moon on or above apparent horizon
           Refracted upper-limb of Moon below apparent horizon OR geocentric
        r  Rise    (target body on or above cut-off RTS elevation)
        t  Transit (target body at or past local maximum RTS elevation)
        s  Set     (target body on or below cut-off RTS elevation)    
        '''
        logger = logging.getLogger('AngularSize')

        self.target = target
        self.senddaterange = True
        self.start  = start.strftime("'%Y-%b-%d %H:%M'")
        self.end    = end.strftime("'%Y-%b-%d %H:%M'")
        self.cal_format = 'CAL'
        self.make_ephem = 'YES'
        self.obj_data   = 'NO'
        self.quantities = '13'
        self.timestep   = step
        self.csvformat  = 'YES'

        response = self.Query()
        lines = response.text.splitlines()
        #logger.debug('Lines = %s', lines)
        firstline = lines.index('$$SOE') + 1
        lastline  = lines.index('$$EOE') - 1
        logger.debug('First line = %s', firstline)
        logger.debug('Last line = %s', lastline)
        numlines = lastline - firstline + 1
        logger.debug('Found %s lines of data:', numlines)

        dateandtime   = numpy.array([])
        angularsize   = numpy.array([])
        solarpresence = numpy.array([])
        lunarpresence = numpy.array([])

        for i in range(firstline, lastline + 1):
            line = lines[i]
            #logger.debug('Line[%s] = %s', i, line)
            if line and 'Daylight' not in line and 'Airmass' not in line:
                linearray = line.split(',')
                dateandtime   = numpy.append(dateandtime, dateutil.parser.parse(linearray[0]))
                solarpresence = numpy.append(solarpresence, str(linearray[1]))
                lunarpresence = numpy.append(lunarpresence, str(linearray[2]))
                angularsize   = numpy.append(angularsize, float(linearray[3]))

        data = {'date-time':dateandtime,
                'solar-presence':solarpresence,
                'lunar-presence':lunarpresence,
                'angular-size':angularsize}

        logger.debug('%s',data)
        return data

    # ----------------------------------------------------------------------------------------------
    
    def AngularSizeLatLong(self, target, start, end, step='1m'):
        '''
        Query Horizons for the angular size of an object and the planetodetic longitude and latitude
        of the center of the target disk seen by the observer.

        Ang-diam = The equatorial angular width of the target body full disk, if it were fully
        visible to the observer.  Units: ARCSECONDS

        Ob-lon Ob-lat =
        Apparent planetodetic longitude and latitude (IAU2009 model) of the center
        of the target disk seen by the observer at print-time.  This is NOT exactly the
        same as the "sub-observer*" (nearest) point for a non-spherical target shape,
        but is generally very close if not an irregular body shape.  Light travel-time
        from target to observer is taken into account.  Latitude is the angle between
        the equatorial plane and the line perpendicular to the reference ellipsoid of
        the body. The reference ellipsoid is an oblate spheroid with a single flatness
        coefficient in which the y-axis body radius is taken to be the same value as
        the x-axis radius.  For the gas giants Jupiter, Saturn, Uranus and Neptune,
        IAU2009 longitude is based on the "System III" prime meridian rotation angle of
        the magnetic field. By contrast, pole direction (thus latitude) is relative to
        the body dynamical equator. There can be an offset between the magnetic pole
        and the dynamical pole of rotation. Positive longitude is to the WEST.

        *The sub-observer point is defined as where a line connecting the observer and
        the center of the body intersects the body\'s surface.  Practically speaking,
        this can be visualized as the surface location at the center of the body\'s apparent disk. 

        Units: DEGREES

        SOLAR PRESENCE (OBSERVING SITE)
        Time tag is followed by a blank, then a solar-presence symbol:
        *  Daylight (refracted solar upper-limb on or above apparent horizon)
        C  Civil twilight/dawn
        N  Nautical twilight/dawn
        A  Astronomical twilight/dawn
           Night OR geocentric ephemeris

        LUNAR PRESENCE WITH TARGET RISE/TRANSIT/SET MARKER (OBSERVING SITE)
        The solar-presence symbol is immediately followed by another marker symbol:
        m  Refracted upper-limb of Moon on or above apparent horizon
           Refracted upper-limb of Moon below apparent horizon OR geocentric
        r  Rise    (target body on or above cut-off RTS elevation)
        t  Transit (target body at or past local maximum RTS elevation)
        s  Set     (target body on or below cut-off RTS elevation)    
        '''
        logger = logging.getLogger('AngularSizeLatLong')

        self.target = target
        self.senddaterange = True
        self.start  = start.strftime("'%Y-%b-%d %H:%M'")
        self.end    = end.strftime("'%Y-%b-%d %H:%M'")
        self.cal_format = 'CAL'
        self.make_ephem = 'YES'
        self.obj_data   = 'NO'
        self.quantities = "'8,13,14'"
        self.timestep   = step
        self.csvformat  = 'YES'

        response = self.Query()
        lines = response.text.splitlines()
        #logger.debug('Lines = %s', lines)
        firstline = lines.index('$$SOE') + 1
        lastline  = lines.index('$$EOE') - 1
        logger.debug('First line = %s', firstline)
        logger.debug('Last line = %s', lastline)
        numlines = lastline - firstline + 1
        logger.debug('Found %s lines of data:', numlines)

        dateandtime   = numpy.array([])
        angularsize   = numpy.array([])
        solarpresence = numpy.array([])
        lunarpresence = numpy.array([])
        airmass       = numpy.array([])
        extinction    = numpy.array([])
        latitude      = numpy.array([])
        longitude     = numpy.array([])

        for i in range(firstline, lastline + 1):
            line = lines[i]
            #logger.debug('Line[%s] = %s', i, line)
            
            if 'No ephemeris meets criteria.' in line:
                logger.warning ('%s not visible between %s and %s', target, self.start, self.end)
                return None
                
            if line and 'Daylight' not in line and 'Airmass' not in line:
                linearray = line.split(',')
                dateandtime   = numpy.append(dateandtime, dateutil.parser.parse(linearray[0]))
                solarpresence = numpy.append(solarpresence, str(linearray[1]))
                lunarpresence = numpy.append(lunarpresence, str(linearray[2]))
                airmass       = numpy.append(airmass,     float(linearray[3]))
                extinction    = numpy.append(extinction,  float(linearray[4]))
                angularsize   = numpy.append(angularsize, float(linearray[5]))
                longitude     = numpy.append(longitude,   float(linearray[6]))
                latitude      = numpy.append(latitude,    float(linearray[7]))

        data = {'date-time':dateandtime,
                'solar-presence':solarpresence,
                'lunar-presence':lunarpresence,
                'airmass':airmass,
                'angular-size':angularsize,
                'latitude':latitude,
                'longitude':longitude}

        logger.debug('%s',data)
        return data

    # ----------------------------------------------------------------------------------------------
    
    def Separation(self, target, start, end, step='1m'):
        '''
        Query Horizons for the angular separation of a non-lunar target body and the center
        of the primary body it revolves around, as seen by the observer (arcseconds) over a
        a specified range of time.
        Start and End must be datetime objects.
 
        Non-lunar natural satellite visibility codes (limb-to-limb):
 
            /t = Transitting primary body disk, /O = Occulted by primary body disk,
            /p = Partial umbral eclipse,        /P = Occulted partial umbral eclipse,
            /u = Total umbral eclipse,          /U = Occulted total umbral eclipse,
            /- = Target is the primary body,    /* = None of above ("free and clear")
 
        The radius of major bodies is taken to be the equatorial value (max)
        defined by the IAU2009 system. Atmospheric effects and oblateness aspect
        are not currently considered in these computations. Light-time is included.

        SOLAR PRESENCE (OBSERVING SITE)
        Time tag is followed by a blank, then a solar-presence symbol:
        *  Daylight (refracted solar upper-limb on or above apparent horizon)
        C  Civil twilight/dawn
        N  Nautical twilight/dawn
        A  Astronomical twilight/dawn
           Night OR geocentric ephemeris

        LUNAR PRESENCE WITH TARGET RISE/TRANSIT/SET MARKER (OBSERVING SITE)
        The solar-presence symbol is immediately followed by another marker symbol:
        m  Refracted upper-limb of Moon on or above apparent horizon
           Refracted upper-limb of Moon below apparent horizon OR geocentric
        r  Rise    (target body on or above cut-off RTS elevation)
        t  Transit (target body at or past local maximum RTS elevation)
        s  Set     (target body on or below cut-off RTS elevation)    
        '''
        logger = logging.getLogger('Separation')

        self.target = target
        self.senddaterange = True
        self.start  = start.strftime("'%Y-%b-%d %H:%M'")
        self.end    = end.strftime("'%Y-%b-%d %H:%M'")
        self.cal_format = 'CAL'
        self.make_ephem = 'YES'
        self.obj_data   = 'NO'
        self.quantities = '12'
        self.timestep   = step
        self.csvformat  = 'NO'

        response = self.Query()
        lines = response.text.splitlines()
        #logger.debug('Lines = %s', lines)
        firstline = lines.index('$$SOE') + 1
        lastline  = lines.index('$$EOE') - 1
        logger.debug('First line = %s', firstline)
        logger.debug('Last line = %s', lastline)
        numlines = lastline - firstline + 1
        logger.debug('Found %s lines of data:', numlines)

        #          1         2         3         4         5         6         7
        #0123456789012345678901234567890123456789012345678901234567890123456789012
        #
        # Date__(UT)__HR:MN      ang-sep/v
        #*********************************
        #$$SOE
        # 2016-Dec-01 14:14      42.5255/*
        # 2016-Dec-15 14:30  m   18.4973/u
        # 2016-Dec-15 14:40  m   16.7923/U
        # 2016-Dec-15 14:56  m   14.1825/P
        # 2016-Dec-15 15:00  m   13.5615/O
        #0123456789012345678901234567890123456789012345678901234567890123456789012
        # 2016-Dec-15 15:50 Am    8.3558/O
        # 2016-Dec-15 16:00 Nm    8.3423/O
        # 2016-Dec-15 16:30 Cm   10.6846/O
        #>..... Daylight Cut-off Requested .....<
        #>..... Airmass Cut-off Requested .....<
        # $$EOE
        # There are a variable number of elements (solar and lunar presence) so extract using
        # character numbers instead of column numbers:
    	# Shoule I use csv output which might be easier to parse ???

        time = numpy.array([])
        sep  = numpy.array([])
        vis  = numpy.array([])
        sol  = numpy.array([])
        lun  = numpy.array([])
        for i in range(firstline, lastline + 1):
            d = lines[i]
            logger.debug('Data[%s] = %s', i, d)
            if d and d[7:15] != 'Daylight' and d[7:14] != 'Airmass':
                time = numpy.append(time, dateutil.parser.parse(d[1:18]))
                sol  = numpy.append(sol,  str(d[19:20]))
                lun  = numpy.append(lun,  str(d[20:21]))
                sep  = numpy.append(sep,  float(d[22:31]))
                vis  = numpy.append(vis,  str(d[32:33]))

        data = {'date-time':time, 'solar-presence':sol, 'lunar-presence':lun, 'ang-sep':sep, 'visibility':vis}
        logger.debug('%s',data)
        return data

    # ----------------------------------------------------------------------------------------------
    # Query Horizons for the ephemeris of a target over the specified date range
    # Start & Stop must be date-time objects

    def Ephemeris(self, target, start, end, step=None):
        logger = logging.getLogger('Ephemeris')

        self.senddaterange = True
        self.target = target
        self.start = start.date()
        self.end = end.date()

        # It would be nice to calculate the optimal step size so that we get ~< 1440 lines.
        # However, we don't apriori know how long the target is up each night.
        # If no step size was supplied, make a conservative estimate of 8 hours/night:
        if step is None:
            duration = (self.end - self.start).total_seconds() / 60.
            logger.debug('Duration = %s minutes', duration)
            self.timestep = str(max(1,int(duration / 3. / 1440.))) + 'm'
        else:
            self.timestep = step
        logger.debug('Time step = %s', self.timestep)

        self.cal_format = 'BOTH'
        self.make_ephem = 'YES'
        self.obj_data   = 'YES'
        self.quantities = "'1,3'"
        response = self.Query()

        # Strip out seconds digits that end with 60.000 and 60.0000

        # Check that the file has less than 1440 lines

        return

    # ----------------------------------------------------------------------------------------------
    # Query Horizons

    def Query(self):
        logger = logging.getLogger('Query')
        logging.getLogger("requests").setLevel(logging.WARNING) # Only show warnings or worse
        
        # The items and order follow the JPL/Horizons batch example:
        # ftp://ssd.jpl.nasa.gov/pub/ssd/horizons_batch_example.long
        # and
        # ftp://ssd.jpl.nasa.gov/pub/ssd/horizons-batch-interface.txt
        # Note that spaces should be converted to '%20'
        
        url = 'https://ssd.jpl.nasa.gov/horizons_batch.cgi'

        logger.debug('self.target = %s', self.target)
        logger.debug('self.start = %s', self.start)
        logger.debug('self.end =   %s', self.end)
        logger.debug('self.timestep = %s', self.timestep)
        logger.debug('self.airmass = %s', self.airmass)
        logger.debug('self.skip_day = %s', self.skip_day)
        
        params = {'batch':1}
        params['COMMAND']    = "'" + self.target + "'"
        params['OBJ_DATA']   = self.obj_data # Toggles return of object summary data (YES or NO)
        params['MAKE_EPHEM'] = self.make_ephem # Toggles generation of ephemeris (YES or NO)
        params['TABLE_TYPE'] = 'OBSERVER'      # OBSERVER, ELEMENTS, VECTORS, or APPROACH 
        params['CENTER']     = self.center     # Set coordinate origin. MK=568, CP=I11, Earth=399
        params['REF_PLANE']  = None            # Table reference plane (ECLIPTIC, FRAME or BODY EQUATOR)
        params['COORD_TYPE'] = None            # Type of user coordinates in SITE_COORD
        params['SITE_COORD'] = None            # '0,0,0'
        if self.senddaterange:
            params['START_TIME'] = self.start      # Ephemeris start time YYYY-MMM-DD {HH:MM} {UT/TT}
            params['STOP_TIME']  = self.end        # Ephemeris stop time YYYY-MMM-DD {HH:MM}
            params['STEP_SIZE']  = "'" + self.timestep + "'" # Ephemeris step: integer# {units} {mode}
            params['TLIST']      = None            # Ephemeris time list

        # This only works for small numbers (~<1000) of times:
        #tlist = ' '.join(map(str,numpy.arange(2457419.5, 2457600.0, 0.3)))
        #params['TLIST']      = tlist            # Ephemeris time list

        params['QUANTITIES'] = self.quantities # Desired output quantity codes
        params['REF_SYSTEM'] = 'J2000'         # Reference frame
        params['OUT_UNITS']  = None            # VEC: Output units
        params['VECT_TABLE'] = None            # VEC: Table format
        params['VECT_CORR']  = None            # VEC: correction level
        params['CAL_FORMAT'] = self.cal_format # OBS: Type of date output (CAL, JD, BOTH)
        params['ANG_FORMAT'] = 'HMS'           # OBS: Angle format (HMS or DEG)
        params['APPARENT']   = None            # OBS: Apparent coord refract corr (AIRLESS or REFRACTED)
        params['TIME_DIGITS'] = 'MINUTES'      # OBS: Precision (MINUTES, SECONDS, or FRACSEC)
        params['TIME_ZONE']  = None            # Local civil time offset relative to UT ('+00:00')
        params['RANGE_UNITS'] = None           # OBS: range units (AU or KM)
        params['SUPPRESS_RANGE_RATE'] = 'NO'   # OBS: turn off output of delta-dot and rdot
        params['ELEV_CUT']   = '-90'           # OBS: skip output when below elevation
        params['SKIP_DAYLT'] = self.skip_day   # OBS: skip output when daylight
        params['SOLAR_ELONG'] = "'0,180'"      # OBS: skip output outside range
        params['AIRMASS']    = self.airmass    # OBS: skip output when airmass is > cutoff
        params['LHA_CUTOFF'] = None            # OBS: skip output when hour angle is > cutoff
        params['EXTRA_PREC'] = 'YES'           # OBS: show additional output digits (YES or NO)
        params['CSV_FORMAT'] = self.csvformat  # Output in comma-separated value format (YES or NO)
        params['VEC_LABELS'] = None            # label each vector component (YES or NO)
        params['ELM_LABELS'] = None            # label each osculating element
        params['TP_TYPE']    = None            # Time of periapsis for osculating element tables
        params['R_T_S_ONLY'] = 'NO'            # Print only rise/transit/set (NO, TVH, GEO, RAD, YES)
        # Skiping the section of close-approch parameters...
        # Skiping the section of heliocentric ecliptic osculating elements...
        response = requests.get(url, params=params)
        logger.debug('URL = %s', response.url)
        logger.debug('Response:\n%s', response.text)

        return response


# --------------------------------------------------------------------------------------------------
# Read a data file and return a target list

def ReadTargetList(datafile):
    logger = logging.getLogger('ReadTargetList')
    try:
        targetlist = numpy.loadtxt(datafile, usecols=[0], dtype='str', unpack=True)
    except:
        logger.error('Error reading %s', datafile)
        raise SystemExit
    return targetlist

# --------------------------------------------------------------------------------------------------

def argsort(seq):
    return sorted(list(range(len(seq))), key = seq.__getitem__)

#---------------------------------------------------------------------------------------------------

def unique(seq):
    return list(set(seq))


#---------------------------------------------------------------------------------------------------
# Return the FIRST index of the list item which contains a substring

def FindSubstringInList(mylist,mystring):
    for i, elem in enumerate(mylist):
        if mystring in elem:
            return i
    return -1

#---------------------------------------------------------------------------------------------------
# Return a list of indices of the list item which contains a substring

def FindSubstringsInList(mylist,mystring):
    indices = []
    for i, elem in enumerate(mylist):
        if mystring in elem:
            indices.append(i)
    return indices

# Could also be written as: indices = [i for i, s in enumerate(mylist) if 'aa' in s]
#---------------------------------------------------------------------------------------------------

def GetHorizonId(target):
    # A not-complete list of solar system major body Horizons IDs
    horiz = {'mercury': '199', 'venus': '299', 'mars': '499', 'jupiter': '599', 'saturn': '699',
             'uranus': '799', 'neptune': '899', 'pluto': '999', 'io': '501'}

    horizid = target
    if target.lower() in horiz.keys():
        horizid = horiz[target.lower()]

    return horizid

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Query JPL Horizons database.',
        epilog='Version: ' + version)

    # Required arguments:
    parser.add_argument('target', help='Target name.  If target = "asteroids",\
    query all asteroids in new-horizons.asteroids.dat.')

    # Boolean options:
    parser.add_argument('-c', '--coords', action='store_true', default=False, help='Show coordinates')
    parser.add_argument('--common', action='store_true', default=False, help='Return the common name')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='Show debugging output')

    parser.add_argument('-q', '--query', action='store_true', default=False,
                        help='Query a target.  For comets you may want to specify the more recent apparition \
                        with "DES=name;CAP"')
    
    parser.add_argument('-e', '--ephemeris', action='store_true', default=False, help='Generate an ephemeris')
    parser.add_argument('--daytime', action='store_true', default=False, help='Include daytime in ephemeris')

    # Options requiring values:
    parser.add_argument('--airmass', action='store', type=float, default=3, help='Airmass cutoff [3]')
    #parser.add_argument('-l', '--location', action='store', type=str, default=3, help='Location {GN,GS}')
    parser.add_argument('--start', action='store', type=str, default='today', help='Start date [today]')
    parser.add_argument('--end', action='store', type=str, default='tomorrow', help='End date [tomorrow]')
    parser.add_argument('--step', action='store', type=str, default=None, help='Step size, e.g. "1h". Unitless steps will return N equally spaced points.')
    parser.add_argument('--site', action='store', type=str, default='GN', help='Gemini site ["GN" or "GS"]')

    args = parser.parse_args()
    main(args)

#---------------------------------------------------------------------------------------------------

