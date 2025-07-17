# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import traceback
import asyncio
from datetime import datetime, timedelta
from dateutil.parser import parse as parsedt
from os import PathLike
from pathlib import Path
from typing import FrozenSet, Iterable, List, Mapping, Optional, Tuple, Dict

from gpp_client.api import WhereProgram, WhereEqProposalStatus, ProposalStatus, WhereOrderProgramId
from gpp_client import GPPClient, GPPDirector

from lucupy.minimodel import (AndOption, Atom, Band, CloudCover, Conditions, Constraints, ElevationType,
                              Group, GroupID, ImageQuality, Magnitude, MagnitudeBands, NonsiderealTarget, Observation,
                              ObservationClass, ObservationID, ObservationMode, ObservationStatus, Priority,
                              Program, ProgramID, ProgramMode, ProgramTypes, QAState, ResourceType,
                              ROOT_GROUP_ID, Semester, SemesterHalf, SetupTimeType, SiderealTarget, Site, SkyBackground,
                              Target, TargetTag, TargetName, TargetType, TimeAccountingCode, TimeAllocation, TimeUsed,
                              TimingWindow, TooType, WaterVapor, Wavelength)
from lucupy.observatory.gemini.geminiobservation import GeminiObservation
from lucupy.resource_manager import ResourceManager
from lucupy.timeutils import sex2dec
from lucupy.types import ZeroTime


from definitions import ROOT_DIR
from scheduler.core.programprovider.abstract import ProgramProvider
from scheduler.core.sources.sources import Sources
from scheduler.services import logger_factory


__all__ = [
    'GppProgramProvider',
    'gpp_program_data'
]

logger = logger_factory.create_logger(__name__)


# DEFAULT_GPP_DATA_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'programs.zip'
DEFAULT_PROGRAM_ID_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'gpp_program_ids.txt'


def get_gpp_data(program_ids: FrozenSet[str]) -> Iterable[dict]:
    """Query GPP for program data"""

    if program_ids:

        where = WhereProgram(id=WhereOrderProgramId(in_=list(program_ids)),
                             proposal_status=WhereEqProposalStatus(eq=ProposalStatus.ACCEPTED))
    else:
        # Bring everything that is accepted.
        where = WhereProgram(proposal_status=WhereEqProposalStatus(eq=ProposalStatus.ACCEPTED))

    try:
        client = GPPClient()
        director = GPPDirector(client)

        ask_director = director.scheduler.program.get_all(where=where)
        result = asyncio.run(ask_director)
        programs = result

        print(f"Adding program: {len(programs)}")
        # Pass the class information as a dictionary to mimic the OCS json format
        yield from programs
    except RuntimeError as e:
        logger.error(f'Problem querying program list {program_ids} data: \n{e}.')


def gpp_program_data(program_list: Optional[bytes] = None) -> Iterable[dict]:
    """Query GPP for the programs in program_list. If not given then query GPP for all appropriate observations"""
    if program_list is None:
        # Let's make it empty so we can remove it in the where
        id_frozenset = frozenset()
    else:
        try:
            # Try to read the file and create a frozenset from its lines
            if program_list.lower() == 'default':
                list_file = DEFAULT_PROGRAM_ID_PATH
            else:
                list_file = program_list

            if isinstance(program_list, bytes):
                file = program_list.decode('utf-8')
                id_frozenset = frozenset(f.strip() for f in file.split('\n') if f.strip() and f.strip()[0] != '#')
            else:
                with list_file.open('r') as file:
                    id_frozenset = frozenset(line.strip() for line in file if line.strip() and line.strip()[0] != '#')
        except FileNotFoundError:
            # If the file does not exist, set id_frozenset to None
            id_frozenset = None
    # return id_frozenset
    return get_gpp_data(id_frozenset)


def parse_preimaging(sequence: List[dict]) -> bool:

    preimaging = False
    try:
        if sequence[0][GppProgramProvider._AtomKeys.PREIMAGING].upper() == 'YES':
            preimaging = True
    except (KeyError, IndexError):
        pass

    return preimaging


class GppProgramProvider(ProgramProvider):
    """
    A ProgramProvider that parses programs from the GPP Observing Database.
    """

    _GPI_FILTER_WAVELENGTHS = {'Y': 1.05, 'J': 1.25, 'H': 1.65, 'K1': 2.05, 'K2': 2.25}
    _NIFS_FILTER_WAVELENGTHS = {'ZJ': 1.05, 'JH': 1.25, 'HK': 2.20}
    _CAL_OBSERVE_TYPES = frozenset(['FLAT', 'ARC', 'DARK', 'BIAS'])

    _site_for_inst = {'GMOS_NORTH': Site.GN, 'GMOS_SOUTH': Site.GS}

    # Allowed instrument statuses
    _OBSERVATION_STATUSES = frozenset({ObservationStatus.READY, ObservationStatus.ONGOING})

    # Translate instrument names to use the OCS Resources
    _gpp_inst_to_ocs = {'GMOS_NORTH': 'GMOS-N', 'GMOS_SOUTH': 'GMOS-S'}

    # GPP GMOS built-in GPU name to barcode
    # ToDo: Eventually this needs to come from another source, e.g. Resource, ICTD, decide whether to use name or barcode
    _fpu_to_barcode = {'GMOS-N': {
        'IFU-R': '10000009',
        'IFU-2': '10000007',
        'IFU-B': '10000008',
        'focus_array_new': '10005360',
        'LONG_SLIT_0_25': '10005351',
        'LONG_SLIT_0_50': '10005352',
        'LONG_SLIT_0_75': '10005353',
        'LONG_SLIT_1_00': '10005354',
        'LONG_SLIT_1_50': '10005355',
        'LONG_SLIT_2_00': '10005356',
        'LONG_SLIT_5_00': '10005357',
        'NS0.5arcsec': '10005367',
        'NS0.75arcsec': '10005368',
        'NS1.0arcsec': '10005369',
        'NS1.5arcsec': '10005358',
        'NS2.0arcsec': '10005359',
    },
        'GMOS-S': {
            'IFU-R': '10000009',
            'IFU-2': '10000007',
            'IFU-B': '10000008',
            'IFU-NS-2': '10000010',
            'IFU-NS-B': '10000011',
            'IFU-NS-R': '10000012',
            'focus_array_new': '10000005',
            'LONG_SLIT_0_25': '10005371',
            'LONG_SLIT_0_50': '10005372',
            'LONG_SLIT_0_75': '10005373',
            'LONG_SLIT_1_00': '10005374',
            'LONG_SLIT_1_50': '10005375',
            'LONG_SLIT_2_00': '10005376',
            'LONG_SLIT_5_00': '10005377',
            'NS0.5arcsec': '10005388',
            'NS0.75arcsec': '10005389',
            'NS1.0arcsec': '10005390',
            'NS1.5arcsec': '10005391',
            'NS2.0arcsec': '10005392',
            'PinholeC': '10005381',
        }
    }

    # GPP to OCS program type translation
    _gpp_prop_type = {
        'CLASSICAL': 'C',
        'DIRECTORS_TIME': 'DD',
        'FAST_TURNAROUND': 'FT',
        'LARGE_PROGRAM': 'LP',
        'POOR_WEATHER': 'PW',
        'QUEUE': 'Q',
        'DEMO_SCIENCE': 'DS',
        'SYSTEM_VERIFICATION': 'SV'
    }

    # We contain private classes with static members for the keys in the associative
    # arrays in order to have this information defined at the top-level once.
    class _ProgramKeys:
        ID = 'reference'
        INTERNAL_ID = 'id'
        BAND = 'queueBand'
        THESIS = 'isThesis'
        MODE = 'programMode'
        TOO_TYPE = 'tooType'
        TIME_ACCOUNT_ALLOCATION = 'allocations'
        TIME_CHARGE = 'timeCharge'
        # NOTE = 'INFO'
        # SCHED_NOTE = 'INFO_SCHEDNOTE'
        # PROGRAM_NOTE = 'INFO_PROGRAMNOTE'

    class _TAKeys:
        # CATEGORIES = 'timeAccountAllocationCategories'
        CATEGORY = 'partner'
        AWARDED_PROG_TIME = 'duration'
        AWARDED_PART_TIME = 'awardedPartnerTime'
        USED_PROG_TIME = 'program'
        USED_PART_TIME = 'partner'
        NOT_CHARGED_TIME = 'nonCharged'
        BAND = 'scienceBand'

    class _GroupKeys:
        ELEMENTS = 'elements'
        DELAY_MIN = 'minimumInterval'
        DELAY_MAX = 'maximumInterval'
        ORDERED = 'ordered'
        NUM_TO_OBSERVE = 'minimumRequired'
        GROUP_NAME = 'name'
        PARENT_ID = 'parentId'
        PARENT_INDEX = 'parentIndex'

    class _ObsKeys:
        # KEY = 'OBSERVATION_BASIC'
        ID = 'reference'
        INTERNAL_ID = 'id'
        # QASTATE = 'qaState'
        # LOG = 'obsLog'
        STATUS = 'status'
        # PRIORITY = 'priority'
        TITLE = 'title'
        INSTRUMENT = 'instrument'
        # SEQUENCE = 'sequence'
        # SETUPTIME_TYPE = 'setupTimeType'
        # SETUPTIME = '' # obs_may5_grp.execution.digest.acquisition.time_estimate.total.hours
        # OBS_CLASS = 'obsClass'
        # PHASE2 = 'phase2Status'
        ACTIVE = 'activeStatus'
        BAND = 'scienceBand'
        # TOO_OVERRIDE_RAPID = 'tooOverrideRapid'

    class _TargetKeys:
        KEY = 'target_environment'
        ASTERISM = 'asterism'
        BASE = 'explicitBase'
        TYPE = 'type'
        RA = 'ra'
        DEC = 'dec'
        PM = 'properMotion'
        EPOCH = 'epoch'
        DES = 'des'
        SIDEREAL_OBJECT_TYPE = 'sidereal'
        NONSIDEREAL_OBJECT_TYPE = 'nonsidereal'
        MAGNITUDES = 'magnitudes'
        NAME = 'name'

    class _TargetEnvKeys:
        GUIDE_GROUPS = 'guideEnvironments'
        # GUIDE_GROUP_NAME = 'name'
        # GUIDE_GROUP_PRIMARY = 'primaryGroup'
        # GUIDE_PROBE = 'probe'
        GUIDE_PROBE_KEY = 'probe'
        # AUTO_GROUP = 'auto'
        TARGET = 'guideTargets'
        # USER_TARGETS = 'userTargets'

    _constraint_to_value = {
        'POINT_ONE': 0.1,
        'POINT_TWO': 0.2,
        'POINT_THREE': 0.3,
        'POINT_FOUR': 0.4,
        'POINT_FIVE': 0.5,
        'POINT_SIX': 0.6,
        'POINT_EIGHT': 0.8,
        'ONE_POINT_ZERO': 1.0,
        'ONE_POINT_FIVE': 1.5,
        'TWO_POINT_ZERO': 2.0,
        'THREE_POINT_ZERO': 3.0,
        'DARKEST': 0.2,
        'DARK': 0.5,
        'GRAY': 0.8,
        'BRIGHT': 1.0,
        'VERY_DRY': 0.2,
        'DRY': 0.5,
        'MEDIAN': 0.8,
        'WET': 1.0,
    }

    class _ConstraintKeys:
        KEY = 'constraintSet'
        CC = 'cloudExtinction'
        IQ = 'imageQuality'
        SB = 'skyBackground'
        WV = 'waterVapor'
        ELEVATION = 'elevationRange'
        AIRMASS_TYPE = 'airMass'
        AIRMASS_MIN = 'min'
        AIRMASS_MAX = 'max'
        HA_TYPE = 'hourAngle'
        HA_MIN = 'minHours'
        HA_MAX = 'maxHours'
        TIMING_WINDOWS = 'timingWindows'

    class _AtomKeys:
        ATOM = 'atom'
        OBS_CLASS = 'class'
        # INSTRUMENT = ''
        # INST_NAME = ''
        WAVELENGTH = 'wavelength'
        # OBSERVED = ''
        TOTAL_TIME = 'duration'
        OFFSET_P = 'p'
        OFFSET_Q = 'q'
        EXPOSURE_TIME = 'exposure'
        # DATA_LABEL = ''
        # COADDS = ''
        FILTER = 'filter'
        DISPERSER = 'grating'
        OBSERVE_TYPE = 'type'
        # PREIMAGING = ''

    class _TimingWindowKeys:
        # TIMING_WINDOWS = 'timingWindows'
        INCLUSION = 'inclusion'
        START = 'startUtc'
        DURATION = 'end'
        ATUTC = 'atUtc'
        AFTER = 'after'
        REPEAT = 'repeat'
        TIMES = 'times'
        PERIOD = 'period'

    class _MagnitudeKeys:
        NAME = 'name'
        VALUE = 'value'

    class _FPUKeys:
        # GSAOI = 'instrument:utilityWheel'
        # GNIRS = 'instrument:slitWidth'
        GMOSN = 'fpu'
        # GPI = 'instrument:observingMode'
        # F2 = 'instrument:fpu'
        GMOSS = 'fpu'
        # NIRI = 'instrument:mask'
        # NIFS = 'instrument:mask'
        CUSTOM = 'fpuCustomMask'

    class _InstrumentKeys:
        NAME = 'instrument:name'
        DECKER = 'instrument:acquisitionMirror'
        ACQ_MIRROR = 'instrument:acquisitionMirror'
        CROSS_DISPERSED = 'instrument:crossDispersed'

    FPU_FOR_INSTRUMENT = {
        # 'GSAOI': _FPUKeys.GSAOI,
        # 'GPI': _FPUKeys.GPI,
        # 'Flamingos2': _FPUKeys.F2,
        # 'NIFS': _FPUKeys.NIFS,
        # 'GNIRS': _FPUKeys.GNIRS,
        'GMOS-N': _FPUKeys.GMOSN,
        'GMOS-S': _FPUKeys.GMOSS,
        # 'NIRI': _FPUKeys.NIRI
    }

    # An empty base target for when the target environment is empty for an Observation.
    _EMPTY_BASE_TARGET = SiderealTarget(
        name=TargetName('Empty'),
        magnitudes=frozenset(),
        type=TargetType.BASE,
        ra=0,
        dec=0,
        pm_ra=0,
        pm_dec=0,
        epoch=2000.0
    )

    _EMPTY_OBSERVATION = GeminiObservation(
        id=ObservationID('Empty-0001'),
        internal_id='None',
        order=0,
        title='Title',
        site=Site.GN,
        status=ObservationStatus.READY,
        active=False,
        priority=Priority.LOW,
        setuptime_type=SetupTimeType.FULL,
        acq_overhead=timedelta(seconds=600),
        obs_class=ObservationClass.SCIENCE,
        targets=[_EMPTY_BASE_TARGET],
        guiding=None,
        sequence=[],
        constraints=None,
        belongs_to=None,
        too_type=None,
        preimaging=False
    )

    _EMPTY_ROOT_GROUP = Group(
        id=GroupID('root'),
        program_id=ProgramID('Empty'),
        group_name='root',
        number_to_observe=1,
        delay_min=0,
        delay_max=0,
        children=[_EMPTY_OBSERVATION],
        group_option=AndOption.CONSEC_ORDERED)

    def __init__(self,
                 obs_classes: FrozenSet[ObservationClass],
                 sources: Sources):
        super().__init__(obs_classes, sources)

    def parse_magnitude(self, data: dict) -> Magnitude:
        band = MagnitudeBands[data[GppProgramProvider._MagnitudeKeys.NAME]]
        value = data[GppProgramProvider._MagnitudeKeys.VALUE]
        return Magnitude(
            band=band,
            value=value,
            error=None)

    def parse_timing_window(self, data: dict) -> TimingWindow:
        """Parse GPP timing windows"""
        # Examples
        # [TimingWindow(inclusion='INCLUDE', start_utc='2024-02-29 17:47:01.657', end=TimingWindowEndAt(__typename__='TimingWindowEndAt', at_utc='2024-04-29 19:47:00')),
        # TimingWindow(inclusion='INCLUDE', start_utc='2024-06-22 18:46:31.08', end=None),
        # TimingWindow(inclusion='INCLUDE', start_utc='2024-06-24 14:08:53.707', end=TimingWindowEndAfter(__typename__='TimingWindowEndAfter', after=TimeSpan(seconds=172800.0), repeat=TimingWindowRepeat(period=TimeSpan(seconds=216000.0), times=None))),
        # TimingWindow(inclusion='INCLUDE', start_utc='2024-06-24 14:09:37.32', end=TimingWindowEndAfter(__typename__='TimingWindowEndAfter', after=TimeSpan(seconds=172800.0), repeat=TimingWindowRepeat(period=TimeSpan(seconds=216000.0), times=6)))]

        repeat_info = TimingWindow.NON_REPEATING
        period_info = TimingWindow.NO_PERIOD

        if data[GppProgramProvider._TimingWindowKeys.INCLUSION] == 'INCLUDE':
            # Start
            start = parsedt(data[GppProgramProvider._TimingWindowKeys.START] + '+00:00')

            # End
            duration_info = data[GppProgramProvider._TimingWindowKeys.DURATION]
            if duration_info is None:
                duration = TimingWindow.INFINITE_DURATION
            else:
                try:
                    at_utc = parsedt(duration_info[GppProgramProvider._TimingWindowKeys.ATUTC] + '+00:00')
                    duration = at_utc - start
                except KeyError:
                    try:
                        duration = timedelta(
                            seconds=duration_info[GppProgramProvider._TimingWindowKeys.AFTER]['seconds'])
                        period_info = duration_info[GppProgramProvider._TimingWindowKeys.REPEAT] \
                            [GppProgramProvider._TimingWindowKeys.PERIOD]['seconds']
                        repeat_info = duration_info[GppProgramProvider._TimingWindowKeys.REPEAT] \
                            [GppProgramProvider._TimingWindowKeys.TIMES]
                    except KeyError:
                        duration = None
                        period_info = TimingWindow.NO_PERIOD
                        repeat_info = TimingWindow.NON_REPEATING

            if repeat_info is None:
                repeat = TimingWindow.OCS_INFINITE_REPEATS
            else:
                repeat = repeat_info

            if repeat == TimingWindow.NON_REPEATING:
                period = TimingWindow.NO_PERIOD
            else:
                period = timedelta(seconds=period_info)

        # else:
        #     continue
        # ToDo: determine how to handle exclusion windows

        return TimingWindow(
            start=start,
            duration=duration,
            repeat=repeat,
            period=period)

    def parse_elevation(self, data: dict) -> Tuple[Optional[float], Optional[float], Optional[ElevationType]]:
        """Parse GPP elevation constraints"""

        try:
            elevation_min = data[GppProgramProvider._ConstraintKeys.AIRMASS_TYPE][
                GppProgramProvider._ConstraintKeys.AIRMASS_MIN]
            elevation_max = data[GppProgramProvider._ConstraintKeys.AIRMASS_TYPE][
                GppProgramProvider._ConstraintKeys.AIRMASS_MAX]
            elevation_type = ElevationType['AIRMASS']
        except KeyError:
            try:
                elevation_min = data[GppProgramProvider._ConstraintKeys.HA_TYPE][
                    GppProgramProvider._ConstraintKeys.HA_MIN]
                elevation_max = data[GppProgramProvider._ConstraintKeys.HA_TYPE][
                    GppProgramProvider._ConstraintKeys.HA_MAX]
                elevation_type = ElevationType['HOUR_ANGLE']
            except KeyError:
                elevation_min = 1.0
                elevation_max = 2.0
                elevation_type = ElevationType['AIRMASS']

        return elevation_min, elevation_max, elevation_type

    def parse_conditions(self, data: dict, x_max: float) -> Conditions:
        def to_value(cond: str) -> float:
            """
            Parse the conditions value as a float out of the string passed by the GPP program extractor.
            """
            value = GppProgramProvider._constraint_to_value[cond]
            try:
                return float(value)
            except (ValueError, TypeError) as e:
                # Either of these will just be a ValueError.
                msg = f'Illegal value for constraint: {value}'
                raise ValueError(e, msg)

        def to_percent_bin(cond, const, x_max) -> float:
            """Convert to legacy OCS percentile bins"""

            cc_bins = [0.1, 0.3, 1.0, 3.0]
            cc_bin_values = [0.5, 0.7, 0.8, 1.0]

            iq_bins = [0.45, 0.75, 1.05, 1.5]  # for r, should be wavelength dependent
            iq_bin_values = [0.2, 0.7, 0.85, 1.0]

            # Numerical value equivalent
            value = to_value(const)

            if cond == GppProgramProvider._ConstraintKeys.CC:
                bin_value = cc_bin_values[-1]
                for i_bin, bin_lim in enumerate(cc_bins):
                    if value <= bin_lim:
                        bin_value = cc_bin_values[i_bin]
                        break
            elif cond == GppProgramProvider._ConstraintKeys.IQ:
                bin_value = iq_bin_values[-1]
                iqzen = value * x_max ** -0.6
                # print(value, iqzen)
                for i_bin, bin_lim in enumerate(iq_bins):
                    if iqzen <= bin_lim:
                        bin_value = iq_bin_values[i_bin]
                        break
            else:
                bin_value = value

            return bin_value


        return Conditions(
            *[lookup(to_percent_bin(key, data[key], x_max)) for lookup, key in
              [(CloudCover, GppProgramProvider._ConstraintKeys.CC),
               (ImageQuality, GppProgramProvider._ConstraintKeys.IQ),
               (SkyBackground, GppProgramProvider._ConstraintKeys.SB),
               (WaterVapor, GppProgramProvider._ConstraintKeys.WV)]])

    def parse_constraints(self, data: dict) -> Constraints:

        # Parse the timing windows.
        timing_windows = [self.parse_timing_window(tw_data)
                          for tw_data in data[GppProgramProvider._ConstraintKeys.TIMING_WINDOWS]]

        # Get the elevation data
        elevation_min, elevation_max, elevation_type = self.parse_elevation(
            data[GppProgramProvider._ConstraintKeys.KEY][GppProgramProvider._ConstraintKeys.ELEVATION])

        # Get the conditions
        # ToDo: support GPP on-target conditions constraints rather than converting to percentile bins
        if elevation_type == ElevationType['AIRMASS']:
            airmass_max = elevation_max
        else:
            airmass_max = 2.0  # should be converted from the max |HA|, but this is better than otherwise
        conditions = self.parse_conditions(data[GppProgramProvider._ConstraintKeys.KEY], airmass_max)

        return Constraints(
            conditions=conditions,
            elevation_type=elevation_type,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
            timing_windows=timing_windows,
            strehl=None)

    def _parse_target_header(self, data: dict) -> Tuple[TargetName, set[Magnitude]]:
        """
        Parse the common target header information out of a target.
        """
        name = TargetName(data[GppProgramProvider._TargetKeys.NAME])
        # magnitude_data = data.setdefault(GppProgramProvider._TargetKeys.MAGNITUDES, [])
        # magnitudes = {self.parse_magnitude(m) for m in magnitude_data}
        magnitudes = {}

        # GPP doesn't have the target TYPE tag, set from the context further up
        # target_type = TargetType.OTHER
        # target_type_data = data[GppProgramProvider._TargetKeys.TYPE].replace('-', '_').replace(' ', '_').upper()
        # try:
        #     target_type = TargetType[target_type_data]
        # except KeyError as e:
        #     msg = f'Target {name} has illegal type {target_type_data}.'
        #     raise KeyError(e, msg)

        return name, magnitudes

    def parse_sidereal_target(self, data: dict, targ_type: str) -> SiderealTarget:
        """Parse GPP sidereal target information"""
        # print(target[ 'id'], target['name'], target['sidereal']['ra']['hms'], target['sidereal']['dec']['dms'],  target['sidereal']['epoch'],
        #   target['sidereal']['proper_motion']['ra']['milliarcseconds_per_year'], target['sidereal']['proper_motion']['dec']['milliarcseconds_per_year'])

        name, magnitudes = self._parse_target_header(data)
        ra_hhmmss = data[GppProgramProvider._TargetKeys.SIDEREAL_OBJECT_TYPE][GppProgramProvider._TargetKeys.RA][
            'hms']
        dec_ddmmss = data[GppProgramProvider._TargetKeys.SIDEREAL_OBJECT_TYPE][GppProgramProvider._TargetKeys.DEC][
            'dms']

        # Convert RA/Dec to decimal degrees
        ra = sex2dec(ra_hhmmss, to_degree=True)
        dec = sex2dec(dec_ddmmss, to_degree=False)

        # Proper motion
        try:
            pm_ra = data[GppProgramProvider._TargetKeys.SIDEREAL_OBJECT_TYPE][GppProgramProvider._TargetKeys.PM] \
                [GppProgramProvider._TargetKeys.RA]['milliarcseconds_per_year']
            pm_dec = data[GppProgramProvider._TargetKeys.SIDEREAL_OBJECT_TYPE][GppProgramProvider._TargetKeys.PM] \
                [GppProgramProvider._TargetKeys.DEC]['milliarcseconds_per_year']
            epoch_str = data[GppProgramProvider._TargetKeys.SIDEREAL_OBJECT_TYPE][GppProgramProvider._TargetKeys.EPOCH]
            # Strip off any leading letter, make float
            epoch = float(epoch_str[1:]) if epoch_str[0] in ['B', 'J'] else float(epoch_str)
        except KeyError as e:
            pm_ra = 0.0
            pm_dec = 0.0
            epoch = 2000.0

        return SiderealTarget(
            name=name,
            magnitudes=frozenset(magnitudes),
            type=TargetType[targ_type],
            ra=ra,
            dec=dec,
            pm_ra=pm_ra,
            pm_dec=pm_dec,
            epoch=epoch)

    def parse_nonsidereal_target(self, data: dict) -> NonsiderealTarget:
        """
        TODO: Retrieve the Ephemeris data.
        TODO: Should we be doing this here, or in the Collector?
        """
        name, magnitudes, target_type = self._parse_target_header(data)
        des = data.get(GppProgramProvider._TargetKeys.DES, name)

        # This is redundant: it is always 'nonsidereal'
        # tag = data[GppProgramProvider._TargetKeys.TAG]

        # This is the tag information that we want: either MAJORBODY, COMET, or ASTEROID
        tag_str = data[GppProgramProvider._TargetKeys.NONSIDEREAL_OBJECT_TYPE]
        tag = TargetTag[tag_str]

        # RA and dec will be looked up when determining target info in Collector.
        return NonsiderealTarget(
            name=name,
            magnitudes=frozenset(magnitudes),
            type=target_type,
            des=des,
            tag=tag)

    @staticmethod
    def _parse_instrument(data: dict) -> str:
        """Get the instrument name"""
        fpu = None
        instrument = None

        for step in data:
            # Ignore acquisitions
            if step[GppProgramProvider._AtomKeys.OBS_CLASS].upper() not in [ObservationClass.ACQ.name,
                                                                            ObservationClass.ACQCAL.name]:
                inst_key = GppProgramProvider._AtomKeys.INSTRUMENT
                # Visitor instrument names are in GppProgramProvider._AtomKeys.INST_NAME
                if GppProgramProvider._AtomKeys.INST_NAME in step:
                    inst_key = GppProgramProvider._AtomKeys.INST_NAME
                instrument = step[inst_key].split(' ')[0]
                # For using OCS Resource
                # print(step[inst_key][0])
                # instrument = GppProgramProvider._gpp_inst_to_ocs[step[inst_key].split(' ')[0]]
                # print(instrument)

                if instrument in GppProgramProvider.FPU_FOR_INSTRUMENT:
                    if GppProgramProvider.FPU_FOR_INSTRUMENT[instrument] in step:
                        fpu = step[GppProgramProvider.FPU_FOR_INSTRUMENT[instrument]]

                if instrument in ['GMOS-N', 'GMOS_NORTH'] and fpu == 'IFU Left Slit (blue)':
                    instrument = 'GRACES'

                break

        return instrument

    @staticmethod
    def _parse_instrument_configuration(data: dict, instrument: str) \
            -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Wavelength]]:
        """
        A dict is return until the Instrument configuration model is created
        """

        def find_filter(filter_input: str, filter_dict: Mapping[str, float]) -> Optional[str]:
            return next(filter(lambda f: f in filter_input, filter_dict), None)

        # FPU
        fpu = None
        if instrument in GppProgramProvider.FPU_FOR_INSTRUMENT:
            if GppProgramProvider._FPUKeys.CUSTOM in data.keys():
                # This will assign the MDF name to the FPU
                fpu = data[GppProgramProvider._FPUKeys.CUSTOM]
            elif GppProgramProvider.FPU_FOR_INSTRUMENT[instrument] in data.keys():
                fpu = data[GppProgramProvider.FPU_FOR_INSTRUMENT[instrument]]

        # Disperser
        disperser = None
        if instrument in ['IGRINS', 'MAROON-X', 'GRACES']:
            disperser = instrument
        elif GppProgramProvider._AtomKeys.DISPERSER in data.keys():
            disperser = data[GppProgramProvider._AtomKeys.DISPERSER]

        # if instrument == 'GNIRS':
        #     if (data[OcsProgramProvider._InstrumentKeys.ACQ_MIRROR] == 'in'
        #             and data[OcsProgramProvider._InstrumentKeys.DECKER] == 'acquisition'):
        #         disperser = 'mirror'
        #     else:
        #         disperser = disperser.replace('grating', '') + data[OcsProgramProvider._InstrumentKeys.CROSS_DISPERSED]
        # elif instrument == 'Flamingos2' and fpu == 'FPU_NONE':
        #     if data['instrument:decker'] == 'IMAGING':
        #         disperser = data['instrument:decker']

        # Filter
        if GppProgramProvider._AtomKeys.FILTER in data.keys():
            filt = data[GppProgramProvider._AtomKeys.FILTER]
        elif instrument == 'GPI':
            filt = find_filter(fpu, GppProgramProvider._GPI_FILTER_WAVELENGTHS)
        else:
            if instrument == 'GNIRS':
                filt = None
            else:
                filt = 'Unknown'
        # if instrument == 'NIFS' and 'Same as Disperser' in filt:
        #     filt = find_filter(disperser[0], OcsProgramProvider._NIFS_FILTER_WAVELENGTHS)

        try:
            wavelength = Wavelength(GppProgramProvider._GPI_FILTER_WAVELENGTHS[filt] if instrument == 'GPI' \
                                        else float(data[GppProgramProvider._AtomKeys.WAVELENGTH]))
        except KeyError:
            wavelength = None

        return fpu, disperser, filt, wavelength

    def parse_atoms(self, site: Site, sequence: List[dict], qa_states: List[QAState] = frozenset([]),
                    split: bool = True, split_by_iterator: bool = False) -> List[Atom]:
        """
        Parse the sequence by GPP atoms
        qa_states: Determine if needed for GPP
        split/split_by_iterator: not used for GPP
        """

        def guide_state(guide_step: dict) -> bool:
            # return any('guideWith' in key and guide == 'guide' for key, guide in guide_step.items())
            return True

        def search_list(val, alist):
            return any(val in elem for elem in alist)

        def determine_mode(inst: str) -> ObservationMode:
            # print(f'inst: {inst} dispersers: {dispersers}')
            # print(f'\t fpus: {fpus}')
            obs_mode = ObservationMode.UNKNOWN

            if 'GMOS' in inst:
                if disperser.upper == 'MIRROR':
                    obs_mode = ObservationMode.IMAGING
                elif search_list('LONG_SLIT', fpus):
                    obs_mode = ObservationMode.LONGSLIT
                elif search_list('IFU', fpus):
                    obs_mode = ObservationMode.IFU
                elif search_list('G', fpus):
                    obs_mode = ObservationMode.MOS

            return obs_mode

        # site = GppProgramProvider._site_for_inst[instrument]
        # instrument = sequence.meta['instrument'].upper()
        instrument = GppProgramProvider._gpp_inst_to_ocs[sequence.meta['instrument'].upper()]
        # print(instrument)

        fpus = []
        dispersers = []
        filters = []
        wavelengths = []
        resources = frozenset()
        fpu_resources = frozenset()
        disperser_resources = frozenset()

        atoms = []
        classes = []
        all_classes = []
        guiding = []
        qa_states = []
        prev_atom_id = -1
        n_atom = 0
        instrument_resources = frozenset([self._sources.origin.resource.lookup_resource(instrument, resource_type=ResourceType.INSTRUMENT)])
        for step in sequence:
            if step[GppProgramProvider._AtomKeys.OBS_CLASS] != 'ACQUISITION':
                next_atom = False
                atom_id = step[GppProgramProvider._AtomKeys.ATOM]
                observe_class = step[GppProgramProvider._AtomKeys.OBS_CLASS]
                step_time = step[GppProgramProvider._AtomKeys.TOTAL_TIME]

                # print(step['atom'], step['class'], step['type'], step['exposure'], step['duration'], step_time, \
                #       step['fpu'], step['grating'], step['filter'], step['wavelength'])

                # Instrument configuration aka Resource.
                fpu, disperser, filt, wavelength = GppProgramProvider._parse_instrument_configuration(step, instrument)
                # print(f"\t{fpu} {disperser} {filt} {wavelength}")

                if atom_id != prev_atom_id:
                    # Convert all the different components into Resources.
                    if 'GMOS' in instrument:
                        # Convert FPUs and dispersers to barcodes. Note that None might be contained in some of these
                        # sets, but we filter below to remove them.
                        # ToDo: decide whether to use FPU names or barcodes for resource matching
                        fpu_resources = frozenset([self._sources.origin.resource.lookup_resource(
                            GppProgramProvider._fpu_to_barcode[instrument][fpu], description=fpu)
                            for fpu in fpus])
                        disperser_resources = frozenset(
                            [self._sources.origin.resource.lookup_resource(disperser.split('_')[0],
                                                                           resource_type=ResourceType.DISPERSER)
                             for disperser in dispersers])
                    resources = frozenset([r for r in fpu_resources | disperser_resources | instrument_resources])

                    # Close previous atom, if any
                    if n_atom > 0:
                        mode = determine_mode(instrument)
                        previous_atom = atoms[-1]
                        previous_atom.qa_state = min(qa_states, default=QAState.NONE)
                        if previous_atom.qa_state is not QAState.NONE:
                            previous_atom.observed = True
                        previous_atom.resources = resources
                        previous_atom.guide_state = any(guiding)
                        previous_atom.wavelengths = frozenset(wavelengths)
                        previous_atom.obs_mode = mode

                    # New atom entry
                    n_atom += 1

                    classes = []
                    guiding = []
                    resources = []
                    wavelengths = []
                    fpus = []
                    dispersers = []
                    filters = []
                    mode = None
                    atoms.append(Atom(id=atom_id,
                                      exec_time=ZeroTime,
                                      prog_time=ZeroTime,
                                      part_time=ZeroTime,
                                      program_used=ZeroTime,
                                      partner_used=ZeroTime,
                                      not_charged=ZeroTime,
                                      observed=False,
                                      qa_state=QAState.NONE,
                                      guide_state=False,
                                      resources=resources,
                                      wavelengths=frozenset(wavelengths),
                                      obs_mode=mode))
                    prev_atom_id = atoms[-1].id

                # If FPU is None, 'None', or FPU_NONE, which are effectively the same thing, we ignore.
                if fpu is not None and fpu != 'None' and fpu != 'FPU_NONE':
                    fpus.append(fpu)
                dispersers.append(disperser)
                if filt and filt != 'None':
                    filters.append(filt)
                wavelengths.append(wavelength)

                # Update atom
                classes.append(observe_class)
                all_classes.append(observe_class)
                guiding.append(guide_state(step))

                atoms[-1].exec_time += timedelta(seconds=step_time)
                # atom_id = n_atom

                # TODO: Add Observe Class enum
                if 'GCAL' in observe_class:
                    atoms[-1].part_time += timedelta(seconds=step_time)
                else:
                    atoms[-1].prog_time += timedelta(seconds=step_time)

            if n_atom > 0:
                # Convert all the different components into Resources.
                if 'GMOS' in instrument:
                    # Convert FPUs and dispersers to barcodes. Note that None might be contained in some of these
                    # sets, but we filter below to remove them.
                    fpu_resources = frozenset([self._sources.origin.resource.lookup_resource(
                        GppProgramProvider._fpu_to_barcode[instrument][fpu], description=fpu)
                        for fpu in fpus])
                    disperser_resources = frozenset(
                        [self._sources.origin.resource.lookup_resource(disperser.split('_')[0])
                         for disperser in dispersers])
                resources = frozenset([r for r in fpu_resources | disperser_resources | instrument_resources])

                mode = determine_mode(instrument)
                previous_atom = atoms[-1]
                previous_atom.qa_state = min(qa_states, default=QAState.NONE)
                if previous_atom.qa_state is not QAState.NONE:
                    previous_atom.observed = True
                previous_atom.resources = resources
                previous_atom.guide_state = any(guiding)
                previous_atom.wavelengths = frozenset(wavelengths)
                previous_atom.obs_mode = mode

        obs_class = ObservationClass.NONE
        if 'SCIENCE' in all_classes:
            obs_class = ObservationClass.SCIENCE
        elif 'PROGRAM_CAL' in all_classes:
            obs_class = ObservationClass.PROGCAL
        elif 'PARTNER_CAL' in all_classes:
            obs_class = ObservationClass.PARTNERCAL
        elif 'DAY_CAL' in all_classes:
            obs_class = ObservationClass.DAYCAL

        return atoms, obs_class

    def parse_target(self, data: dict, targ_type: str) -> Target:
        """
        Parse a general target - either sidereal or nonsidereal - from the supplied data.
        If we are a ToO, we don't have a target, and thus we don't have a tag. Thus, this raises a KeyError.
        """
        if data[GppProgramProvider._TargetKeys.SIDEREAL_OBJECT_TYPE]:
            return self.parse_sidereal_target(data, targ_type)
        elif data[GppProgramProvider._TargetKeys.NONSIDEREAL_OBJECT_TYPE]:
            return self.parse_nonsidereal_target(data, targ_type)
        else:
            msg = f'Illegal target type.'
            raise ValueError(msg)

    def parse_observation(self,
                          data: dict,
                          num: Tuple[Optional[int], int],
                          program_id: ProgramID,
                          split: bool = True,
                          split_by_iterator: bool = False) -> Optional[Observation]:
        """
        Parse GPP observation query dictionary into the minimodel observation
        """
        # folder_num is not currently used.
        folder_num, obs_num = num

        # Check the obs_class. If it is illegal, return None.
        # At the same time, ignore inactive observations.
        # ToDo: Eventually the obs_id should be the reference label, the id is the internal_id
        internal_id = data[GppProgramProvider._ObsKeys.INTERNAL_ID]
        # obs_id = f"{program_id.id}-{internal_id.replace('-', '')}"
        obs_id = data[GppProgramProvider._ObsKeys.ID]['label'] if GppProgramProvider._ObsKeys.ID in data.keys() \
            else f"{program_id.id}-{internal_id.replace('-', '')}"

        order = None
        obs_class = ObservationClass.NONE
        belongs_to = program_id

        try:
            print("OBS: ", data)
            # doesnt exits anymore
            active = True
            # active = data.get(GppProgramProvider._ObsKeys.ACTIVE)

            # if not active or active.upper() != 'INACTIVE':
            #     logger.warning(f"Observation {obs_id} is inactive (skipping).")
            #     print(f"Observation {obs_id} is inactive (skipping).")
            #    return None

            # ToDo: there is no longer an observation-level obs_class, maybe check later from atom classes
            # obs_class = ObservationClass[data[GppProgramProvider._ObsKeys.OBS_CLASS].upper()]
            # if obs_class not in self._obs_classes or not active:
            #     logger.warning(f'Observation {obs_id} not in a specified class (skipping): {obs_class.name}.')
            #     return None

            # By default, assume ToOType of None unless otherwise indicated.
            too_type: Optional[TooType] = None

            title = data[GppProgramProvider._ObsKeys.TITLE]
            # site = Site[data[GppProgramProvider._ObsKeys.ID].split('-')[0]]
            site = self._site_for_inst[data[GppProgramProvider._ObsKeys.INSTRUMENT]]
            # priority = Priority[data[GppProgramProvider._ObsKeys.PRIORITY].upper()]
            priority = Priority.MEDIUM

            # If the status is not legal, terminate parsing.
            status = ObservationStatus.READY
            # status = ObservationStatus[data[GppProgramProvider._ObsKeys.STATUS].upper()]
            # if status not in GppProgramProvider._OBSERVATION_STATUSES:
            #    logger.warning(f"Observation {obs_id} has invalid status {status.name}.")
            #     print(f"Observation {obs_id} has invalid status {status.name}.")
            #    return None

            # ToDo: where to get the setup type?
            # setuptime_type = SetupTimeType[data[GppProgramProvider._ObsKeys.SETUPTIME_TYPE]]
            setuptime_type = SetupTimeType.FULL
            # acq_overhead = timedelta(seconds=data['execution']['digest']['setup']['full']['seconds'])
            acq_overhead = timedelta(seconds=0)

            # Science band
            band_value = data.get(GppProgramProvider._ObsKeys.BAND)
            band = Band[band_value] if band_value is not None else None


            # Constraints
            find_constraints = {
                GppProgramProvider._ConstraintKeys.KEY: data[GppProgramProvider._ConstraintKeys.KEY],
                GppProgramProvider._ConstraintKeys.TIMING_WINDOWS: data[
                    GppProgramProvider._ConstraintKeys.TIMING_WINDOWS]}
            # print(find_constraints)
            constraints = self.parse_constraints(find_constraints) if find_constraints else None

            # QA states, needed?
            # qa_states = [QAState[log_entry[GppProgramProvider._ObsKeys.QASTATE].upper()] for log_entry in
            #              data[GppProgramProvider._ObsKeys.LOG]]

            # Atoms
            # ToDo: Perhaps add the sequence query to the original observation query
            # TODO change pyexplore to other api query
            # sequence = explore.sequence(internal_id, include_acquisition=True)
            sequence = data["sequence"]
            atoms, obs_class = self.parse_atoms(site, sequence)

            # Pre-imaging
            preimaging = False

            # Targets
            targets = []

            # Get the target environment. Each observation should have exactly one, but the name will
            # not necessarily be predictable as we number them.
            guiding = {}
            guide_group = None
            target_env_keys = [key for key in data.keys() if key.startswith(GppProgramProvider._TargetKeys.KEY)]
            if len(target_env_keys) > 1:
                raise ValueError(f'Observation {obs_id} has multiple target environments. Cannot process.')

            if not target_env_keys:
                # No target environment. Use the empty target.
                logger.warning(f'No target environment found for observation {obs_id}. Using empty base target.')
                targets.append(GppProgramProvider._EMPTY_BASE_TARGET)

            else:
                # Process the target environment.
                target_env = data[target_env_keys[0]]
                # Use the explicit base if available, otherwise the first target in the asterism
                try:
                    target_info = target_env[GppProgramProvider._TargetKeys.BASE]
                except KeyError:
                    target_info = target_env[GppProgramProvider._TargetKeys.ASTERISM][0]

                # Get the target
                try:
                    base = self.parse_target(target_info, targ_type='BASE')
                    targets.append(base)
                except KeyError:
                    logger.warning(f"No base target found for observation {obs_id}. Using empty base target.")
                    targets.append(GppProgramProvider._EMPTY_BASE_TARGET)

                # Parse the guide stars if guide star data is supplied.
                try:
                    # Let's ignore the guide environment
                    # guide_groups = target_env[GppProgramProvider._TargetEnvKeys.GUIDE_GROUPS]
                    # ToDo: is there a better option than the first one?
                    # guide_group = guide_groups[0]
                    #     auto_guide_group = [group for group in guide_groups
                    #                         if group[GppProgramProvider._TargetEnvKeys.GUIDE_GROUP_NAME] ==
                    #                         GppProgramProvider._TargetEnvKeys.AUTO_GROUP]
                    #     primary_guide_group = [group for group in guide_groups
                    #                            if group[GppProgramProvider._TargetEnvKeys.GUIDE_GROUP_PRIMARY]]

                    #     guide_group = None
                    #     if auto_guide_group:
                    #         if len(auto_guide_group) > 1:
                    #             raise ValueError(f'Multiple auto guide groups found for {obs_id}.')
                    #         guide_group = auto_guide_group[0]
                    #     elif primary_guide_group:
                    #         if len(primary_guide_group) > 1:
                    #             raise ValueError(f'Multiple primary guide groups found for {obs_id}.')
                    #         guide_group = primary_guide_group[0]

                    #     # Now we parse out the guideProbe list, which contains the information about the
                    #     # guide probe keys and the targets.
                    if guide_group is not None:
                        for guide_targ in guide_group[GppProgramProvider._TargetEnvKeys.TARGET]:
                            guider = guide_targ[GppProgramProvider._TargetEnvKeys.GUIDE_PROBE_KEY]
                            resource = ResourceManager().lookup_resource(rid=guider, rtype=ResourceType.WFS)
                            target = self.parse_target(guide_targ, targ_type='GUIDESTAR')
                            guiding[resource] = target
                            targets.append(target)

                except KeyError:
                    logger.warning(f'No guide group data found for observation {obs_id}')

                # If the ToO override rapid setting is in place, set to RAPID.
                # Otherwise, set as None, and we will propagate down from the groups.
                # if (GppProgramProvider._ObsKeys.TOO_OVERRIDE_RAPID in data and
                #         data[GppProgramProvider._ObsKeys.TOO_OVERRIDE_RAPID]):
                #     too_type = TooType.RAPID

            return GeminiObservation(
                id=ObservationID(obs_id),
                internal_id=internal_id,
                order=obs_num,
                title=title,
                site=site,
                status=status,
                active=active,
                priority=priority,
                setuptime_type=setuptime_type,
                acq_overhead=acq_overhead,
                obs_class=obs_class,
                targets=targets,
                guiding=guiding,
                sequence=atoms,
                constraints=constraints,
                belongs_to=program_id,
                too_type=too_type,
                preimaging=preimaging,
                band=band
            )

        except KeyError as ex:
            logger.error(f'KeyError while reading {obs_id}: {ex} (skipping).')
            print(f'KeyError while reading {obs_id}: {ex} (skipping).')

        except ValueError as ex:
            logger.error(f'ValueError while reading {obs_id}: {ex} (skipping).')
            print(f'ValueError while reading {obs_id}: {ex} (skipping).')

        except Exception as ex:
            logger.error(f'Unexpected exception while reading {obs_id}: {ex} (skipping).')
            print(f'Unexpected exception while reading {obs_id}: {ex} (skipping).')

        return None

    def parse_group(self, data: dict, program_id: ProgramID, group_id: GroupID,
                    split: bool, split_by_iterator: bool) -> Optional[Group]:
        """
        This method parses group information from GPP
        """

        # Get the group name: ROOT_GROUP_ID if the root group and otherwise the name.
        if group_id == ROOT_GROUP_ID:
            group_name = ROOT_GROUP_ID.id
            delay_min = None
            delay_max = None
            group_option = AndOption.ANYORDER
            number_to_observe = 1 # not that straightforward to get this number without iterating over elements.
            elements_list = data['elements']
            parent_id = ROOT_GROUP_ID.id # this should be None?
            parent_index = 0
        else:
            group_name = data[GppProgramProvider._GroupKeys.GROUP_NAME]

            group_delay_min = data.get(GppProgramProvider._GroupKeys.DELAY_MIN)
            group_delay_max = data[GppProgramProvider._GroupKeys.DELAY_MAX]
            if group_delay_min:
                delay_min = timedelta(seconds=data[GppProgramProvider._GroupKeys.DELAY_MIN]["seconds"])
                if group_delay_max:
                    delay_max = timedelta(seconds=data[GppProgramProvider._GroupKeys.DELAY_MAX]["seconds"])
                else:
                    delay_max = None
            else:
                delay_min = None
                delay_max = None

            # there can be empty groups for Calibrations, lucupy forces to have at least one
            # so we need to address that better.
            number_to_observe = data.get(GppProgramProvider._GroupKeys.NUM_TO_OBSERVE) or 1


            # Set group_option from Ordered
            ordered = data[GppProgramProvider._GroupKeys.ORDERED]
            # print(f"Ordered: {ordered}")

            # Currently if delay_min is not None, delay_max will be at least delay_min
            # OR group, delays are None, number_to_observe not None, set to NONE
            # AND cadence - delays not None, number_to_observe None, ordered should be forced to True
            # AND conseq - delays None,  number_to_observe None
            group_option = AndOption.ANYORDER
            if delay_max is not None:
                group_option = AndOption.CUSTOM
                ordered = True
                number_to_observe = len(data[GppProgramProvider._GroupKeys.ELEMENTS])
            else:
                if number_to_observe is not None:
                    group_option = AndOption.NONE  # OR group
                else:
                    group_option = AndOption.CONSEC_ORDERED if ordered else AndOption.CONSEC_ANYORDER
                    number_to_observe = len(data[GppProgramProvider._GroupKeys.ELEMENTS])

            parent_id = GroupID(data.get(GppProgramProvider._GroupKeys.PARENT_ID))
            parent_index = data.get(GppProgramProvider._GroupKeys.PARENT_INDEX)
            elements_list = list(reversed(data[GppProgramProvider._GroupKeys.ELEMENTS]))

        # Original empty lists
        children = []
        observations = []
        obs_parent_indices = []

        # Recursively process the group elements, reversing required to get the order
        # as in Explore
        for element in elements_list:
            if element['observation']:
                elem_parent_index = element.get(GppProgramProvider._GroupKeys.PARENT_INDEX)
                obs = self.parse_observation(element['observation'], program_id=program_id, num=(0, 0),
                                             split=split, split_by_iterator=split_by_iterator)
                if obs is not None:
                    observations.append(obs)
                    obs_parent_indices.append(elem_parent_index)
            elif element['group']:
                subgroup_id = GroupID(element['group']['id'])
                subgroup = self.parse_group(element['group'], program_id, subgroup_id, split=split,
                                            split_by_iterator=split_by_iterator)
                if subgroup is not None:
                    children.append(subgroup)

        # Put all the observations in trivial AND groups and extend the children to include them.
        trivial_groups = [
            Group(
                id=GroupID(obs.id.id),
                parent_id=group_id,
                parent_index=obs_parent_indices[idx_obs],
                program_id=program_id,
                group_name=obs.title,
                number_to_observe=1,
                delay_min=None,
                delay_max=None,
                children=obs,
                group_option=AndOption.CONSEC_ORDERED)
            for idx_obs, obs in enumerate(observations)]
        children.extend(trivial_groups)
        # [children.insert(0, child) for child in trivial_groups]

        # If there are no children to observe, terminate with None
        # number_to_observe = len(children)
        if len(children) == 0:
            logger.warning(f"Program {program_id} group {group_id} has no candidate children. Skipping.")
            return None

        # Put all the observations in the one big AND group and return it.
        return Group(
            id=group_id,
            parent_id=parent_id,
            parent_index=parent_index,
            program_id=program_id,
            group_name=group_name,
            number_to_observe=number_to_observe,
            delay_min=delay_min,
            delay_max=delay_max,
            children=list(reversed(children)),  # to get the order correct
            group_option=group_option)

    def parse_time_allocation(self, data: dict, band: Band = None) -> TimeAllocation:
        """Time allocations by category and band"""
        category = TimeAccountingCode[data[GppProgramProvider._TAKeys.CATEGORY]]
        program_awarded = timedelta(hours=data[GppProgramProvider._TAKeys.AWARDED_PROG_TIME]['hours'])
        partner_awarded = timedelta(hours=0.0)

        if band is None:
            sciband = Band(int(data['science_band'][-1]))
        else:
            sciband = band

        return TimeAllocation(
            category=category,
            program_awarded=program_awarded,
            partner_awarded=partner_awarded,
            band=sciband
        )

    def parse_time_used(self, data: dict) -> TimeUsed:
        """Previously used/charged time"""
        program_used = timedelta(hours=data[GppProgramProvider._TAKeys.USED_PROG_TIME]['hours'])
        partner_used = timedelta(hours=data[GppProgramProvider._TAKeys.USED_PART_TIME]['hours'])
        not_charged = timedelta(hours=data[GppProgramProvider._TAKeys.NOT_CHARGED_TIME]['hours'])
        # ToDo: include band

        return TimeUsed(
            program_used=program_used,
            partner_used=partner_used,
            not_charged=not_charged
        )

    def parse_program(self, data: dict) -> Optional[Program]:
        """
        Parse the program-level details from the JSON data.

        1. The root group is always an AND group with any order.
        2. The scheduling groups are AND groups with any order.
        3. The organizational folders are ignored and their observations are considered top-level.
        4. Each observation goes in its own AND group of size 1 as per discussion.
        """
        internal_id = data[GppProgramProvider._ProgramKeys.INTERNAL_ID]
        # program_id = ProgramID(internal_id)
        # Uncomment below once we have the observation labels
        program_id = ProgramID(data[GppProgramProvider._ProgramKeys.ID]['label']) \
            if GppProgramProvider._ProgramKeys.ID in data.keys() else ProgramID(internal_id)

        # Initialize split variables - not used by GPP
        split = True
        split_by_iterator = False

        # Now we parse the groups.
        # root_group = GppProgramProvider._EMPTY_ROOT_GROUP
        root_group = self.parse_group(data['root'], program_id, ROOT_GROUP_ID,
                                      split=split, split_by_iterator=split_by_iterator)
        if root_group is None:
            logger.warning(f'Program {program_id} has empty root group. Skipping.')
            return None

        # Extract the semester and program type
        sem = data['proposal']['call']['semester']  # Program.Proposal.call.semester
        semester = Semester(year=int(sem[0:4]), half=SemesterHalf(sem[-1]))
        program_type = None
        gpp_prog_type = data['type']
        if gpp_prog_type in ['CALIBRATION', 'ENGINEERING']:
            prog_type = gpp_prog_type[0:3]
        elif gpp_prog_type == 'SCIENCE':
            gpp_prop_subtype = data['proposal']['type']['science_subtype']
            prog_type = self._gpp_prop_type[gpp_prop_subtype]

        program_type = ProgramTypes[prog_type]  # Program.Proposal.type.science_subtype

        if semester is None:
            logger.warning(f'Could not determine semester for program {program_id}. Skipping.')
            return None

        if program_type is None:
            logger.warning(f'Could not determine program type for program {program_id}. Skipping.')
            return None

        if prog_type == ProgramTypes.C:
            program_mode = ProgramMode['CLASSICAL']  # get from program_type
        else:
            program_mode = ProgramMode['QUEUE']  # Need a separate field to specify PV

        # Band
        band = Band(1)  # Should be an allocation by band, then the band is set for each observation
        # try:
        #     band = Band(int(data[GppProgramProvider._ProgramKeys.BAND]))
        # except ValueError:
        #     # Treat classical as Band 1, other types as Band 2
        #     if program_mode == ProgramMode.CLASSICAL:
        #         band = Band(1)
        #     else:
        #         band = Band(2)
        thesis = False
        # thesis = data[GppProgramProvider._ProgramKeys.THESIS]
        # print(f'\t program_mode = {program_mode}, band = {band}')

        # Determine the start and end date of the program.
        # NOTE that this includes the fuzzy boundaries.
        start_date = (datetime.fromisoformat(data['proposal']['call']['active']['start'].isoformat() + 'T00:00:00')
                      - Program.FUZZY_BOUNDARY)
        end_date = (datetime.fromisoformat(data['proposal']['call']['active']['end'].isoformat() + 'T00:00:00')
                    + Program.FUZZY_BOUNDARY)

        # Parse the time accounting allocation data.
        # time_act_alloc = None # Program.allocations
        time_act_alloc_data = data[GppProgramProvider._ProgramKeys.TIME_ACCOUNT_ALLOCATION]
        time_act_alloc = frozenset(self.parse_time_allocation(ta_data) for ta_data in time_act_alloc_data)

        # Previous time used - eventually loop over bands
        time_used = frozenset([self.parse_time_used(data[GppProgramProvider._ProgramKeys.TIME_CHARGE])])

        # ToOs
        too_type = None
        # too_type = TooType[data[GppProgramProvider._ProgramKeys.TOO_TYPE].upper()] if \
        #     data[GppProgramProvider._ProgramKeys.TOO_TYPE] != 'None' else None

        # Propagate the ToO type down through the root group to get to the observation.
        # GppProgramProvider._check_too_type(program_id, too_type, root_group)

        return Program(
            id=program_id,
            internal_id=internal_id,
            semester=semester,
            # band=band,
            thesis=thesis,
            mode=program_mode,
            type=program_type,
            start=start_date,
            end=end_date,
            allocated_time=time_act_alloc,
            used_time=time_used,
            root_group=root_group,
            too_type=too_type)

    @staticmethod
    def _check_too_type(program_id: ProgramID, too_type: TooType, group: Group) -> None:
        """
        Determine the validity of the TooTypes of the Observations in a Program.

        A Program with a TooType that is not None will have Observations that are the same TooType
        as the Program, unless their tooRapidOverride is set to True (in which case, the Program will
        need to have a TooType of at least RAPID).

        A Program with a TooType that is None should have all Observations with their
        tooRapidOverride set to False.

        In the context of OCS, we do not have TooTypes of INTERRUPT.

        TODO: This logic can probably be extracted from this class and moved to a general-purpose
        TODO: method as it will apply to all implementations of the API.
        """
        if too_type == TooType.INTERRUPT:
            msg = f'OCS program {program_id} has a ToO type of INTERRUPT.'
            raise ValueError(msg)

        def compatible(sub_too_type: Optional[TooType]) -> bool:
            """
            Determine if the TooType passed into this method is compatible with
            the TooType for the program.

            If the Program is not set up with a TooType, then none of its Observations can be.

            If the Program is set up with a TooType, then its Observations can either not be, or have a
            type that is as stringent or less than the Program's.
            """
            if too_type is None:
                return sub_too_type is None
            return sub_too_type is None or sub_too_type <= too_type

        def process_group(pgroup: Group):
            """
            Traverse down through the group, processing Observations and subgroups.
            """
            if isinstance(pgroup.children, Observation):
                observation: Observation = pgroup.children

                # If the observation's ToO type is None, we set it from the program.
                if observation.too_type is None:
                    observation.too_type = too_type

                # Check compatibility between the observation's ToO type and the program's ToO type.
                if not compatible(too_type):
                    nc_msg = f'Observation {observation.id} has illegal ToO type for its program.'
                    raise ValueError(nc_msg)
                observation.too_type = too_type
            else:
                for subgroup in pgroup.children:
                    process_group(subgroup)

        process_group(group)
