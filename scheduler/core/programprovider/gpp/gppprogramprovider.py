# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import calendar
import json
import zipfile
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from typing import FrozenSet, Iterable, List, Mapping, Optional, Tuple, Dict

import numpy as np
from lucupy.helpers import dmsstr2deg
from lucupy.minimodel import (AndGroup, AndOption, Atom, Band, CloudCover, Conditions, Constraints, ElevationType,
                              Group, GroupID, ImageQuality, Magnitude, MagnitudeBands, NonsiderealTarget, Observation,
                              ObservationClass, ObservationID, ObservationMode, ObservationStatus, OrGroup, Priority,
                              Program, ProgramID, ProgramMode, ProgramTypes, QAState, ResourceType,
                              ROOT_GROUP_ID, Semester, SemesterHalf, SetupTimeType, SiderealTarget, Site, SkyBackground,
                              Target, TargetTag, TargetName, TargetType, TimeAccountingCode, TimeAllocation,
                              TimingWindow, TooType, WaterVapor, Wavelength)
from lucupy.observatory.gemini.geminiobservation import GeminiObservation
from lucupy.resource_manager import ResourceManager
from lucupy.timeutils import sex2dec
from lucupy.types import ZeroTime
from scipy.signal import find_peaks

from definitions import ROOT_DIR
from scheduler.core.programprovider.abstract import ProgramProvider
from scheduler.core.sources.sources import Sources
from scheduler.services import logger_factory


__all__ = [
    'GppProgramProvider',
    'ocs_program_data',
]

logger = logger_factory.create_logger(__name__)


DEFAULT_OCS_DATA_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'programs.zip'
DEFAULT_PROGRAM_ID_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.txt'


def ocs_program_data(program_list: Optional[bytes] = None) -> Iterable[dict]:
    try:
        # Try to read the file and create a frozenset from its lines
        if program_list:
            list_file = program_list
        else:
            list_file = DEFAULT_PROGRAM_ID_PATH

        if isinstance(program_list, bytes):
            file = program_list.decode('utf-8')
            id_frozenset = frozenset(f.strip() for f in file.split('\n') if f.strip() and f.strip()[0] != '#')
        else:
            with list_file.open('r') as file:
                id_frozenset = frozenset(line.strip() for line in file if line.strip() and line.strip()[0] != '#')
    except FileNotFoundError:
        # If the file does not exist, set id_frozenset to None
        id_frozenset = None
    return read_ocs_zipfile(DEFAULT_OCS_DATA_PATH, id_frozenset)


def read_ocs_zipfile(zip_file: str | PathLike[str], program_ids: Optional[FrozenSet[str]] = None) -> Iterable[dict]:
    """
    Since for OCS we will use a collection of extracted ODB data, this is a
    convenience method to parse the data into a list of the JSON program data.
    """
    with zipfile.ZipFile(zip_file, 'r') as zf:
        for filename in zf.namelist():
            program_id = Path(filename).stem
            if program_ids is None or program_id in program_ids:
                with zf.open(filename) as f:
                    contents = f.read().decode('utf-8')
                    logger.info(f'Adding program {program_id}.')
                    yield json.loads(contents)
            else:
                logger.info(f'Skipping program {program_id} as it is not in the list.')


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

    # GPP GMOS built-in GPU name to barcode
    # ToDo: Eventually this needs to come from another source, e.g. Resource, ICTD, decide whether to use name or barcode
    _fpu_to_barcode = {'GMOS_NORTH': {
        'IFU-R':           '10000009',
        'IFU-2':           '10000007',
        'IFU-B':           '10000008',
        'focus_array_new': '10005360',
        'LONG_SLIT_0_25':  '10005351',
        'LONG_SLIT_0_50':  '10005352',
        'LONG_SLIT_0_75':  '10005353',
        'LONG_SLIT_1_00':  '10005354',
        'LONG_SLIT_1_50':  '10005355',
        'LONG_SLIT_2_00':  '10005356',
        'LONG_SLIT_5_00':  '10005357',
        'NS0.5arcsec':     '10005367',
        'NS0.75arcsec':    '10005368',
        'NS1.0arcsec':     '10005369',
        'NS1.5arcsec':     '10005358',
        'NS2.0arcsec':     '10005359',
    },
    'GMOS_SOUTH': {
        'IFU-R':           '10000009',
        'IFU-2':           '10000007',
        'IFU-B':           '10000008',
        'IFU-NS-2':        '10000010',
        'IFU-NS-B':        '10000011',
        'IFU-NS-R':        '10000012',
        'focus_array_new': '10000005',
        'LONG_SLIT_0_25':  '10005371',
        'LONG_SLIT_0_50':  '10005372',
        'LONG_SLIT_0_75':  '10005373',
        'LONG_SLIT_1_00':  '10005374',
        'LONG_SLIT_1_50':  '10005375',
        'LONG_SLIT_2_00':  '10005376',
        'LONG_SLIT_5_00':  '10005377',
        'NS0.5arcsec':     '10005388',
        'NS0.75arcsec':    '10005389',
        'NS1.0arcsec':     '10005390',
        'NS1.5arcsec':     '10005391',
        'NS2.0arcsec':     '10005392',
        'PinholeC':        '10005381',
    }
                      }
    # Note that we want to include OBSERVED observations here since this is legacy data, so most if not all observations
    # should be marked OBSERVED and we will reset this later to READY.
    _OBSERVATION_STATUSES = frozenset({ObservationStatus.READY, ObservationStatus.ONGOING, ObservationStatus.OBSERVED})

    # We contain private classes with static members for the keys in the associative
    # arrays in order to have this information defined at the top-level once.
    class _ProgramKeys:
        ID = 'programId'
        INTERNAL_ID = 'key'
        BAND = 'queueBand'
        THESIS = 'isThesis'
        MODE = 'programMode'
        TOO_TYPE = 'tooType'
        TIME_ACCOUNT_ALLOCATION = 'timeAccountAllocationCategories'
        NOTE = 'INFO'
        SCHED_NOTE = 'INFO_SCHEDNOTE'
        PROGRAM_NOTE = 'INFO_PROGRAMNOTE'

    class _NoteKeys:
        TITLE = 'title'
        TEXT = 'text'

    # Strings in notes that indicate that an observation should not be splittable.
    _NO_SPLIT_STRINGS = frozenset({"do not split",
                                   "do not interrupt",
                                   "entire sequence",
                                   "full sequence"})
    # Strings in notes that indicate that the sequence should be split by the top-most changing iterator.
    _SPLIT_BY_ITER_STRINGS = frozenset({"split by iterator",
                                        "split by sequence iterator"})

    class _TAKeys:
        CATEGORIES = 'timeAccountAllocationCategories'
        CATEGORY = 'category'
        AWARDED_PROG_TIME = 'awardedProgramTime'
        AWARDED_PART_TIME = 'awardedPartnerTime'
        USED_PROG_TIME = 'usedProgramTime'
        USED_PART_TIME = 'usedPartnerTime'

    class _GroupKeys:
        SCHEDULING_GROUP = 'GROUP_GROUP_SCHEDULING'
        ORGANIZATIONAL_FOLDER = 'GROUP_GROUP_FOLDER'
        GROUP_NAME = 'name'

    class _ObsKeys:
        KEY = 'OBSERVATION_BASIC'
        ID = 'observationId'
        INTERNAL_ID = 'key'
        QASTATE = 'qaState'
        LOG = 'obsLog'
        STATUS = 'obsStatus'
        PRIORITY = 'priority'
        TITLE = 'title'
        SEQUENCE = 'sequence'
        SETUPTIME_TYPE = 'setupTimeType'
        SETUPTIME = 'setupTime'
        OBS_CLASS = 'obsClass'
        PHASE2 = 'phase2Status'
        TOO_OVERRIDE_RAPID = 'tooOverrideRapid'

    class _TargetKeys:
        KEY = 'TELESCOPE_TARGETENV'
        BASE = 'base'
        TYPE = 'type'
        RA = 'ra'
        DEC = 'dec'
        DELTA_RA = 'deltara'
        DELTA_DEC = 'deltadec'
        EPOCH = 'epoch'
        DES = 'des'
        TAG = 'tag'
        NONSIDEREAL_OBJECT_TYPE = 'nonsiderealObjectType'
        MAGNITUDES = 'magnitudes'
        NAME = 'name'

    class _TargetEnvKeys:
        GUIDE_GROUPS = 'guideGroups'
        GUIDE_GROUP_NAME = 'name'
        GUIDE_GROUP_PRIMARY = 'primaryGroup'
        GUIDE_PROBE = 'guideProbe'
        GUIDE_PROBE_KEY = 'guideProbeKey'
        AUTO_GROUP = 'auto'
        TARGET = 'target'
        USER_TARGETS = 'userTargets'

    class _ConstraintKeys:
        KEY = 'SCHEDULING_CONDITIONS'
        CC = 'cc'
        IQ = 'iq'
        SB = 'sb'
        WV = 'wv'
        ELEVATION_TYPE = 'elevationConstraintType'
        ELEVATION_MIN = 'elevationConstraintMin'
        ELEVATION_MAX = 'elevationConstraintMax'
        TIMING_WINDOWS = 'timingWindows'

    class _AtomKeys:
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
        TIMING_WINDOWS = 'timingWindows'
        START = 'start'
        DURATION = 'duration'
        REPEAT = 'repeat'
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
                      'GMOS_NORTH': _FPUKeys.GMOSN,
                      'GMOS_SOUTH': _FPUKeys.GMOSS,
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

    def __init__(self,
                 obs_classes: FrozenSet[ObservationClass],
                 sources: Sources):
        super().__init__(obs_classes, sources)

    @staticmethod
    def parse_notes(notes: Iterable[Tuple[str, str]], search_strings: frozenset) -> bool:
        """Search note title and content strings
           Returns a boolean indicating whether the strings were found
           notes: list of note tuples,  [(title, text), (title, text),...]
           search_strings: forzenset of strings to search for"""

        # Search for any indications in the note that an observation cannot be split.
        for note in notes:
            title, content = note
            if title is not None:
                title_lower = title.lower()
                if any(s in title_lower for s in search_strings):
                    return True
            if content is not None:
                content_lower = content.lower()
                if any(s in content_lower for s in search_strings):
                    return True
        return False

    def parse_magnitude(self, data: dict) -> Magnitude:
        band = MagnitudeBands[data[GppProgramProvider._MagnitudeKeys.NAME]]
        value = data[GppProgramProvider._MagnitudeKeys.VALUE]
        return Magnitude(
            band=band,
            value=value,
            error=None)

    @staticmethod
    def _get_program_dates(program_type: ProgramTypes,
                           program_id: ProgramID,
                           note_titles: List[str]) -> Tuple[datetime, datetime]:
        """
        Find the start and end dates of a program.
        This requires special handling for FT programs, which must contain a note with the information
        at the program level with key INFO_SCHEDNOTE, INFO_PROGRAMNOTE, or INFO_NOTE.
        """
        year_str = program_id.id[3:7]
        try:
            year = int(year_str)
        except ValueError as e:
            msg = f'Illegal year specified for program {program_id}: {year_str}.'
            raise ValueError(e, msg)
        except TypeError as e:
            msg = f'Illegal type data specified for program {program_id}: {year_str}.'
            raise TypeError(e, msg)
        next_year = year + 1

        # Make sure the actual year is in the valid range.
        if year < 2000 or year > 2100:
            msg = f'Illegal year specified for program {program_id}: {year_str}.'
            raise ValueError(msg)

        half_char = program_id.id[7]
        try:
            semester = SemesterHalf(half_char)
        except ValueError as e:
            msg = f'Illegal semester specified for program {program_id}: {half_char}'
            raise ValueError(msg, e)

        # Special handling for FT programs.
        if program_type is ProgramTypes.FT:
            months_list = [x.lower() for x in calendar.month_name[1:]]

            def is_ft_note(curr_note_title: str) -> bool:
                """
                Determine if the note is a note with title information for a FT program.
                """
                if curr_note_title is None:
                    return False
                curr_note_title = curr_note_title.lower()
                return 'cycle' in curr_note_title or 'active' in curr_note_title

            def month_number(month: str, months: List[str]) -> int:
                month = month.lower()
                return [i for i, m in enumerate(months) if month in m].pop() + 1

            def parse_dates(curr_note_title: str) -> Optional[Tuple[datetime, datetime]]:
                """
                Using the information in a note title, try to determine the start and end dates
                for a FT program.

                The month information in the note title can be of the forms:
                * MON-MON-MON
                * Month, Month, and Month
                and have additional data / spacing following.

                Raises an IndexError if there are any issues in getting the months.
                """
                # Convert month data as above to a list of months.
                curr_note_months = curr_note_title.strip().replace('and ', ' ').replace('  ', ' ').replace(', ', '-'). \
                    split(' ')[-1].lower()
                month_list = [month for month in curr_note_months.split('-') if month in months_list]
                m1 = month_number(month_list[0], months_list)
                m2 = month_number(month_list[-1], months_list)

                if semester == SemesterHalf.B and m1 < 6:
                    program_start = datetime(next_year, m1, 1)
                    program_end = datetime(next_year, m2, calendar.monthrange(next_year, m2)[1])
                else:
                    program_start = datetime(year, m1, 1)
                    if m2 > m1:
                        program_end = datetime(year, m2, calendar.monthrange(year, m2)[1])
                    else:
                        program_end = datetime(next_year, m2, calendar.monthrange(next_year, m2)[1])
                return program_start, program_end

            # Find the note (if any) that contains the information.
            note_title = next(filter(is_ft_note, note_titles), None)
            if note_title is None:
                msg = f'Fast turnaround program {id} has no note containing start / end date information.'
                raise ValueError(msg)

            # Parse the month information.
            try:
                date_info = parse_dates(note_title)

            except IndexError as e:
                msg = f'Fast turnaround program {id} note title has improper form: {note_title}.'
                raise ValueError(e, msg)

            start_date, end_date = date_info

        else:
            # Not a FT program, so handle normally.
            if semester is SemesterHalf.A:
                start_date = datetime(year, 2, 1)
                end_date = datetime(year, 7, 31)
            else:
                start_date = datetime(year, 8, 1)
                end_date = datetime(next_year, 1, 31)

        # Account for the flexible boundary on programs.
        return start_date - Program.FUZZY_BOUNDARY, end_date + Program.FUZZY_BOUNDARY

    def parse_timing_window(self, data: dict) -> TimingWindow:
        start = datetime.utcfromtimestamp(data[GppProgramProvider._TimingWindowKeys.START] / 1000.0)

        duration_info = data[GppProgramProvider._TimingWindowKeys.DURATION]
        if duration_info == TimingWindow.INFINITE_DURATION_FLAG:
            duration = TimingWindow.INFINITE_DURATION
        else:
            duration = timedelta(milliseconds=duration_info)

        repeat_info = data[GppProgramProvider._TimingWindowKeys.REPEAT]
        if repeat_info == TimingWindow.FOREVER_REPEATING:
            repeat = TimingWindow.OCS_INFINITE_REPEATS
        else:
            repeat = repeat_info

        if repeat == TimingWindow.NON_REPEATING:
            period = None
        else:
            period = timedelta(milliseconds=data[GppProgramProvider._TimingWindowKeys.PERIOD])

        return TimingWindow(
            start=start,
            duration=duration,
            repeat=repeat,
            period=period)

    def parse_conditions(self, data: dict) -> Conditions:
        def to_value(cond: str) -> float:
            """
            Parse the conditions value as a float out of the string passed by the OCS program extractor.
            """
            value = cond.split('/')[0].split('%')[0]
            try:
                return 1.0 if value == 'Any' else float(value) / 100
            except (ValueError, TypeError) as e:
                # Either of these will just be a ValueError.
                msg = f'Illegal value for constraint: {value}'
                raise ValueError(e, msg)

        return Conditions(
            *[lookup(to_value(data[key])) for lookup, key in
              [(CloudCover, GppProgramProvider._ConstraintKeys.CC),
               (ImageQuality, GppProgramProvider._ConstraintKeys.IQ),
               (SkyBackground, GppProgramProvider._ConstraintKeys.SB),
               (WaterVapor, GppProgramProvider._ConstraintKeys.WV)]])

    def parse_constraints(self, data: dict) -> Constraints:
        # Get the conditions
        conditions = self.parse_conditions(data)

        # Parse the timing windows.
        timing_windows = [self.parse_timing_window(tw_data)
                          for tw_data in data[GppProgramProvider._ConstraintKeys.TIMING_WINDOWS]]

        # Get the elevation data.
        elevation_type_data = data[GppProgramProvider._ConstraintKeys.ELEVATION_TYPE].replace(' ', '_').upper()
        elevation_type = ElevationType[elevation_type_data]
        elevation_min = data[GppProgramProvider._ConstraintKeys.ELEVATION_MIN]
        elevation_max = data[GppProgramProvider._ConstraintKeys.ELEVATION_MAX]

        return Constraints(
            conditions=conditions,
            elevation_type=elevation_type,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
            timing_windows=timing_windows,
            strehl=None)

    def _parse_target_header(self, data: dict) -> Tuple[TargetName, set[Magnitude], TargetType]:
        """
        Parse the common target header information out of a target.
        """
        name = TargetName(data[GppProgramProvider._TargetKeys.NAME])
        magnitude_data = data.setdefault(GppProgramProvider._TargetKeys.MAGNITUDES, [])
        magnitudes = {self.parse_magnitude(m) for m in magnitude_data}

        target_type_data = data[GppProgramProvider._TargetKeys.TYPE].replace('-', '_').replace(' ', '_').upper()
        try:
            target_type = TargetType[target_type_data]
        except KeyError as e:
            msg = f'Target {name} has illegal type {target_type_data}.'
            raise KeyError(e, msg)

        return name, magnitudes, target_type

    def parse_sidereal_target(self, data: dict) -> SiderealTarget:
        name, magnitudes, target_type = self._parse_target_header(data)
        ra_hhmmss = data[GppProgramProvider._TargetKeys.RA]
        dec_ddmmss = data[GppProgramProvider._TargetKeys.DEC]

        # TODO: Is this the proper way to handle conversions from hms and dms?
        ra = sex2dec(ra_hhmmss, to_degree=True)
        dec = dmsstr2deg(dec_ddmmss)

        pm_ra = data.setdefault(GppProgramProvider._TargetKeys.DELTA_RA, 0.0)
        pm_dec = data.setdefault(GppProgramProvider._TargetKeys.DELTA_DEC, 0.0)
        epoch = data.setdefault(GppProgramProvider._TargetKeys.EPOCH, 2000)

        return SiderealTarget(
            name=name,
            magnitudes=frozenset(magnitudes),
            type=target_type,
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

                if instrument in GppProgramProvider.FPU_FOR_INSTRUMENT:
                    if GppProgramProvider.FPU_FOR_INSTRUMENT[instrument] in step:
                        fpu = step[GppProgramProvider.FPU_FOR_INSTRUMENT[instrument]]

                if instrument == 'GMOS-N' and fpu == 'IFU Left Slit (blue)':
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

    def parse_atoms(self, instrument, sequence: List[dict]) -> List[Atom]:
        """
        Parse the sequence by atoms
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

            # if 'GMOS' in inst:
            #     if disperser.upper == 'MIRROR':
            #         obs_mode = ObservationMode.IMAGING
            #     elif 'LONG_SLIT' in fpu:
            #         obs_mode = ObservationMode.LONGSLIT
            #     elif 'IFU' in fpu:
            #         obs_mode = ObservationMode.IFU
            #     elif 'G' in fpu:
            #         obs_mode = ObservationMode.MOS

            return obs_mode

        site = GppProgramProvider._site_for_inst[instrument]

        fpus = []
        dispersers = []
        filters = []
        wavelengths = []
        resources = frozenset()
        fpu_resources = frozenset()
        disperser_resources = frozenset()

        atoms = []
        classes = []
        guiding = []
        qa_states = []
        prev_atom_id = -1
        n_atom = 0
        instrument_resources = frozenset([self._sources.origin.resource.lookup_resource(instrument)])
        for step in sequence:
            if step['class'] != 'ACQUISITION':
                next_atom = False
                atom_id = step['atom']
                observe_class = step[GppProgramProvider._AtomKeys.OBS_CLASS]
                step_time = step[GppProgramProvider._AtomKeys.TOTAL_TIME]

                print(step['atom'], step['class'], step['type'], step['exposure'], step['duration'], step_time, \
                      step['fpu'], step['grating'], step['filter'], step['wavelength'])

                # Instrument configuration aka Resource.
                fpu, disperser, filt, wavelength = GppProgramProvider._parse_instrument_configuration(step, instrument)
                print(f"\t{fpu} {disperser} {filt} {wavelength}")

                if atom_id != prev_atom_id:
                    # Convert all the different components into Resources.
                    if 'GMOS' in instrument:
                        # Convert FPUs and dispersers to barcodes. Note that None might be contained in some of these
                        # sets, but we filter below to remove them.
                        # ToDo: decide whether to use FPU names or barcodes for resource matching
                        fpu_resources = frozenset([self._sources.origin.resource.lookup_resource(
                            GppProgramProvider._fpu_to_barcode[instrument][fpu], description=fpu)
                                                   for fpu in fpus])
                        # fpu_resources = frozenset([ResourceService.lookup_resource(fpu)
                        #                            for fpu in fpus])
                        disperser_resources = frozenset([self._sources.origin.resource.lookup_resource(disperser.split('_')[0])
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
                    # fpu_resources = frozenset([self._sources.origin.resource.fpu_to_barcode(site, fpu, instrument)
                    #                            for fpu in fpus])
                    # fpu_resources = frozenset([ResourceService.lookup_resource(fpu)
                    #                            for fpu in fpus])
                    fpu_resources = frozenset([self._sources.origin.resource.lookup_resource(
                        GppProgramProvider._fpu_to_barcode[instrument][fpu], description=fpu)
                                               for fpu in fpus])
                    disperser_resources = frozenset([self._sources.origin.resource.lookup_resource(disperser.split('_')[0])
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

        return atoms

    def parse_target(self, data: dict) -> Target:
        """
        Parse a general target - either sidereal or nonsidereal - from the supplied data.
        If we are a ToO, we don't have a target, and thus we don't have a tag. Thus, this raises a KeyError.
        """
        tag = data[GppProgramProvider._TargetKeys.TAG]
        if tag == 'sidereal':
            return self.parse_sidereal_target(data)
        elif tag == 'nonsidereal':
            return self.parse_nonsidereal_target(data)
        else:
            msg = f'Illegal target tag type: {tag}.'
            raise ValueError(msg)

    def parse_observation(self,
                          data: dict,
                          num: Tuple[Optional[int], int],
                          program_id: ProgramID,
                          split: bool,
                          split_by_iterator: bool) -> Optional[Observation]:
        """
        Right now, obs_num contains an optional organizational folder number and
        a mandatory observation number, which may overlap with others in organizational folders.

        In the current list of observations, we are parsing the data for:
        OBSERVATION_BASIC-{num[1]}
        in (if defined) GROUP_GROUP_FOLDER-{num[0]}.
        """
        # folder_num is not currently used.
        folder_num, obs_num = num

        # Check the obs_class. If it is illegal, return None.
        # At the same time, ignore inactive observations.
        obs_id = data[GppProgramProvider._ObsKeys.ID]

        try:
            active = data[GppProgramProvider._ObsKeys.PHASE2] != 'Inactive'
            if not active:
                logger.warning(f"Observation {obs_id} is inactive (skipping).")
                return None

            obs_class = ObservationClass[data[GppProgramProvider._ObsKeys.OBS_CLASS].upper()]
            if obs_class not in self._obs_classes or not active:
                logger.warning(f'Observation {obs_id} not in a specified class (skipping): {obs_class.name}.')
                return None

            # By default, assume ToOType of None unless otherwise indicated.
            too_type: Optional[TooType] = None

            internal_id = data[GppProgramProvider._ObsKeys.INTERNAL_ID]
            title = data[GppProgramProvider._ObsKeys.TITLE]
            site = Site[data[GppProgramProvider._ObsKeys.ID].split('-')[0]]
            status = ObservationStatus[data[GppProgramProvider._ObsKeys.STATUS].upper()]
            priority = Priority[data[GppProgramProvider._ObsKeys.PRIORITY].upper()]

            # If the status is not legal, terminate parsing.
            if status not in GppProgramProvider._OBSERVATION_STATUSES:
                return None

            setuptime_type = SetupTimeType[data[GppProgramProvider._ObsKeys.SETUPTIME_TYPE]]
            acq_overhead = timedelta(milliseconds=data[GppProgramProvider._ObsKeys.SETUPTIME])

            find_constraints = [data[key] for key in data.keys()
                                if key.startswith(GppProgramProvider._ConstraintKeys.KEY)]
            constraints = self.parse_constraints(find_constraints[0]) if find_constraints else None

            # TODO: Do we need this? It is being passed to the parse_atoms method.
            # TODO: We have a qaState on the Observation as well.
            qa_states = [QAState[log_entry[GppProgramProvider._ObsKeys.QASTATE].upper()] for log_entry in
                         data[GppProgramProvider._ObsKeys.LOG]]

            # Parse notes for "do not split" information if not found previously
            notes = [(data[key][GppProgramProvider._NoteKeys.TITLE], data[key][GppProgramProvider._NoteKeys.TEXT])
                     for key in data.keys() if key.startswith(GppProgramProvider._ProgramKeys.NOTE)]
            if split:
                split = not self.parse_notes(notes, GppProgramProvider._NO_SPLIT_STRINGS)
            # Parse notes for "split by interator" information if not found previously
            if not split_by_iterator:
                split_by_iterator = self.parse_notes(notes, GppProgramProvider._SPLIT_BY_ITER_STRINGS)
            # If splitting not allowed, then can't split by iterator, not splitting takes precedence
            if not split:
                split_by_iterator = False

            # print(f'\nparse_observation: {obs_id}')

            atoms = self.parse_atoms(site, data[GppProgramProvider._ObsKeys.SEQUENCE], qa_states,
                                     split=split, split_by_iterator=split_by_iterator)
            # exec_time = sum([atom.exec_time for atom in atoms], ZeroTime) + acq_overhead
            # for atom in atoms:
            #     print(f'\t\t\t {atom.id} {atom.exec_time} {atom.obs_mode} {atom.resources}')

            # Check sequence for the pre-imaging flag
            preimaging = parse_preimaging(data[GppProgramProvider._ObsKeys.SEQUENCE])

            # TODO: Should this be a list of all targets for the observation?
            targets = []

            # Get the target environment. Each observation should have exactly one, but the name will
            # not necessarily be predictable as we number them.
            guiding = {}
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

                # Get the base.
                try:
                    base = self.parse_target(target_env[GppProgramProvider._TargetKeys.BASE])
                    targets.append(base)
                except KeyError:
                    logger.warning(f"No base target found for observation {obs_id}. Using empty base target.")
                    targets.append(GppProgramProvider._EMPTY_BASE_TARGET)

                # Parse the guide stars if guide star data is supplied.
                # We are only interested in the auto guide group, or the primary guide group if there
                # is not the auto guide group.
                try:
                    guide_groups = target_env[GppProgramProvider._TargetEnvKeys.GUIDE_GROUPS]
                    auto_guide_group = [group for group in guide_groups
                                        if group[GppProgramProvider._TargetEnvKeys.GUIDE_GROUP_NAME] ==
                                        GppProgramProvider._TargetEnvKeys.AUTO_GROUP]
                    primary_guide_group = [group for group in guide_groups
                                           if group[GppProgramProvider._TargetEnvKeys.GUIDE_GROUP_PRIMARY]]

                    guide_group = None
                    if auto_guide_group:
                        if len(auto_guide_group) > 1:
                            raise ValueError(f'Multiple auto guide groups found for {obs_id}.')
                        guide_group = auto_guide_group[0]
                    elif primary_guide_group:
                        if len(primary_guide_group) > 1:
                            raise ValueError(f'Multiple primary guide groups found for {obs_id}.')
                        guide_group = primary_guide_group[0]

                    # Now we parse out the guideProbe list, which contains the information about the
                    # guide probe keys and the targets.
                    if guide_group is not None:
                        for guide_data in guide_group[GppProgramProvider._TargetEnvKeys.GUIDE_PROBE]:
                            guider = guide_data[GppProgramProvider._TargetEnvKeys.GUIDE_PROBE_KEY]
                            resource = ResourceManager().lookup_resource(rid=guider, rtype=ResourceType.WFS)
                            target = self.parse_target(guide_data[GppProgramProvider._TargetEnvKeys.TARGET])
                            guiding[resource] = target
                            targets.append(target)

                except KeyError:
                    logger.warning(f'No guide group data found for observation {obs_id}')

                # Process the user targets.
                user_targets_data = target_env.setdefault(GppProgramProvider._TargetEnvKeys.USER_TARGETS, [])
                for user_target_data in user_targets_data:
                    user_target = self.parse_target(user_target_data)
                    targets.append(user_target)

                # If the ToO override rapid setting is in place, set to RAPID.
                # Otherwise, set as None, and we will propagate down from the groups.
                if (GppProgramProvider._ObsKeys.TOO_OVERRIDE_RAPID in data and
                        data[GppProgramProvider._ObsKeys.TOO_OVERRIDE_RAPID]):
                    too_type = TooType.RAPID

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
                preimaging=preimaging
            )

        except KeyError as ex:
            logger.error(f'KeyError while reading {obs_id}: {ex} (skipping).')

        except ValueError as ex:
            logger.error(f'ValueError while reading {obs_id}: {ex} (skipping).')

        except Exception as ex:
            logger.error(f'Unexpected exception while reading {obs_id}: {ex} (skipping).')

        return None

    def parse_time_allocation(self, data: dict) -> TimeAllocation:
        category = TimeAccountingCode(data[GppProgramProvider._TAKeys.CATEGORY])
        program_awarded = timedelta(milliseconds=data[GppProgramProvider._TAKeys.AWARDED_PROG_TIME])
        partner_awarded = timedelta(milliseconds=data[GppProgramProvider._TAKeys.AWARDED_PART_TIME])
        program_used = timedelta(milliseconds=data[GppProgramProvider._TAKeys.USED_PROG_TIME])
        partner_used = timedelta(milliseconds=data[GppProgramProvider._TAKeys.USED_PART_TIME])

        return TimeAllocation(
            category=category,
            program_awarded=program_awarded,
            partner_awarded=partner_awarded,
            program_used=program_used,
            partner_used=partner_used)

    def parse_or_group(self, data: dict, program_id: ProgramID, group_id: GroupID) -> OrGroup:
        """
        There are no OR groups in the OCS, so this method simply throws a
        NotImplementedError if it is called.
        """
        raise NotImplementedError('OCS does not support OR groups.')

    def parse_and_group(self, data: dict, program_id: ProgramID, group_id: GroupID,
                        split: bool, split_by_iterator: bool) -> Optional[AndGroup]:
        """
        In the OCS, a SchedulingFolder or a program are AND groups.
        We do not allow nested groups in OCS, so this is relatively easy.

        This method expects the data from a SchedulingFolder or from the program.

        Organizational folders are ignored, so they require some special handling:
        we retrieve all the observations here that are in organizational folders and
        simply stick them in this level.
        """
        delay_min = timedelta.min
        delay_max = timedelta.max

        # Get the group name: ROOT_GROUP_ID if the root group and otherwise the name.
        if GppProgramProvider._GroupKeys.GROUP_NAME in data:
            group_name = data[GppProgramProvider._GroupKeys.GROUP_NAME]
        else:
            group_name = ROOT_GROUP_ID.id

        # Parse notes for "do not split" information if not found previously
        notes = [(data[key][GppProgramProvider._NoteKeys.TITLE], data[key][GppProgramProvider._NoteKeys.TEXT])
                 for key in data.keys() if key.startswith(GppProgramProvider._ProgramKeys.NOTE)]
        if split:
            split = not self.parse_notes(notes, GppProgramProvider._NO_SPLIT_STRINGS)
        # Parse notes for "split by interator" information if not found previously
        if not split_by_iterator:
            split_by_iterator = self.parse_notes(notes, GppProgramProvider._SPLIT_BY_ITER_STRINGS)

        # Collect all the children of this group.
        children = []

        # Parse out the scheduling groups recursively.
        scheduling_group_keys = sorted(key for key in data
                                       if key.startswith(GppProgramProvider._GroupKeys.SCHEDULING_GROUP))
        for key in scheduling_group_keys:
            subgroup_id = GroupID(key.split('-')[-1])
            subgroup = self.parse_and_group(data[key], program_id, subgroup_id, split=split,
                                            split_by_iterator=split_by_iterator)
            if subgroup is not None:
                children.append(subgroup)

        # Grab the observation data from any organizational folders.
        # We want the number of the organizational folder that the observation is in so that we can make it unique
        # since observation IDs in organizational folders may clash with top-level observation IDs or observation IDs
        # in other organizational folders.
        # org_folders contains entries of the form (organizational_folder_key, data).
        def parse_unique_obs_id(of_key: Optional[str], obs_key: str) -> Tuple[Optional[int], int]:
            """
            Generate a unique observation ID. This is done to handle observational folders.
            Convert an organizational folder key and an observation key to a hybrid key to make an observation
            in an organizational folder have a unique key.
            """
            of_num = '' if of_key is None else int(of_key.split('-')[-1])
            obs_num = int(obs_key.split('-')[-1])
            return of_num, obs_num

        # Get the top-level observation data and represent it as:
        # (None, 'OBSERVATION_BASIC-#', obs_data)
        # with the None indicating that it is not part of an organizational folder.
        top_level_obs_data_blocks = [(None, obs_key, data[obs_key]) for obs_key in data
                                     if obs_key.startswith(GppProgramProvider._ObsKeys.KEY)]

        # Get the org_folders and store as entries of form ('GROUP_GROUP_FOLDER-#', folder-data).
        # Then convert to:
        # ('GROUP_GROUP_FOLDER-#', 'OBSERVATION_BASIC-#', obs-data)
        # to be consistent with the top_level_obs_data.
        org_folders = [(key, data[key]) for key in data
                       if key.startswith(GppProgramProvider._GroupKeys.ORGANIZATIONAL_FOLDER)]
        org_folders_obs_data_blocks = [(of_key, obs_key, of[obs_key]) for of_key, of in org_folders
                                       for obs_key in of if obs_key.startswith(GppProgramProvider._ObsKeys.KEY)]

        # Combine the top-level data and the organizational folder data.
        obs_data_blocks = top_level_obs_data_blocks + org_folders_obs_data_blocks

        # Parse out all the top level observations in this group.
        # Only observations that are valid, active, and have on acceptable obs_class will be returned.
        observations = []
        for *keys, obs_data in obs_data_blocks:
            obs_id = parse_unique_obs_id(*keys)
            obs = self.parse_observation(obs_data, obs_id, program_id,
                                         split=split, split_by_iterator=split_by_iterator)
            if obs is not None:
                observations.append(obs)

        # Put all the observations in trivial AND groups and extend the children to include them.
        trivial_groups = [
            AndGroup(
                id=GroupID(obs.id.id),
                program_id=program_id,
                group_name=obs.title,
                number_to_observe=1,
                delay_min=delay_min,
                delay_max=delay_max,
                children=obs,
                group_option=AndOption.ANYORDER)
            for obs in observations]
        children.extend(trivial_groups)

        # If there are no children to observe, terminate with None
        number_to_observe = len(children)
        if number_to_observe == 0:
            logger.warning(f"Program {program_id} group {group_id} has no candidate children. Skipping.")
            return None

        # Put all the observations in the one big AND group and return it.
        return AndGroup(
            id=group_id,
            program_id=program_id,
            group_name=group_name,
            number_to_observe=number_to_observe,
            delay_min=delay_min,
            delay_max=delay_max,
            children=children,
            # TODO: Should this be ANYORDER OR CONSEC_ORDERED?
            group_option=AndOption.CONSEC_ORDERED)

    def parse_program(self, data: dict) -> Optional[Program]:
        """
        Parse the program-level details from the JSON data.

        1. The root group is always an AND group with any order.
        2. The scheduling groups are AND groups with any order.
        3. The organizational folders are ignored and their observations are considered top-level.
        4. Each observation goes in its own AND group of size 1 as per discussion.
        """
        program_id = ProgramID(data[GppProgramProvider._ProgramKeys.ID])
        internal_id = data[GppProgramProvider._ProgramKeys.INTERNAL_ID]

        # # Get all the note information as they may contain FT scheduling data comments.
        note_titles = [data[key][GppProgramProvider._NoteKeys.TITLE] for key in data.keys()
                       if key.startswith(GppProgramProvider._ProgramKeys.NOTE)]
        notes = [(data[key][GppProgramProvider._NoteKeys.TITLE], data[key][GppProgramProvider._NoteKeys.TEXT])
                 for key in data.keys() if key.startswith(GppProgramProvider._ProgramKeys.NOTE)]

        # Initialize split variable, split observations by default
        split = True
        split_by_iterator = False
        # Parse notes for "do not split" information if not found previously
        if split:
            split = not self.parse_notes(notes, GppProgramProvider._NO_SPLIT_STRINGS)
        # Parse notes for "split by interator" information if not found previously
        if not split_by_iterator:
            split_by_iterator = self.parse_notes(notes, GppProgramProvider._SPLIT_BY_ITER_STRINGS)

        # Now we parse the groups. For this, we need:
        # 1. A list of Observations at the root level.
        # 2. A list of Observations for each Scheduling Group.
        # 3. A list of Observations for each Organizational Folder.
        # We can treat (1) the same as (2) and (3) by simply passing all the JSON
        # data to the parse_and_group method.
        root_group = self.parse_and_group(data, program_id, ROOT_GROUP_ID,
                                          split=split, split_by_iterator=split_by_iterator)
        if root_group is None:
            logger.warning(f'Program {program_id} has empty root group. Skipping.')
            return None

        # Extract the semester and program type, if it can be inferred from the filename.
        # TODO: The program type may be obtainable via the ODB. Should we extract it?
        semester = None
        program_type = None
        try:
            id_split = program_id.id.split('-')
            semester_year = int(id_split[1][:4])
            semester_half = SemesterHalf[id_split[1][4]]
            semester = Semester(year=semester_year, half=semester_half)
            program_type = ProgramTypes[id_split[2]]
        except (IndexError, ValueError) as e:
            logger.warning(f'Program ID {program_id} cannot be parsed: {e}.')

        if semester is None:
            logger.warning(f'Could not determine semester for program {program_id}. Skipping.')
            return None

        if program_type is None:
            logger.warning(f'Could not determine program type for program {program_id}. Skipping.')
            return None

        program_mode = ProgramMode[data[GppProgramProvider._ProgramKeys.MODE].upper()]
        try:
            band = Band(int(data[GppProgramProvider._ProgramKeys.BAND]))
        except ValueError:
            # Treat classical as Band 1, other types as Band 2
            if program_mode == ProgramMode.CLASSICAL:
                band = Band(1)
            else:
                band = Band(2)
        thesis = data[GppProgramProvider._ProgramKeys.THESIS]
        # print(f'\t program_mode = {program_mode}, band = {band}')

        # Determine the start and end date of the program.
        # NOTE that this includes the fuzzy boundaries.
        start_date, end_date = GppProgramProvider._get_program_dates(program_type, program_id, note_titles)

        # Parse the time accounting allocation data.
        time_act_alloc_data = data[GppProgramProvider._ProgramKeys.TIME_ACCOUNT_ALLOCATION]
        time_act_alloc = frozenset(self.parse_time_allocation(ta_data) for ta_data in time_act_alloc_data)

        too_type = TooType[data[GppProgramProvider._ProgramKeys.TOO_TYPE].upper()] if \
            data[GppProgramProvider._ProgramKeys.TOO_TYPE] != 'None' else None

        # Propagate the ToO type down through the root group to get to the observation.
        GppProgramProvider._check_too_type(program_id, too_type, root_group)

        return Program(
            id=program_id,
            internal_id=internal_id,
            semester=semester,
            band=band,
            thesis=thesis,
            mode=program_mode,
            type=program_type,
            start=start_date,
            end=end_date,
            allocated_time=time_act_alloc,
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