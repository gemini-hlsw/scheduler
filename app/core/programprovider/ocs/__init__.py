# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import calendar
import json
import logging
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, NoReturn, Tuple, List, Optional, Mapping

import numpy as np
from lucupy.helpers import dmsstr2deg
from lucupy.minimodel import AndGroup, AndOption, Atom, Band, CloudCover, Conditions, Constraints, ElevationType, \
    Group, ImageQuality, Magnitude, MagnitudeBands, NonsiderealTarget, Observation, ObservationClass, ObservationMode, \
    ObservationStatus, OrGroup, Priority, Program, ProgramMode, ProgramTypes, QAState, Resource, Semester, \
    SemesterHalf, SetupTimeType, SiderealTarget, Site, SkyBackground, Target, TargetType, TimeAccountingCode, \
    TimeAllocation, TimingWindow, TooType, WaterVapor
from lucupy.observatory.gemini.geminiobservation import GeminiObservation
from lucupy.timeutils import sex2dec
from scipy.signal import find_peaks

from app.core.programprovider.abstract import ProgramProvider
from mock.resource import ResourceMock


def read_ocs_zipfile(zip_file: str) -> Iterable[dict]:
    """
    Since for OCS we will use a collection of extracted ODB data, this is a
    convenience method to parse the data into a list of the JSON program data.
    """
    with zipfile.ZipFile(zip_file, 'r') as zf:
        for filename in zf.namelist():
            with zf.open(filename) as f:
                contents = f.read().decode('utf-8')
                logging.info(f'Adding program {Path(filename).with_suffix("")}.')
                yield json.loads(contents)


class OcsProgramProvider(ProgramProvider):
    """
    A ProgramProvider that parses programs from JSON extracted from the OCS
    Observing Database.
    """

    _GPI_FILTER_WAVELENGTHS = {'Y': 1.05, 'J': 1.25, 'H': 1.65, 'K1': 2.05, 'K2': 2.25}
    _NIFS_FILTER_WAVELENGTHS = {'ZJ': 1.05, 'JH': 1.25, 'HK': 2.20}
    _OBSERVE_TYPES = frozenset(['FLAT', 'ARC', 'DARK', 'BIAS'])

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
        SCHED_NOTE = 'INFO_SCHEDNOTE'
        PROGRAM_NOTE = 'INFO_PROGRAMNOTE'

    class _NoteKeys:
        TITLE = 'title'

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
        OBS_CLASS = 'observe:class'
        INSTRUMENT = 'instrument:instrument'
        WAVELENGTH = 'instrument:observingWavelength'
        OBSERVED = 'metadata:complete'
        TOTAL_TIME = 'totalTime'
        OFFSET_P = 'telescope:p'
        OFFSET_Q = 'telescope:q'
        EXPOSURE_TIME = 'observe:exposureTime'
        DATA_LABEL = 'observe:dataLabel'
        COADDS = 'observe:coadds'
        FILTER = 'instrument:filter'
        DISPERSER = 'instrument:disperser'
        OBSERVE_TYPE = 'observe:observeType'

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
        GSAOI = 'instrument:utilityWheel'
        GNIRS = 'instrument:slitWidth'
        GMOSN = 'instrument:fpu'
        GPI = 'instrument:observingMode'
        F2 = 'instrument:fpu'
        GMOSS = 'instrument:fpu'
        NIRI = 'instrument:mask'
        NIFS = 'instrument:mask'
        CUSTOM = 'instrument:fpuCustomMask'

    class _InstrumentKeys:
        NAME = 'instrument:name'
        DECKER = 'instrument:acquisitionMirror'
        ACQ_MIRROR = 'instrument:acquisitionMirror'
        CROSS_DISPERSED = 'instrument:crossDispersed'

    FPU_FOR_INSTRUMENT = {'GSAOI': _FPUKeys.GSAOI,
                          'GPI': _FPUKeys.GPI,
                          'Flamingos2': _FPUKeys.F2,
                          'NIFS': _FPUKeys.NIFS,
                          'GNIRS': _FPUKeys.GNIRS,
                          'GMOS-N': _FPUKeys.GMOSN,
                          'GMOS-S': _FPUKeys.GMOSS,
                          'NIRI': _FPUKeys.NIRI}

    @staticmethod
    def parse_magnitude(data: dict) -> Magnitude:
        band = MagnitudeBands[data[OcsProgramProvider._MagnitudeKeys.NAME]]
        value = data[OcsProgramProvider._MagnitudeKeys.VALUE]
        return Magnitude(
            band=band,
            value=value,
            error=None)

    @staticmethod
    def _get_program_dates(prog_type: ProgramTypes, prog_id: str, note_titles: List[str]) -> Tuple[datetime, datetime]:
        """
        Find the start and end dates of a program.
        This requires special handling for FT programs, which must contain a note with the information
        at the program level with key INFO_SCHEDNOTE, INFO_PROGRAMNOTE, or INFO_NOTE.
        """
        try:
            year = int(prog_id[3:7])
        except ValueError as e:
            msg = f'Illegal year specified for program {prog_id}: {prog_id[3:7]}.'
            raise ValueError(e, msg)
        except TypeError as e:
            msg = f'Illegal type data specified for program {prog_id}: {prog_id[3:7]}.'
            raise TypeError(e, msg)
        next_year = year + 1

        # Make sure the actual year is in the valid range.
        if year < 2000 or year > 2100:
            msg = f'Illegal year specified for program {prog_id}: {prog_id[3:7]}.'
            raise ValueError(msg)

        try:
            semester = SemesterHalf(prog_id[7])
        except ValueError as e:
            msg = f'Illegal semester specified for program {prog_id}: {prog_id[7]}'
            raise ValueError(msg, e)

        # Special handling for FT programs.
        if prog_type is ProgramTypes.FT:
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
            start_date = datetime(year, 8, 1)
            end_date = datetime(next_year, 1, 31)

        # Account for the flexible boundary on programs.
        return start_date - Program.FUZZY_BOUNDARY, end_date + Program.FUZZY_BOUNDARY

    @staticmethod
    def parse_timing_window(data: dict) -> TimingWindow:
        start = datetime.utcfromtimestamp(data[OcsProgramProvider._TimingWindowKeys.START] / 1000.0)

        duration_info = data[OcsProgramProvider._TimingWindowKeys.DURATION]
        if duration_info == TimingWindow.INFINITE_DURATION_FLAG:
            duration = TimingWindow.INFINITE_DURATION
        else:
            duration = timedelta(milliseconds=duration_info)

        repeat_info = data[OcsProgramProvider._TimingWindowKeys.REPEAT]
        if repeat_info == TimingWindow.FOREVER_REPEATING:
            repeat = TimingWindow.OCS_INFINITE_REPEATS
        else:
            repeat = repeat_info

        if repeat == TimingWindow.NON_REPEATING:
            period = None
        else:
            period = timedelta(milliseconds=data[OcsProgramProvider._TimingWindowKeys.PERIOD])

        return TimingWindow(
            start=start,
            duration=duration,
            repeat=repeat,
            period=period)

    @staticmethod
    def parse_conditions(data: dict) -> Conditions:
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
              [(CloudCover, OcsProgramProvider._ConstraintKeys.CC),
               (ImageQuality, OcsProgramProvider._ConstraintKeys.IQ),
               (SkyBackground, OcsProgramProvider._ConstraintKeys.SB),
               (WaterVapor, OcsProgramProvider._ConstraintKeys.WV)]])

    @staticmethod
    def parse_constraints(data: dict) -> Constraints:
        # Get the conditions
        conditions = OcsProgramProvider.parse_conditions(data)

        # Parse the timing windows.
        timing_windows = [OcsProgramProvider.parse_timing_window(tw_data)
                          for tw_data in data[OcsProgramProvider._ConstraintKeys.TIMING_WINDOWS]]

        # Get the elevation data.
        elevation_type_data = data[OcsProgramProvider._ConstraintKeys.ELEVATION_TYPE].replace(' ', '_').upper()
        elevation_type = ElevationType[elevation_type_data]
        elevation_min = data[OcsProgramProvider._ConstraintKeys.ELEVATION_MIN]
        elevation_max = data[OcsProgramProvider._ConstraintKeys.ELEVATION_MAX]

        return Constraints(
            conditions=conditions,
            elevation_type=elevation_type,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
            timing_windows=timing_windows,
            strehl=None)

    @staticmethod
    def _parse_target_header(data: dict) -> Tuple[str, set[Magnitude], TargetType]:
        """
        Parse the common target header information out of a target.
        """
        name = data[OcsProgramProvider._TargetKeys.NAME]
        magnitude_data = data.setdefault(OcsProgramProvider._TargetKeys.MAGNITUDES, [])
        magnitudes = {OcsProgramProvider.parse_magnitude(m) for m in magnitude_data}

        target_type_data = data[OcsProgramProvider._TargetKeys.TYPE].replace('-', '_').replace(' ', '_').upper()
        try:
            target_type = TargetType[target_type_data]
        except KeyError as e:
            msg = f'Target {name} has illegal type {target_type_data}.'
            raise KeyError(e, msg)

        return name, magnitudes, target_type

    @staticmethod
    def parse_sidereal_target(data: dict) -> SiderealTarget:
        name, magnitudes, target_type = OcsProgramProvider._parse_target_header(data)
        ra_hhmmss = data[OcsProgramProvider._TargetKeys.RA]
        dec_ddmmss = data[OcsProgramProvider._TargetKeys.DEC]

        # TODO: Is this the proper way to handle conversions from hms and dms?
        ra = sex2dec(ra_hhmmss, todegree=True)
        # dec = sex2dec(dec_ddmmss, todegree=False)
        # ra = hmsstr2deg(ra_hhmmss)
        dec = dmsstr2deg(dec_ddmmss)

        pm_ra = data.setdefault(OcsProgramProvider._TargetKeys.DELTA_RA, 0.0)
        pm_dec = data.setdefault(OcsProgramProvider._TargetKeys.DELTA_DEC, 0.0)
        epoch = data.setdefault(OcsProgramProvider._TargetKeys.EPOCH, 2000)

        return SiderealTarget(
            name=name,
            magnitudes=magnitudes,
            type=target_type,
            ra=ra,
            dec=dec,
            pm_ra=pm_ra,
            pm_dec=pm_dec,
            epoch=epoch)

    @staticmethod
    def parse_nonsidereal_target(data: dict) -> NonsiderealTarget:
        """
        TODO: Retrieve the Ephemeris data.
        TODO: Should we be doing this here, or in the Collector?
        """
        name, magnitudes, target_type = OcsProgramProvider._parse_target_header(data)
        des = data[OcsProgramProvider._TargetKeys.DES]
        tag = data[OcsProgramProvider._TargetKeys.TAG]

        # TODO: ra and dec are last two parameters. Fill here or elsewhere?
        return NonsiderealTarget(
            name=name,
            magnitudes=magnitudes,
            type=target_type,
            des=des,
            tag=tag,
            ra=np.empty([]),
            dec=np.empty([]))

    @staticmethod
    def _parse_instrument_configuration(data: dict, instrument: str) \
            -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
        """
        A dict is return until the Instrument configuration model is created
        """

        def find_filter(filter_input: str, filter_dict: Mapping[str, float]) -> Optional[str]:
            return next(filter(lambda f: f in filter_input, filter_dict), None)

        if instrument == 'Visitor Instrument':
            instrument = data[OcsProgramProvider._InstrumentKeys.NAME].split(' ')[0]
            if instrument in ["'Alopeke", "Zorro"]:
                fpu = None
            else:
                fpu = instrument
        else:
            if instrument in OcsProgramProvider.FPU_FOR_INSTRUMENT:
                if OcsProgramProvider.FPU_FOR_INSTRUMENT[instrument] in data:
                    fpu = data[OcsProgramProvider.FPU_FOR_INSTRUMENT[instrument]]
                else:
                    # TODO: Might need to raise an exception here. Check code with science.
                    fpu = None
            else:
                raise ValueError(f'Instrument {instrument} not supported')

        if OcsProgramProvider._AtomKeys.DISPERSER in data:
            disperser = data[OcsProgramProvider._AtomKeys.DISPERSER]
        elif instrument in ['IGRINS', 'MAROON-X']:
            disperser = instrument
        else:
            disperser = None

        if instrument == 'GNIRS':
            if (data[OcsProgramProvider._InstrumentKeys.ACQ_MIRROR] == 'in'
                    and data[OcsProgramProvider._InstrumentKeys.DECKER] == 'acquisition'):
                disperser = 'mirror'
            else:
                disperser = disperser.replace('grating', '') + data[OcsProgramProvider._InstrumentKeys.CROSS_DISPERSED]
        elif instrument == 'Flamingos2' and fpu == 'FPU_NONE':
            if data['instrument:decker'] == 'IMAGING':
                disperser = data['instrument:decker']

        if OcsProgramProvider._AtomKeys.FILTER in data:
            filt = data[OcsProgramProvider._AtomKeys.FILTER]
        elif instrument == 'GPI':
            filt = find_filter(fpu, OcsProgramProvider._GPI_FILTER_WAVELENGTHS)
        else:
            if instrument == 'GNIRS':
                filt = None
            else:
                filt = 'Unknown'
        if instrument == 'NIFS' and 'Same as Disperser' in filt:
            filt = find_filter(disperser[0], OcsProgramProvider._NIFS_FILTER_WAVELENGTHS)
        wavelength = (OcsProgramProvider._GPI_FILTER_WAVELENGTHS[filt] if instrument == 'GPI'
                      else float(data[OcsProgramProvider._AtomKeys.WAVELENGTH]))

        return fpu, disperser, filt, wavelength

    @staticmethod
    def parse_atoms(site: Site, sequence: List[dict], qa_states: List[QAState]) -> List[Atom]:
        """
        Atom handling logic.
        """
        # The different configurations that make up the instrument.
        fpus = []
        dispersers = []
        filters = []
        wavelengths = []

        def guide_state(guide_step: dict) -> bool:
            return any('guideWith' in key and guide == 'guide' for key, guide in guide_step.items())

        def search_list(val, alist):
            return any(val in elem for elem in alist)

        def determine_mode(inst: str) -> ObservationMode:
            obs_mode = ObservationMode.UNKNOWN
            if search_list('GMOS', inst):
                if 'MIRROR' in dispersers:
                    obs_mode = ObservationMode.IMAGING
                elif search_list('arcsec', fpus):
                    obs_mode = ObservationMode.LONGSLIT
                elif search_list('IFU', fpus):
                    obs_mode = ObservationMode.IFU
                elif 'CUSTOM_MASK' in fpus:
                    obs_mode = ObservationMode.MOS
            elif inst in ["GSAOI", "'Alopeke", "Zorro"]:
                obs_mode = ObservationMode.IMAGING
            elif inst in ['IGRINS', 'MAROON-X']:
                obs_mode = ObservationMode.LONGSLIT
            elif inst in ['GHOST', 'MAROON-X', 'GRACES', 'Phoenix']:
                obs_mode = ObservationMode.XD
            elif inst == 'Flamingos2':
                if search_list('LONGSLIT', fpus):
                    obs_mode = ObservationMode.LONGSLIT
                if search_list('FPU_NONE', fpu) and search_list('IMAGING', dispersers):
                    obs_mode = ObservationMode.IMAGING
            elif inst == 'NIRI':
                if search_list('NONE', dispersers) and search_list('MASK_IMAGING', fpus):
                    obs_mode = ObservationMode.IMAGING
            elif inst == 'NIFS':
                obs_mode = ObservationMode.IFU
            elif inst == 'GNIRS':
                if search_list('mirror', dispersers):
                    obs_mode = ObservationMode.IMAGING
                elif search_list('XD', dispersers):
                    obs_mode = ObservationMode.XD
                else:
                    obs_mode = ObservationMode.LONGSLIT
            elif inst == 'GPI':
                if search_list('CORON', fpus):
                    obs_mode = ObservationMode.CORON
                elif search_list('NRM', fpus):
                    obs_mode = ObservationMode.NRM
                elif search_list('DIRECT', fpus):
                    obs_mode = ObservationMode.IMAGING
                else:
                    obs_mode = ObservationMode.IFU
            return obs_mode

        def autocorr_lag(x):
            """
            Test for patterns with auto-correlation
            """
            # Auto correlation
            result = np.correlate(x, x, mode='full')
            corrmax = np.max(result)
            if corrmax != 0.0:
                result = result / corrmax
            peaks, _ = find_peaks(result[result.size // 2:], height=(0, None), prominence=(0.25, None))
            return peaks[0] if len(peaks) > 0 else 0

        n_atom = 0
        atom_id = 0
        classes = []
        guiding = []
        atoms = []

        p_offsets = []
        q_offsets = []
        sky_p_offsets = []
        sky_q_offsets = []

        exposure_times = []
        coadds = []
        do_not_split = False

        # all atoms must have the same instrument
        instrument = sequence[0][OcsProgramProvider._AtomKeys.INSTRUMENT]
        for step in sequence:

            # Instrument configuration aka Resource.
            # TODO: We don't have wavelengths as Resources right now.
            fpu, disperser, filt, wavelength = OcsProgramProvider._parse_instrument_configuration(step, instrument)

            # If FPU is None, 'None', or FPU_NONE, which are effectively the same thing, we ignore.
            if fpu is not None and fpu != 'None' and fpu != 'FPU_NONE':
                fpus.append(fpu)
            dispersers.append(disperser)
            if filt and filt != 'None':
                filters.append(filt)
            wavelengths.append(wavelength)

            p = 0.0
            q = 0.0

            # Exposures on sky for dither pattern analysis
            if step[OcsProgramProvider._AtomKeys.OBSERVE_TYPE].upper() not in OcsProgramProvider._OBSERVE_TYPES:
                p = float(step[OcsProgramProvider._AtomKeys.OFFSET_P]) if OcsProgramProvider._AtomKeys.OFFSET_P in step else 0.0
                q = float(step[OcsProgramProvider._AtomKeys.OFFSET_Q]) if OcsProgramProvider._AtomKeys.OFFSET_Q in step else 0.0
                sky_p_offsets.append(p)
                sky_q_offsets.append(q)
            coadds.append(int(step[OcsProgramProvider._AtomKeys.COADDS])
                          if OcsProgramProvider._AtomKeys.COADDS in step else 1)
            exposure_times.append(step[OcsProgramProvider._AtomKeys.EXPOSURE_TIME])
            p_offsets.append(p)
            q_offsets.append(q)

        # Transform Resources.
        # TODO: For now, we focus on instruments, and GMOS FPUs and dispersers exclusively.
        instrument_resources = frozenset([ResourceMock().lookup_resource(instrument)])
        if 'GMOS' in instrument:
            # Convert FPUs and dispersers to barcodes.
            fpu_resources = frozenset([ResourceMock().fpu_to_barcode(site, fpu) for fpu in fpus])
            disperser_resources = frozenset([ResourceMock().lookup_resource(disperser.split('_')[0])
                                             for disperser in dispersers])
            resources = frozenset([r for r in fpu_resources | disperser_resources | instrument_resources])
        else:
            resources = instrument_resources

        # Remove the None values.
        resources = frozenset([res for res in resources if res is not None])

        mode = determine_mode(instrument)
        if instrument == 'GPI':
            do_not_split = True

        # Analyze sky offset patterns using auto-correlation
        # The lag is the length of any pattern, 0 means no repeating pattern
        p_lag = 0
        q_lag = 0
        if do_not_split:
            offset_lag = len(sequence)
        else:
            if len(sky_p_offsets) > 1:
                p_lag = autocorr_lag(np.array(sky_p_offsets))
            if len(sky_q_offsets) > 1:
                q_lag = autocorr_lag(np.array(sky_q_offsets))
            # Special cases
            if p_lag == 0 and q_lag == 0 and len(sky_q_offsets) == 4:
                # single ABBA pattern, which the auto-correlation won't find
                if sky_q_offsets[0] == sky_q_offsets[3] and sky_q_offsets == sky_q_offsets[2]:
                    q_lag = 4
            elif len(sky_q_offsets) == 2:
                # If only two steps, put them together, might be AB, also silly to split only two steps
                q_lag = 2

            offset_lag = q_lag
            if p_lag > 0 and p_lag != q_lag:
                offset_lag = 0
        # Group by changes in exptimes / coadds?
        exp_time_groups = False
        n_offsets = 0
        n_pattern = offset_lag
        prev = 0
        for step_id, step in enumerate(sequence):
            next_atom = False

            observe_class = step[OcsProgramProvider._AtomKeys.OBS_CLASS]
            step_time = step[OcsProgramProvider._AtomKeys.TOTAL_TIME] / 1000

            # Any wavelength/filter change is a new atom
            if step_id == 0 or (step_id > 0 and wavelengths[step_id] != wavelengths[step_id - 1]):
                next_atom = True

            # A change in exposure time or coadds is a new atom for science exposures
            if step[OcsProgramProvider._AtomKeys.OBSERVE_TYPE].upper() not in OcsProgramProvider._OBSERVE_TYPES:
                if (observe_class.upper() == ObservationClass.SCIENCE.name and step_id > 0 and
                        (exposure_times[step_id] != exposure_times[prev] or coadds[step_id] != coadds[prev])):
                    next_atom = True
                    # logging.info('Atom for exposure time change')

                # Offsets - a new offset pattern is a new atom
                if offset_lag != 0 or not exp_time_groups:
                    # For NIR imaging, need to have at least two offset positions if no repeating pattern
                    # New atom after every 2nd offset (noffsets is odd)
                    if mode is ObservationMode.IMAGING and offset_lag == 0 and all(w > 1.0 for w in wavelengths):
                        if step_id == 0:
                            n_offsets += 1
                        else:
                            if p_offsets[step_id] != p_offsets[prev] or q_offsets[step_id] != q_offsets[prev]:
                                n_offsets += 1
                        if n_offsets % 2 == 1:
                            next_atom = True
                            # logging.info('Atom for offset pattern')
                    else:
                        n_pattern -= 1
                        if n_pattern < 0:
                            next_atom = True
                            # logging.info('Atom for exposure time change')
                            n_pattern = offset_lag - 1
                prev = step_id

            # New atom entry
            if next_atom:
                # Get class, qastate, guiding for previous atom
                if n_atom > 0:
                    previous_atom = atoms[-1]
                    previous_atom.qa_state = min(qa_states, default=QAState.NONE)
                    if previous_atom.qa_state is not QAState.NONE:
                        previous_atom.observed = True
                    previous_atom.resources = resources
                    previous_atom.guide_state = any(guiding)
                    previous_atom.wavelengths = frozenset(wavelengths)

                n_atom += 1

                # Convert all the different components into Resources.
                classes = []
                guiding = []
                atoms.append(Atom(id=atom_id,
                                  exec_time=timedelta(0),
                                  prog_time=timedelta(0),
                                  part_time=timedelta(0),
                                  observed=False,
                                  qa_state=QAState.NONE,
                                  guide_state=False,
                                  resources=resources,
                                  wavelengths=frozenset(wavelengths)))

                if (step[OcsProgramProvider._AtomKeys.OBSERVE_TYPE].upper() not in OcsProgramProvider._OBSERVE_TYPES and
                        n_pattern == 0):
                    n_pattern = offset_lag
                n_offsets = 1

            # Update atom
            classes.append(observe_class)
            guiding.append(guide_state(step))

            atoms[-1].exec_time += timedelta(seconds=step_time)
            atom_id = n_atom

            # TODO: Add Observe Class enum  
            if 'partnerCal' in observe_class:
                atoms[-1].part_time = timedelta(seconds=step_time)
                atoms[-1].prog_time = timedelta(seconds=0)
            else:
                atoms[-1].part_time = timedelta(seconds=0)
                atoms[-1].prog_time = timedelta(seconds=step_time)

        if n_atom > 0:
            previous_atom = atoms[-1]
            previous_atom.qa_state = min(qa_states, default=QAState.NONE)
            if previous_atom.qa_state is not QAState.NONE:
                previous_atom.observed = True
            previous_atom.resources = resources
            previous_atom.guide_state = any(guiding)
            previous_atom.wavelengths = frozenset(wavelengths)

        return atoms

    @staticmethod
    def parse_target(data: dict) -> Target:
        """
        Parse a general target - either sidereal or nonsidereal - from the supplied data.
        If we are a ToO, we don't have a target, and thus we don't have a tag. Thus, this raises a KeyError.
        """
        tag = data[OcsProgramProvider._TargetKeys.TAG]
        if tag == 'sidereal':
            return OcsProgramProvider.parse_sidereal_target(data)
        elif tag == 'nonsidereal':
            return OcsProgramProvider.parse_nonsidereal_target(data)
        else:
            msg = f'Illegal target tag type: {tag}.'
            raise ValueError(msg)

    @staticmethod
    def parse_observation(data: dict, num: int) -> Observation:
        """
        In the current list of observations, we are parsing the data for:
        OBSERVATION_BASIC-{num}. Note that these numbers ARE in the correct order
        for scheduling groups, so we should sort on the OBSERVATION_BASIC-{num}
        keys prior to doing the parsing.
        """
        obs_id = data[OcsProgramProvider._ObsKeys.ID]
        internal_id = data[OcsProgramProvider._ObsKeys.INTERNAL_ID]
        title = data[OcsProgramProvider._ObsKeys.TITLE]
        site = Site[data[OcsProgramProvider._ObsKeys.ID].split('-')[0]]
        status = ObservationStatus[data[OcsProgramProvider._ObsKeys.STATUS].upper()]
        active = data[OcsProgramProvider._ObsKeys.PHASE2] != 'Inactive'
        priority = Priority[data[OcsProgramProvider._ObsKeys.PRIORITY].upper()]

        setuptime_type = SetupTimeType[data[OcsProgramProvider._ObsKeys.SETUPTIME_TYPE]]
        acq_overhead = timedelta(milliseconds=data[OcsProgramProvider._ObsKeys.SETUPTIME])
        obs_class = ObservationClass[data[OcsProgramProvider._ObsKeys.OBS_CLASS].upper()]

        find_constraints = [data[key] for key in data.keys() if key.startswith(OcsProgramProvider._ConstraintKeys.KEY)]
        constraints = OcsProgramProvider.parse_constraints(find_constraints[0]) if find_constraints else None

        # TODO: Do we need this? It is being passed to the parse_atoms method.
        # TODO: We have a qaState on the Observation as well.
        qa_states = [QAState[log_entry[OcsProgramProvider._ObsKeys.QASTATE].upper()] for log_entry in
                     data[OcsProgramProvider._ObsKeys.LOG]]

        atoms = OcsProgramProvider.parse_atoms(site, data[OcsProgramProvider._ObsKeys.SEQUENCE], qa_states)
        # exec_time = sum([atom.exec_time for atom in atoms], timedelta()) + acq_overhead

        # TODO: Should this be a list of all targets for the observation?
        targets = []

        # Get the target environment. Each observation should have exactly one, but the name will
        # not necessarily be predictable as we number them.
        guiding = {}
        target_env_keys = [key for key in data.keys() if key.startswith(OcsProgramProvider._TargetKeys.KEY)]
        if len(target_env_keys) != 1:
            msg = f'Invalid target environment information found for {obs_id}.'
            raise ValueError(msg)
        target_env = data[target_env_keys[0]]

        # Get the base.
        base = OcsProgramProvider.parse_target(target_env[OcsProgramProvider._TargetKeys.BASE])
        targets.append(base)

        # Parse the guide stars if guide star data is supplied.
        # We are only interested in the auto guide group, or the primary guide group if there
        # is not the auto guide group.
        try:
            guide_groups = target_env[OcsProgramProvider._TargetEnvKeys.GUIDE_GROUPS]
            auto_guide_group = [group for group in guide_groups
                                if group[OcsProgramProvider._TargetEnvKeys.GUIDE_GROUP_NAME] ==
                                OcsProgramProvider._TargetEnvKeys.AUTO_GROUP]
            primary_guide_group = [group for group in guide_groups
                                   if group[OcsProgramProvider._TargetEnvKeys.GUIDE_GROUP_PRIMARY]]

            guide_group = None
            if auto_guide_group:
                if len(auto_guide_group) > 1:
                    msg = f'Multiple auto guide groups found for {obs_id}.'
                    raise ValueError(msg)
                guide_group = auto_guide_group[0]
            elif primary_guide_group:
                if len(primary_guide_group) > 1:
                    msg = f'Multiple primary guide groups found for {obs_id}.'
                    raise ValueError(msg)
                guide_group = primary_guide_group[0]

            # Now we parse out the guideProbe list, which contains the information about the
            # guide probe keys and the targets.
            if guide_group is not None:
                for guide_data in guide_group[OcsProgramProvider._TargetEnvKeys.GUIDE_PROBE]:
                    guider = guide_data[OcsProgramProvider._TargetEnvKeys.GUIDE_PROBE_KEY]
                    # TODO: We don't have guiders as resources in ResourceMock.
                    resource = Resource(id=guider)
                    target = OcsProgramProvider.parse_target(guide_data[OcsProgramProvider._TargetEnvKeys.TARGET])
                    guiding[resource] = target
                    targets.append(target)

        except KeyError:
            pass

        # Process the user targets.
        user_targets_data = target_env.setdefault(OcsProgramProvider._TargetEnvKeys.USER_TARGETS, [])
        for user_target_data in user_targets_data:
            user_target = OcsProgramProvider.parse_target(user_target_data)
            targets.append(user_target)

        # If the ToO override rapid setting is in place, set to RAPID.
        # Otherwise, set as None, and we will propagate down from the groups.
        if OcsProgramProvider._ObsKeys.TOO_OVERRIDE_RAPID in data and \
                data[OcsProgramProvider._ObsKeys.TOO_OVERRIDE_RAPID]:
            too_type = TooType.RAPID
        else:
            too_type = None

        return GeminiObservation(
            id=obs_id,
            internal_id=internal_id,
            order=num,
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
            too_type=too_type)

    @staticmethod
    def parse_time_allocation(data: dict) -> TimeAllocation:
        category = TimeAccountingCode(data[OcsProgramProvider._TAKeys.CATEGORY])
        program_awarded = timedelta(milliseconds=data[OcsProgramProvider._TAKeys.AWARDED_PROG_TIME])
        partner_awarded = timedelta(milliseconds=data[OcsProgramProvider._TAKeys.AWARDED_PART_TIME])
        program_used = timedelta(milliseconds=data[OcsProgramProvider._TAKeys.USED_PROG_TIME])
        partner_used = timedelta(milliseconds=data[OcsProgramProvider._TAKeys.USED_PART_TIME])

        return TimeAllocation(
            category=category,
            program_awarded=program_awarded,
            partner_awarded=partner_awarded,
            program_used=program_used,
            partner_used=partner_used)

    @staticmethod
    def parse_or_group(data: dict, group_id: str) -> OrGroup:
        """
        There are no OR groups in the OCS, so this method simply throws a
        NotImplementedError if it is called.
        """
        raise NotImplementedError('OCS does not support OR groups.')

    @staticmethod
    def parse_and_group(data: dict, group_id: str) -> AndGroup:
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

        # Get the group name: Root if the root group and otherwise the name.
        if OcsProgramProvider._GroupKeys.GROUP_NAME in data:
            group_name = data[OcsProgramProvider._GroupKeys.GROUP_NAME]
        else:
            group_name = 'Root'

        # Parse out the scheduling groups recursively.
        scheduling_group_keys = sorted(key for key in data
                                       if key.startswith(OcsProgramProvider._GroupKeys.SCHEDULING_GROUP))
        children = [OcsProgramProvider.parse_and_group(data[key], key.split('-')[-1]) for key in scheduling_group_keys]

        # Grab the observation data from the complete data.
        top_level_obsdata = [(key, data[key]) for key in data
                             if key.startswith(OcsProgramProvider._ObsKeys.KEY)]

        # Grab the observation data from any organizational folders.
        org_folders = [data[key] for key in data
                       if key.startswith(OcsProgramProvider._GroupKeys.ORGANIZATIONAL_FOLDER)]
        org_folders_obsdata = [(key, of[key]) for of in org_folders
                               for key in of if key.startswith(OcsProgramProvider._ObsKeys.KEY)]

        # TODO: How do we sort the observations at the top level and in organizational
        # TODO: folders correctly? The numbering overlaps and we don't want them to intermingle.
        obs_data_blocks = top_level_obsdata + org_folders_obsdata

        # Parse out all the top level observations in this group.
        observations = [OcsProgramProvider.parse_observation(obs_data, int(obs_key.split('-')[-1]))
                        for obs_key, obs_data in obs_data_blocks]

        # Put all the observations in trivial AND groups.
        trivial_groups = [
            AndGroup(
                id=obs.id,
                group_name=obs.title,
                number_to_observe=1,
                delay_min=delay_min,
                delay_max=delay_max,
                children=obs,
                group_option=AndOption.ANYORDER)
            for obs in observations]
        children.extend(trivial_groups)

        number_to_observe = len(children)

        # Put all the observations in the one big AND group and return it.
        return AndGroup(
            id=group_id,
            group_name=group_name,
            number_to_observe=number_to_observe,
            delay_min=delay_min,
            delay_max=delay_max,
            children=children,
            # TODO: Should this be ANYORDER OR CONSEC_ORDERED?
            group_option=AndOption.CONSEC_ORDERED)

    @staticmethod
    def parse_program(data: dict) -> Program:
        """
        Parse the program-level details from the JSON data.

        1. The root group is always an AND group with any order.
        2. The scheduling groups are AND groups with any order.
        3. The organizational folders are ignored and their observations are considered top-level.
        4. Each observation goes in its own AND group of size 1 as per discussion.
        """
        program_id = data[OcsProgramProvider._ProgramKeys.ID]
        internal_id = data[OcsProgramProvider._ProgramKeys.INTERNAL_ID]

        # Extract the semester and program type, if it can be inferred from the filename.
        # TODO: The program type may be obtainable via the ODB. Should we extract it?
        semester = None
        program_type = None
        try:
            id_split = program_id.split('-')
            semester_year = int(id_split[1][:4])
            semester_half = SemesterHalf[id_split[1][4]]
            semester = Semester(year=semester_year, half=semester_half)
            program_type = ProgramTypes[id_split[2]]
        except (IndexError, ValueError) as e:
            logging.warning(f'Program ID {program_id} cannot be parsed: {e}.')

        band = Band(int(data[OcsProgramProvider._ProgramKeys.BAND]))
        thesis = data[OcsProgramProvider._ProgramKeys.THESIS]
        program_mode = ProgramMode[data[OcsProgramProvider._ProgramKeys.MODE].upper()]

        # Get all the SCHEDNOTE and PROGRAMNOTE titles as they may contain FT data.
        note_titles = [data[key][OcsProgramProvider._NoteKeys.TITLE] for key in data.keys()
                       if key.startswith(OcsProgramProvider._ProgramKeys.SCHED_NOTE)
                       or key.startswith(OcsProgramProvider._ProgramKeys.PROGRAM_NOTE)]

        # Determine the start and end date of the program.
        # NOTE that this includes the fuzzy boundaries.
        start_date, end_date = OcsProgramProvider._get_program_dates(program_type, program_id, note_titles)

        # Parse the time accounting allocation data.
        time_act_alloc_data = data[OcsProgramProvider._ProgramKeys.TIME_ACCOUNT_ALLOCATION]
        time_act_alloc = frozenset(OcsProgramProvider.parse_time_allocation(ta_data) for ta_data in time_act_alloc_data)

        # Now we parse the groups. For this, we need:
        # 1. A list of Observations at the root level.
        # 2. A list of Observations for each Scheduling Group.
        # 3. A list of Observations for each Organizational Folder.
        # We can treat (1) the same as (2) and (3) by simply passing all the JSON
        # data to the parse_and_group method.
        root_group = OcsProgramProvider.parse_and_group(data, 'Root')

        too_type = TooType[data[OcsProgramProvider._ProgramKeys.TOO_TYPE].upper()] if \
            data[OcsProgramProvider._ProgramKeys.TOO_TYPE] != 'None' else None

        # Propagate the ToO type down through the root group to get to the observation.
        OcsProgramProvider._check_too_type(program_id, too_type, root_group)

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
    def _check_too_type(program_id: str, too_type: TooType, group: Group) -> NoReturn:
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
