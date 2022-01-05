import calendar
import json
from typing import NoReturn, Tuple

from astropy.coordinates import SkyCoord
import numpy as np

from api.abstract import ProgramProvider
from common.minimodel import *


class OcsProgramProvider(ProgramProvider):
    """
    A ProgramProvider that parses programs from JSON extracted from the OCS
    Observing Database.
    """
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
        ORGANIZATIONAL_FOLDER = 'ORGANIZATIONAL_FOLDER'

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
        DELTARA = 'deltara'
        DELTADEC = 'deltadec'
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

    class _TimingWindowKeys:
        TIMING_WINDOWS = 'timingWindows'
        START = 'start'
        DURATION = 'duration'
        REPEAT = 'repeat'
        PERIOD = 'period'

    class _MagnitudeKeys:
        NAME = 'name'
        VALUE = 'value'

    # TODO: ??? Why is this here???
    def __init__(self, path):
        OcsProgramProvider.path = path

    @staticmethod
    def load_program(path: str) -> dict:
        """
        Parse the program file at the given path into JSON and return it.
        TODO: Why do we have this method?
        """
        with open(path, 'r') as f:
            return json.loads(f.read())

    @staticmethod
    def parse_magnitude(data: dict) -> Magnitude:
        band = MagnitudeBands[data[OcsProgramProvider._MagnitudeKeys.NAME]]
        value = data[OcsProgramProvider._MagnitudeKeys]
        return Magnitude(band, value, None)

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
            logging.error(msg)
            raise ValueError(msg, e)
        except TypeError as e:
            msg = f'Illegal type data specified for program {prog_id}: {prog_id[3:7]}.'
            logging.error(msg)
            raise TypeError(msg, e)
        next_year = year + 1

        # Make sure the actual year is in the valid range.
        if year < 2000 or year > 2100:
            msg = f'Illegal year specified for program {prog_id}: {prog_id[3:7]}.'
            logging.error(msg)
            raise ValueError(msg)

        try:
            semester = SemesterHalf(prog_id[7])
        except ValueError as e:
            msg = f'Illegal semester specified for program {prog_id}: {prog_id[7]}'
            logging.error(msg)
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
                logging.error(msg)
                raise ValueError(msg)

            # Parse the month information.
            try:
                date_info = parse_dates(note_title)

            except IndexError as e:
                msg = f'Fast turnaround program {id} note title has improper form: {note_title}.'
                logging.error(msg)
                raise ValueError(msg, e)

            start_date, end_date = date_info

        else:
            # Not a FT program, so handle normally.
            start_date = datetime(year, 8, 1)
            end_date = datetime(next_year, 1, 31)

        return start_date, end_date

    @staticmethod
    def parse_timing_window(data: dict) -> TimingWindow:
        start = datetime.fromtimestamp(data[OcsProgramProvider._TimingWindowKeys.START] / 1000)
        duration = timedelta(milliseconds=data[OcsProgramProvider._TimingWindowKeys.DURATION])
        repeat = data[OcsProgramProvider._TimingWindowKeys.REPEAT]
        period = timedelta(milliseconds=data[OcsProgramProvider._TimingWindowKeys.PERIOD]) \
            if repeat != TimingWindow.NON_REPEATING else None
        return TimingWindow(start, duration, repeat, period)

    @staticmethod
    def parse_constraints(data: dict) -> Constraints:
        # Parse the timing windows.
        timing_windows = [OcsProgramProvider.parse_timing_window(tw_data)
                          for tw_data in data[OcsProgramProvider._ConstraintKeys.TIMING_WINDOWS]]

        def to_value(cond: str) -> float:
            """
            Parse the conditions value as a float out of the string passed by the OCS program extractor.
            """
            value = cond.split('/')[0].split('%')[0]
            try:
                return 1.0 if value == 'Any' else float(value)
            except (ValueError, TypeError) as e:
                # Either of these will just be a ValueError.
                msg = f'Illegal value for constraint: {value}'
                logging.error(msg)
                raise ValueError(msg, e)

        conditions = [lookup[to_value(data[key])] for lookup, key in
                      [(CloudCover, OcsProgramProvider._ConstraintKeys.CC),
                       (ImageQuality, OcsProgramProvider._ConstraintKeys.IQ),
                       (SkyBackground, OcsProgramProvider._ConstraintKeys.SB),
                       (WaterVapor, OcsProgramProvider._ConstraintKeys.WV)]]

        # Get the elevation data.
        elevation_type_data = data[OcsProgramProvider._ConstraintKeys.ELEVATION_TYPE].replace(' ', '_').upper()
        elevation_type = ElevationType(elevation_type_data)
        elevation_min = data[OcsProgramProvider._ConstraintKeys.ELEVATION_MIN]
        elevation_max = data[OcsProgramProvider._ConstraintKeys.ELEVATION_MAX]

        return Constraints(*conditions,
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
            target_type = TargetType(target_type_data)
        except KeyError as e:
            msg = f'Target {name} has illegal type {target_type_data}'
            logging.error(msg)
            raise KeyError(msg, e)

        return name, magnitudes, target_type

    @staticmethod
    def parse_sidereal_target(data: dict) -> SiderealTarget:
        name, magnitudes, target_type = OcsProgramProvider._parse_target_header(data)
        ra_hhmmss = data[OcsProgramProvider._TargetKeys.RA]
        dec_ddmmss = data[OcsProgramProvider._TargetKeys.DEC]
        coords = SkyCoord(ra_hhmmss, dec_ddmmss)
        pm_ra = data.setdefault(OcsProgramProvider._TargetKeys.DELTARA, 0.0)
        pm_dec = data.setdefault(OcsProgramProvider._TargetKeys.DELTADEC, 0.0)
        epoch = data.setdefault(OcsProgramProvider._TargetKeys.EPOCH, 2000)

        return SiderealTarget(
            name,
            magnitudes,
            target_type,
            coords.ra,
            coords.dec,
            pm_ra,
            pm_dec,
            epoch)

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
            name,
            magnitudes,
            target_type,
            des,
            tag,
            np.empty([]),
            np.empty([]))

    @staticmethod
    def parse_atoms(sequence: List[dict], qa_states: List[QAState]) -> List[Atom]:
        n_steps = len(sequence)
        n_abba = 0
        n_atom = 0
        atoms = []
        for atom_id, step in enumerate(sequence):
            next_atom = False
            obs_class = step[OcsProgramProvider._AtomKeys.OBS_CLASS]

            # TODO: Should the resource ID and name be the same?
            instrument = Resource(step[OcsProgramProvider._AtomKeys.INSTRUMENT],
                                  step[OcsProgramProvider._AtomKeys.INSTRUMENT])

            # TODO: Check if this is the right wavelength.
            wavelength = float(step[OcsProgramProvider._AtomKeys.WAVELENGTH])
            observed = step[OcsProgramProvider._AtomKeys.OBSERVED]
            step_time = timedelta(milliseconds=step[OcsProgramProvider._AtomKeys.TOTAL_TIME] / 1000)

            # Offset information
            offset_p = OcsProgramProvider._AtomKeys.OFFSET_P
            offset_q = OcsProgramProvider._AtomKeys.OFFSET_Q
            p = float(step[offset_p]) if offset_p in step.keys() else None
            q = float(step[offset_q]) if offset_q in step.keys() else None

            # Any wavelength/filter_name change is a new atom
            if atom_id == 0 or (atom_id > 0 and
                                wavelength != float(sequence[atom_id - 1][OcsProgramProvider._AtomKeys.WAVELENGTH])):
                next_atom = True

            # Patterns:
            # AB
            # ABBA
            if q is not None and n_steps >= 4 and n_steps - atom_id > 3 and n_abba == 0:
                if (q == float(sequence[atom_id + 3][offset_q]) and
                        q != float(sequence[atom_id + 1][offset_q]) and
                        float(sequence[atom_id + 1][offset_q]) == float(sequence[atom_id + 2][offset_q])):
                    n_abba = 3
                    next_atom = True
            else:
                n_abba -= 1

            if next_atom:
                n_atom += 1
                atoms.append(Atom(
                    id=n_atom,
                    exec_time=timedelta(milliseconds=0),
                    prog_time=timedelta(milliseconds=0),
                    part_time=timedelta(milliseconds=0),
                    observed=observed,
                    qa_state=QAState.NONE,
                    guide_state=False,
                    required_resources={instrument},
                    wavelength=wavelength))

            atoms[-1].exec_time += step_time

            if 'partnerCal' in obs_class:
                atoms[-1].part_time += step_time
            else:
                atoms[-1].prog_time += step_time

            if n_atom > 0 and qa_states:
                if atom_id < len(qa_states):
                    atoms[-1].qa_state = qa_states[atom_id - 1]
                else:
                    atoms[-1].qa_state = qa_states[-1]

        return atoms

    @staticmethod
    def parse_target(data: dict) -> Target:
        """
        Parse a general target - either sidereal or nonsidereal - from the supplied data.
        """
        tag = data[OcsProgramProvider._TargetKeys.TAG]
        if tag == 'sidereal':
            return OcsProgramProvider.parse_sidereal_target(data)
        elif tag == 'nonsidereal':
            return OcsProgramProvider.parse_sidereal_target(data)
        else:
            msg = f'Illegal target tag type: {tag}'
            logging.error(msg)
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
        status = ObservationStatus[data[OcsProgramProvider._ObsKeys.STATUS]]
        active = data[OcsProgramProvider._ObsKeys.PHASE2] != 'Inactive'
        priority = Priority[data[OcsProgramProvider._ObsKeys.PRIORITY]]

        # TODO: Instrument configuration?
        instrument_configuration = None

        setuptime_type = SetupTimeType[data[OcsProgramProvider._ObsKeys.SETUPTIME_TYPE]]
        acq_overhead = timedelta(milliseconds=data[OcsProgramProvider._ObsKeys.SETUPTIME])
        obs_class = ObservationClass[data[OcsProgramProvider._ObsKeys.OBS_CLASS].upper()]

        find_constraints = [data[key] for key in data.keys() if key.startswith(OcsProgramProvider._ConstraintKeys.KEY)]
        constraints = OcsProgramProvider.parse_constraints(find_constraints[0]) if find_constraints else None

        # TODO: Do we need this? It is being passed to the parse_atoms method.
        # TODO: We have a qaState on the Observation as well.
        qa_states = [QAState[log_entry[OcsProgramProvider._ObsKeys.QASTATE].upper()] for log_entry in
                     data[OcsProgramProvider._ObsKeys.LOG]]

        atoms = OcsProgramProvider.parse_atoms(data[OcsProgramProvider._ObsKeys.SEQUENCE], qa_states)
        exec_time = sum([atom.exec_time for atom in atoms], timedelta()) + acq_overhead

        # TODO: Should this be a list of all targets for the observation?
        targets = []

        # Get the target environment. Each observation should have exactly one, but the name will
        # not necessarily be predictable as we number them.
        guiding = {}
        target_env_keys = [key for key in data.keys() if key.startswith(OcsProgramProvider._TargetKeys.KEY)]
        if len(target_env_keys) != 1:
            msg = f'Invalid target environment information found for {obs_id}.'
            logging.error(msg)
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
                    logging.error(msg)
                    raise ValueError(msg)
                guide_group = auto_guide_group[0]
            elif primary_guide_group:
                if len(primary_guide_group) > 1:
                    msg = f'Multiple primary guide groups found for {obs_id}.'
                    logging.error(msg)
                    raise ValueError(msg)
                guide_group = primary_guide_group[0]

            # Now we parse out the guideProbe list, which contains the information about the
            # guide probe keys and the targets.
            if guide_group is not None:
                for guide_data in guide_group[OcsProgramProvider._TargetEnvKeys.GUIDE_PROBE]:
                    guider = guide_data[OcsProgramProvider._TargetEnvKeys.GUIDE_PROBE_KEY]
                    resource = Resource(guider, guider, None)
                    target = OcsProgramProvider.parse_target(guide_data[OcsProgramProvider._TargetEnvKeys.TARGET])
                    guiding[resource] = target
                    targets.append(target)

        except KeyError:
            pass

        # Process the user targets.
        user_targets_data = data.setdefault(target_env[OcsProgramProvider._TargetEnvKeys.USER_TARGETS], [])
        for user_target_data in user_targets_data:
            user_target = OcsProgramProvider.parse_target(user_target_data)
            targets.append(user_target)

        # If the ToO override rapid setting is in place, set to RAPID.
        # Otherwise set as None and we will propagate down from the groups.
        if OcsProgramProvider._ObsKeys.TOO_OVERRIDE_RAPID in data:
            too_type = TooType.RAPID
        else:
            too_type = None

        return Observation(
            obs_id,
            internal_id,
            num,
            title,
            site,
            status,
            active,
            priority,
            instrument_configuration,
            setuptime_type,
            acq_overhead,
            exec_time,
            obs_class,
            targets,
            guiding,
            atoms,
            constraints,
            too_type)

    @staticmethod
    def parse_time_allocation(data: dict) -> TimeAllocation:
        category = data[OcsProgramProvider._TAKeys.CATEGORY]
        program_awarded = timedelta(milliseconds=data[OcsProgramProvider._TAKeys.AWARDED_PROG_TIME])
        partner_awarded = timedelta(milliseconds=data[OcsProgramProvider._TAKeys.AWARDED_PART_TIME])
        program_used = timedelta(milliseconds=data[OcsProgramProvider._TAKeys.USED_PROG_TIME])
        partner_used = timedelta(milliseconds=data[OcsProgramProvider._TAKeys.USED_PART_TIME])

        return TimeAllocation(
            category,
            program_awarded,
            partner_awarded,
            program_used,
            partner_used)

    @staticmethod
    def parse_or_group(data: dict) -> OrGroup:
        """
        There are no OR groups in the OCS.
        """
        raise NotImplementedError('OCS does not support OR groups.')

    @staticmethod
    def parse_and_group(data: dict, group_id: str, group_name: str) -> AndGroup:
        """
        In the OCS, a SchedulingFolder or a program are AND groups.
        We do not allow nested groups in OCS, so this is relatively easy.

        This method expects the data from a SchedulingFolder or from the program.
        Organizational folders are ignored, so we retrieve all of the observations
        here that are in organizational folders and simply stick them in this level.
        """
        delay_min = timedelta.min
        delay_max = timedelta.max

        # Parse out the scheduling groups recursively.
        scheduling_group_keys = sorted(key for key in data.keys()
                                       if key.startswith(OcsProgramProvider._GroupKeys.SCHEDULING_GROUP))
        children = [OcsProgramProvider.parse_and_group(data[key], key, 'Scheduling') for key in scheduling_group_keys]

        # Now get all the observations in this data block and any organizational folders
        # that are in this block.
        obs_data_blocks = [data] + [data[key] for key in data.keys()
                                    if key.startswith(OcsProgramProvider._GroupKeys.ORGANIZATIONAL_FOLDER)]

        for obs_data_block in obs_data_blocks:
            # We must sort on the key since this is the correct order of the observations.
            sorted_obs_keys = sorted(key for key in obs_data_block.keys()
                                     if key.startswith(OcsProgramProvider._ObsKeys.KEY))
            observations = [OcsProgramProvider.parse_observation(data[key], key) for key in sorted_obs_keys]

            # Put all of the observations in trivial AND groups.
            trivial_groups = [AndGroup(
                obs.id,
                obs.title,
                1,
                delay_min,
                delay_max,
                obs,
                AndOption.ANYORDER,
                None
            ) for obs in observations]
            children.extend(trivial_groups)

        number_to_observe = len(children)

        # Put all of the observations in the one big AND group and return it.
        return AndGroup(
            group_id,
            group_name,
            number_to_observe,
            delay_min,
            delay_max,
            children,
            AndOption.ANYORDER,
            None
        )

    @staticmethod
    def parse_program(data: dict) -> Program:
        """
        Parse the program-level details from the JSON data.

        1. The root group is always an AND group with any order.
        TODO: verify point 2 here.
        2. The scheduling groups are AND groups with any order.
        3. The organizational folders are folders with any order.
        4. Each observation goes in its own AND group of size 1 as per discussion.
        """
        # root_group = OcsProgramProvider.parse_root_group(data)
        program_id = data[OcsProgramProvider._ProgramKeys.ID]
        internal_id = data[OcsProgramProvider._ProgramKeys.INTERNAL_ID]
        band = Band(int(data[OcsProgramProvider._ProgramKeys.BAND]))
        thesis = data[OcsProgramProvider._ProgramKeys.THESIS]
        program_mode = ProgramMode[data[OcsProgramProvider._ProgramKeys.MODE].upper()]
        program_type = ProgramTypes[program_id.split('-')[2]]

        # Get all the SCHEDNOTE and PROGRAMNOTE titles as they may contain FT data.
        note_titles = [data[key][OcsProgramProvider._NoteKeys.TITLE] for key in data.keys()
                       if key.startswith(OcsProgramProvider._ProgramKeys.SCHED_NOTE)
                       or key.startswith(OcsProgramProvider._ProgramKeys.PROGRAM_NOTE)]
        start_date, end_date = OcsProgramProvider._get_program_dates(program_type, program_id, note_titles)

        # Parse the time accounting allocation data.
        time_act_alloc_data = data[OcsProgramProvider._ProgramKeys.TIME_ACCOUNT_ALLOCATION]
        time_act_alloc = set(OcsProgramProvider.parse_time_allocation(ta_data) for ta_data in time_act_alloc_data)

        # Now we parse the groups. For this, we need:
        # 1. A list of Observations at the root level.
        # 2. A list of Observations for each Scheduling Group.
        # 3. A list of Observations for each Organizational Folder.
        # We can treat (1) the same as (2) and (3) by simply passing all of the JSON
        # data to the parse_and_group method.
        root_group = OcsProgramProvider.parse_and_group(data, "Root", "Root")

        too_type = TooType(data[OcsProgramProvider._ProgramKeys.TOO_TYPE].upper()) if \
            data[OcsProgramProvider._ProgramKeys.TOO_TYPE] != 'None' else None

        # Propagate the ToO type down through the root group to get to the observation.
        OcsProgramProvider._propagate_too_type(program_id, too_type, root_group)

        return Program(
            program_id,
            internal_id,
            band,
            thesis,
            program_mode,
            program_type,
            start_date,
            end_date,
            time_act_alloc,
            root_group,
            too_type)

    @staticmethod
    def _propagate_too_type(program_id: str, too_type: TooType, group: NodeGroup) -> NoReturn:
        """
        Determine the TooTypes of the Observations in a Program.

        A Program with a TooType that is not None will have Observations that are the same TooType
        as the Program, unless their tooRapidOverride is set to True (in which case, the Program will
        need to have a TooType of at least RAPID).

        A Program with a TooType that is None should have all Observations with their
        tooRapidOverride set to False.

        In the context of OCS, we do not have TooTypes of INTERRUPT.
        """
        if too_type == TooType.INTERRUPT:
            msg = f'OCS program {program_id} has a ToO type of INTERRUPT.'
            logging.error(msg)
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
                return sub_too_type is not None
            return sub_too_type is None or sub_too_type <= too_type

        def process_group(group: NodeGroup):
            """
            Traverse down through the group, processing Observations and subgroups.
            """
            if isinstance(group.children, Observation):
                observation: Observation = group.children

                # If the observation's ToO type is None, we set it from the program.
                if observation.too_type is None:
                    observation.too_type = too_type

                # Check compatibility between the observation's ToO type and the program's ToO type.
                if not compatible(too_type):
                    nc_msg = f'Observation {observation.id} has illegal ToO type for its program.'
                    logging.error(nc_msg)
                    raise ValueError(nc_msg)
                observation.too_type = too_type
            else:
                for subgroup in group.children:
                    if isinstance(subgroup, NodeGroup):
                        node_subgroup: NodeGroup = subgroup
                        process_group(node_subgroup)
