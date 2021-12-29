import calendar
import json
import os
from typing import NoReturn, Tuple

from api import ProgramProvider
from common.minimodel import *


class JsonProvider(ProgramProvider):
    """
    A ProgramProvider that parses programs from JSON extracted from the OCS
    Observing Database.
    """
    obs_classes = {'partnerCal': ObservationClass.PARTNER_CAL,
                   'science': ObservationClass.SCIENCE,
                   'programCal': ObservationClass.PROG_CAL,
                   'acq': ObservationClass.ACQ,
                   'acqCal': ObservationClass.ACQ_CAL,
                   'dayCal': None}

    program_types = {'C': ProgramTypes.C,
                     'CAL': ProgramTypes.CAL,
                     'DD': ProgramTypes.DD,
                     'DS': ProgramTypes.DS,
                     'ENG': ProgramTypes.ENG,
                     'FT': ProgramTypes.FT,
                     'LP': ProgramTypes.LP,
                     'Q': ProgramTypes.Q,
                     'SV': ProgramTypes.SV}

    elevation_types = {
        'None': None,
        'Airmass': ElevationType.AIRMASS,
        'Hour Angle': ElevationType.HOUR_ANGLE}

    sb = {'20': SkyBackground.SB20,
          '50': SkyBackground.SB50,
          '80': SkyBackground.SB80,
          'Any': SkyBackground.SBANY}

    cc = {'50': CloudCover.CC50,
          '70': CloudCover.CC70,
          '80': CloudCover.CC80,
          'Any': CloudCover.CCANY}
    
    wv = {'20': WaterVapor.WV20,
          '50': WaterVapor.WV50,
          '80': WaterVapor.WV80, 
          'Any': WaterVapor.WVANY}

    iq = {'20': ImageQuality.IQ20,
          '70': ImageQuality.IQ70,
          '85': ImageQuality.IQ85,
          'Any': ImageQuality.IQANY}

    class _ProgramKeys(Enum):
        ID = 'programId'
        INTERNAL_ID = 'key'
        BAND = 'queueBand'
        THESIS = 'isThesis'
        MODE = 'programMode'
        TOO_TYPE = 'tooType'
        NOTE = 'INFO_SCHEDNOTE'

    class _NoteKeys(Enum):
        TITLE = 'title'
        TEXT = 'text'
        KEY = 'key'

    class _TAKeys(Enum):
        CATEGORIES = 'timeAccountAllocationCategories'
        CATEGORY = 'category'
        AWARDED_TIME = 'awardedTime'
        PROGRAM_TIME = 'programTime'
        PARTNER_TIME = 'partnerTime'
    
    class _GroupsKeys(Enum):
        KEY = 'GROUP_GROUP_SCHEDULING'
        ORGANIZATIONAL_FOLDER = 'ORGANIZATIONAL_FOLDER'
    
    class _ObsKeys(Enum):
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

    class _TargetKeys(Enum):
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

    class _ConstraintsKeys(Enum):
        KEY = 'SCHEDULING_CONDITIONS'
        CC = 'cc'
        IQ = 'iq'
        SB = 'sb'
        WV = 'wv'
        ELEVATION_TYPE = 'elevationConstraintType'
        ELEVATION_MIN = 'elevationConstraintMin'
        ELEVATION_MAX = 'elevationConstraintMax'
        TIMING_WINDOWS = 'timingWindows'
    
    class _AtomKeys(Enum):
        OBS_CLASS = 'observe:class'
        INSTRUMENT = 'instrument:instrument'
        WAVELENGTH = 'instrument:observingWavelength'
        OBSERVED = 'metadata:complete'
        TOTAL_TIME = 'totalTime'
        OFFSET_P = 'telescope:p'
        OFFSET_Q = 'telescope:q'
    
    class _TimingWindowsKeys(Enum):
        TIMING_WINDOWS = 'timingWindows'
        START = 'start'
        DURATION = 'duration'
        REPEAT = 'repeat'
        PERIOD = 'period'

    class _MagnitudeKeys(Enum):
        NAME = 'name'
        VALUE = 'value'

    def __init__(self, path):
        self.path = path

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
        band = MagnitudeBands[data[JsonProvider._MagnitudeKeys.NAME.value]]
        value = data[JsonProvider._MagnitudeKeys.VALUE.value]
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
                curr_note_months = curr_note_title.strip().replace('and ', ' ').replace('  ', ' ').replace(', ', '-').\
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
        start = datetime.fromtimestamp(data[JsonProvider._TimingWindowsKeys.START.value]/1000)
        duration = timedelta(milliseconds=data[JsonProvider._TimingWindowsKeys.DURATION.value])
        repeat = data[JsonProvider._TimingWindowsKeys.REPEAT.value]
        period = timedelta(milliseconds=data[JsonProvider._TimingWindowsKeys.PERIOD.value]) \
            if repeat != TimingWindow.NON_REPEATING else None
        return TimingWindow(start, duration, repeat, period)

    @staticmethod
    def parse_constraints(data: dict) -> Constraints:
        # Parse the timing windows.
        timing_windows = [JsonProvider.parse_timing_window(tw_data)
                          for tw_data in data[JsonProvider._ConstraintsKeys.TIMING_WINDOWS.value]]

        # Parse the conditions.
        def strip_constraint(constraint: str) -> str:
            return constraint.split('/')[0].split('%')[0]
        conditions = [lookup[strip_constraint(data[key.value])] for lookup, key in
                      [(JsonProvider.cc, JsonProvider._ConstraintsKeys.CC),
                       (JsonProvider.iq, JsonProvider._ConstraintsKeys.IQ),
                       (JsonProvider.sb, JsonProvider._ConstraintsKeys.SB),
                       (JsonProvider.wv, JsonProvider._ConstraintsKeys.WV)]]

        # Get the elevation data.
        elevation_type = JsonProvider.elevation_types[data[JsonProvider._ConstraintsKeys.ELEVATION_TYPE.value]]
        elevation_min = data[JsonProvider._ConstraintsKeys.ELEVATION_MIN.value]
        elevation_max = data[JsonProvider._ConstraintsKeys.ELEVATION_MAX.value]

        return Constraints(*conditions,
                           elevation_type=elevation_type,
                           elevation_min=elevation_min,
                           elevation_max=elevation_max,
                           timing_windows=timing_windows,
                           strehl=None)
    
    @staticmethod
    def parse_atom(data: dict, atom_id: int, qa_state: QAState) -> Atom:
        ...

    @staticmethod
    def parse_sidereal_target(data: dict) -> SiderealTarget:
        base = JsonProvider._TargetKeys.BASE.value
        magnitudes = []
        if 'magnitudes' in data[base]:
            magnitudes = [JsonProvider.parse_magnitude(m) for m in data[base][JsonProvider._TargetKeys.MAGNITUDES.value]]
        return SiderealTarget(data[base][JsonProvider._TargetKeys.NAME.value],
                              set(magnitudes),
                              data[base][JsonProvider._TargetKeys.TYPE.value],
                              data[base][JsonProvider._TargetKeys.RA.value],
                              data[base][JsonProvider._TargetKeys.DEC.value],
                              data[base][JsonProvider._TargetKeys.DELTARA.value] if 'deltara' in data[base] else None,
                              data[base][JsonProvider._TargetKeys.DELTADEC.value] if 'deltadec' in data[base] else None,
                              data[base][JsonProvider._TargetKeys.EPOCH.value] if 'epoch' in data[base] else None)

    @staticmethod
    def parse_nonsidereal_target(data: dict) -> NonsiderealTarget:
        """
        TODO: Retrieve the Ephemeris data.
        """
        base = JsonProvider._TargetKeys.BASE.value
        return NonsiderealTarget(data[base][JsonProvider._TargetKeys.DES.value],
                                 data[base][JsonProvider._TargetKeys.TAG.value],
                                 ra=None,
                                 dec=None)
    
    @staticmethod
    def parse_atoms(sequence: List[dict], qa_states: List[QAState]) -> List[Atom]:
        n_steps = len(sequence)
        n_abba = 0
        n_atom = 0
        atoms = []
        for atom_id, step in enumerate(sequence):
            next_atom = False
            obs_class = step[JsonProvider._AtomKeys.OBS_CLASS.value]
            instrument = Resource(step[JsonProvider._AtomKeys.INSTRUMENT.value],
                                  step[JsonProvider._AtomKeys.INSTRUMENT.value])

            # TODO: Check if this is the right wavelength.
            wavelength = float(step[JsonProvider._AtomKeys.WAVELENGTH.value])
            observed = step[JsonProvider._AtomKeys.OBSERVED.value]
            step_time = timedelta(milliseconds=step[JsonProvider._AtomKeys.TOTAL_TIME.value]/1000)
            
            # Offset information
            offset_p = JsonProvider._AtomKeys.OFFSET_P.value
            offset_q = JsonProvider._AtomKeys.OFFSET_Q.value
            p = float(step[offset_p]) if offset_p in step.keys() else None
            q = float(step[offset_q]) if offset_q in step.keys() else None

            # Any wavelength/filter_name change is a new atom
            if atom_id == 0 or (atom_id > 0 and
                                wavelength != float(sequence[atom_id - 1][JsonProvider._AtomKeys.WAVELENGTH.value])):
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
                atoms.append(Atom(n_atom,
                                  timedelta(milliseconds=0),
                                  timedelta(milliseconds=0),
                                  timedelta(milliseconds=0),
                                  observed,
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
    def parse_observation(data: dict, name: str) -> Observation:
        targets = []
        guide_stars = {}
        for key in data.keys():
            if key.startswith(self._TargetKeys.KEY.value):
                target = None
                if data[key][self._TargetKeys.BASE.value][self._TargetKeys.TAG.value] == 'sidereal':
                    target = self.parse_sidereal_target(data[key])
                else:
                    target = self.parse_sidereal_target(data[key])
                targets.append(target)

                for guide_group in data[key]['guideGroups']:
                    
                    if type(guide_group[1]) == dict:
                        guide_stars[target.name] = self.parse_guide_star(guide_group[1])
        
        find_constraints = [data[key] for key in data.keys() if key.startswith(self._ConstraintsKeys.KEY.value)]
        constraints = self.parse_constraints(find_constraints[0]) if len(find_constraints) > 0 else None
                
        qa_states = [QAState[log_entry[self._ObsKeys.QASTATE.value].upper()] for log_entry in data[self._ObsKeys.LOG.value]]

        site = Site.GN if data[self._ObsKeys.ID.value].split('-')[0] == 'GN' else Site.GS
        status = ObservationStatus[data[self._ObsKeys.STATUS.value]]
        priority = Priority.HIGH if data[self._ObsKeys.PRIORITY.value] == 'HIGH' else (Priority.LOW if data[self._ObsKeys.PRIORITY.value] == 'LOW' else Priority.MEDIUM)
        atoms = self.parse_atoms(data[self._ObsKeys.SEQUENCE.value], qa_states)
 
        obs = Observation(data[self._ObsKeys.ID.value],
                          data[self._ObsKeys.INTERNAL_ID.value],
                          int(name.split('-')[1]),
                          data[self._ObsKeys.TITLE.value],
                          site,
                          status,
                          True if data[self._ObsKeys.PHASE2.value] != 'Inactive' else False,
                          priority,
                          None,
                          SetupTimeType[data[self._ObsKeys.SETUPTIME_TYPE.value]],
                          timedelta(milliseconds=data[self._ObsKeys.SETUPTIME.value]),
                          None,
                          None,
                          None,
                          self.obs_classes[data[self._ObsKeys.OBS_CLASS.value]],
                          targets,
                          guide_stars,
                          atoms,
                          constraints,
                          None)
        obs.exec_time = sum([atom.exec_time for atom in atoms], timedelta()) + obs.acq_overhead

        return obs
    
    def parse_time_allocation(self, json: dict) -> TimeAllocation:
        return TimeAllocation(TimeAccountingCode(json[self._TAKeys.CATEGORIES.value][0][self._TAKeys.CATEGORY.value]),
                              timedelta(milliseconds=json[self._TAKeys.AWARDED_TIME.value]),
                              timedelta(milliseconds=0),
                              timedelta(milliseconds=json[self._TAKeys.CATEGORIES.value][0][self._TAKeys.PROGRAM_TIME.value]),
                              timedelta(milliseconds=json[self._TAKeys.CATEGORIES.value][0][self._TAKeys.PARTNER_TIME.value]))

    def parse_or_group(self, json: dict) -> OrGroup:
        
        # Find nested OR groups/AND groups
        # TODO: is this correct if there are not nested groups in OCS natively

        observations = [self.parse_observation(json[key], key) for key in json.keys() if key.startswith('OBSERVATION_BASIC')]

        number_to_observe = len(observations)
        delay_max, delay_min = 0, 0 # TODO: What are these?
        or_group = OrGroup(json['key'], json['name'], number_to_observe, delay_min, delay_max, observations)
        return or_group

    @staticmethod
    def parse_guide_star(json: dict) -> Resource:

        # TODO: Maybe a GuideStart class should be in place to handle this better
        res = Resource(json['name'], json['tag'])
        return res
    
    def parse_root_group(self, json: dict) -> AndGroup:
        # Find nested OR groups/AND groups
        groups = [self.parse_and_group(json[key]) for key in json.keys() if key.startswith(self._GroupsKeys.KEY.value)]
        if any(key.startswith(self._GroupsKeys.ORGANIZATIONAL_FOLDER.value) for key in json.keys()):
            for key in json.keys():
                if key.startswith(self._GroupsKeys.ORGANIZATIONAL_FOLDER.value):
                    groups.append(self.parse_or_group(json[key]))
        num_to_observe = len(groups)
        root_group = AndGroup(None, None, num_to_observe, 0, 0, groups, AndOption.ANYORDER)
        return root_group

    def parse_and_group(self, json: dict) -> AndGroup:
        observations = [self.parse_observation(json[key], key) for key in json.keys() if key.startswith(self._ObsKeys.KEY.value)]

        number_to_observe = len(observations)
        delay_max, delay_min = 0, 0 # TODO: What are these?
        return AndGroup(json['key'], json['name'], number_to_observe, delay_min, delay_max, observations, AndOption.ANYORDER)

    def parse_program(self, json: dict) -> Program:
        too_type = TooType(json[self._ProgramKeys.TOO_TYPE.value]) if json[self._ProgramKeys.TOO_TYPE.value] != 'None' else None
        ta = self.parse_time_allocation(json)
        root_group = self.parse_root_group(json)
        id = json[self._ProgramKeys.ID.value]
        program_type = ProgramTypes[id.split('-')[2]]

        notes = [json[key] for key in json.keys() if key.startswith(self._ProgramKeys.NOTE.value)]

        start, end = self._get_program_dates(program_type, id, notes)
        
        return Program(id,
                       json[self._ProgramKeys.INTERNAL_ID.value],
                       Band(int(json[self._ProgramKeys.BAND.value])),
                       bool(json[self._ProgramKeys.THESIS.value]),
                       ProgramMode[json[self._ProgramKeys.MODE.value].upper()],
                       program_type,
                       start,
                       end,
                       ta,
                       root_group,
                       too_type)
    
    @staticmethod
    def _calculate_too_type_for_obs(program: Program) -> NoReturn:
        """
        Determine the TooTypes of the Observations in a Program.

        A Program with a TooType that is not None will have Observations that are the same TooType
        as the Program, unless their tooRapidOverride is set to True (in which case, the Program will
        need to have a TooType of at least RAPID).

        A Program with a TooType that is None should have all Observations with their
        tooRapidOverride set to False.

        In the context of OCS, we do not have TooTypes of INTERRUPT.
        """
        if program.too_type == TooType.INTERRUPT:
            msg = f'OCS program {program.id} has a ToO type of INTERRUPT.'
            logging.error(msg)
            raise ValueError(msg)

        def compatible(too_type: Optional[TooType]) -> bool:
            """
            Determine if the TooType passed into this method is compatible with
            the TooType for the program.

            If the Program is not set up with a TooType, then none of its Observations can be.

            If the Program is set up with a TooType, then its Observations can either not be, or have a
            type that is as stringent or less than the Program's.
            """
            if program.too_type is None:
                return too_type is not None
            return too_type is None or too_type <= program.too_type

        def process_group(group: NodeGroup):
            """
            Traverse down through the group, processing Observations and subgroups.
            """
            if isinstance(group.children, Observation):
                observation: Observation = group.children
                too_type = TooType.RAPID if json[JsonProvider._ObsKeys.TOO_OVERRIDE_RAPID.value] else program.too_type
                if not compatible(too_type):
                    msg = f'Observation {observation.id} has illegal ToO type for its program.'
                    logging.error(msg)
                    raise ValueError(msg)
                observation.too_type = too_type
            else:
                for subgroup in group.children:
                    if isinstance(subgroup, NodeGroup):
                        node_subgroup: NodeGroup = subgroup
                        process_group(node_subgroup)


if __name__ == '__main__':
    provider = JsonProvider(os.path.join('..', 'data', 'programs.json'))
    json = provider.load_program(os.path.join('data', 'GN-2018B-Q-101.json'))

    program = provider.parse_program(json['PROGRAM_BASIC'])
    provider._calculate_too_type_for_obs(program)
    print(f'Program: {program.id}')

    for group in program.root_group.children:
        print(f'----- Group: {group.id}')
        for obs in group.children:
            print(f'----- ----- Observation: {obs.id}')
            for atom in obs.sequence:
                print(f'----- ----- ----- {atom}')
