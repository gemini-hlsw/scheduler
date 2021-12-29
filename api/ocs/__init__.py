import calendar
import json
import os
from typing import NoReturn, Tuple
 
from api.abstract import ProgramProvider
from common.minimodel import *
import api.ocs.ocs_keys as keys

class OCSProgramProvider(ProgramProvider):

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
    
    _TAKeys = keys.TAKeys()
    _MagnitudeKeys = keys.MagnitudeKeys()
    _ProgramKeys = keys.ProgramKeys()
    _NoteKeys = keys.NoteKeys()
    _GroupsKeys = keys.GroupsKeys()
    _ObsKeys = keys.ObsKeys()
    _TargetKeys = keys.TargetKeys()
    _ConstraintsKeys =  keys.ConstraintsKeys()
    _AtomKeys = keys.AtomKeys()
    _TimingWindowsKeys = keys.TimingWindowsKeys()

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
        band = MagnitudeBands[data[OCSProgramProvider._MagnitudeKeys.NAME]]
        value = data[OCSProgramProvider._MagnitudeKeys.VALUE]
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
        start = datetime.fromtimestamp(data[OCSProgramProvider._TimingWindowsKeys.START]/1000)
        duration = timedelta(milliseconds=data[OCSProgramProvider._TimingWindowsKeys.DURATION])
        repeat = data[OCSProgramProvider._TimingWindowsKeys.REPEAT]
        period = timedelta(milliseconds=data[OCSProgramProvider._TimingWindowsKeys.PERIOD]) \
            if repeat != TimingWindow.NON_REPEATING else None
        return TimingWindow(start, duration, repeat, period)

    @staticmethod
    def parse_constraints(data: dict) -> Constraints:
        # Parse the timing windows.
        timing_windows = [OCSProgramProvider.parse_timing_window(tw_data)
                          for tw_data in data[OCSProgramProvider._ConstraintsKeys.TIMING_WINDOWS]]

        # Parse the conditions.
        def strip_constraint(constraint: str) -> str:
            return constraint.split('/')[0].split('%')[0]
        conditions = [lookup[strip_constraint(data[key])] for lookup, key in
                      [(OCSProgramProvider.cc, OCSProgramProvider._ConstraintsKeys.CC),
                       (OCSProgramProvider.iq, OCSProgramProvider._ConstraintsKeys.IQ),
                       (OCSProgramProvider.sb, OCSProgramProvider._ConstraintsKeys.SB),
                       (OCSProgramProvider.wv, OCSProgramProvider._ConstraintsKeys.WV)]]

        # Get the elevation data.
        elevation_type = OCSProgramProvider.elevation_types[data[OCSProgramProvider._ConstraintsKeys.ELEVATION_TYPE]]
        elevation_min = data[OCSProgramProvider._ConstraintsKeys.ELEVATION_MIN]
        elevation_max = data[OCSProgramProvider._ConstraintsKeys.ELEVATION_MAX]

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
        base = OCSProgramProvider._TargetKeys.BASE
        magnitudes = []
        if 'magnitudes' in data[base]:
            magnitudes = [OCSProgramProvider.parse_magnitude(m) for m in data[base][OCSProgramProvider._TargetKeys.MAGNITUDES]]
        return SiderealTarget(data[base][OCSProgramProvider._TargetKeys.NAME],
                              set(magnitudes),
                              data[base][OCSProgramProvider._TargetKeys.TYPE],
                              data[base][OCSProgramProvider._TargetKeys.RA],
                              data[base][OCSProgramProvider._TargetKeys.DEC],
                              data[base][OCSProgramProvider._TargetKeys.DELTARA] if 'deltara' in data[base] else None,
                              data[base][OCSProgramProvider._TargetKeys.DELTADEC] if 'deltadec' in data[base] else None,
                              data[base][OCSProgramProvider._TargetKeys.EPOCH] if 'epoch' in data[base] else None)

    @staticmethod
    def parse_nonsidereal_target(data: dict) -> NonsiderealTarget:
        """
        TODO: Retrieve the Ephemeris data.
        """
        base = OCSProgramProvider._TargetKeys.BASE
        return NonsiderealTarget(data[base][OCSProgramProvider._TargetKeys.DES],
                                 data[base][OCSProgramProvider._TargetKeys.TAG],
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
            obs_class = step[OCSProgramProvider._AtomKeys.OBS_CLASS]
            instrument = Resource(step[OCSProgramProvider._AtomKeys.INSTRUMENT],
                                  step[OCSProgramProvider._AtomKeys.INSTRUMENT])

            # TODO: Check if this is the right wavelength.
            wavelength = float(step[OCSProgramProvider._AtomKeys.WAVELENGTH])
            observed = step[OCSProgramProvider._AtomKeys.OBSERVED]
            step_time = timedelta(milliseconds=step[OCSProgramProvider._AtomKeys.TOTAL_TIME]/1000)
            
            # Offset information
            offset_p = OCSProgramProvider._AtomKeys.OFFSET_P
            offset_q = OCSProgramProvider._AtomKeys.OFFSET_Q
            p = float(step[offset_p]) if offset_p in step.keys() else None
            q = float(step[offset_q]) if offset_q in step.keys() else None

            # Any wavelength/filter_name change is a new atom
            if atom_id == 0 or (atom_id > 0 and
                                wavelength != float(sequence[atom_id - 1][OCSProgramProvider._AtomKeys.WAVELENGTH])):
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
            if key.startswith(OCSProgramProvider._TargetKeys.KEY):
                target = None
                if data[key][OCSProgramProvider._TargetKeys.BASE][OCSProgramProvider._TargetKeys.TAG] == 'sidereal':
                    target = OCSProgramProvider.parse_sidereal_target(data[key])
                else:
                    target = OCSProgramProvider.parse_sidereal_target(data[key])
                targets.append(target)

                for guide_group in data[key]['guideGroups']:
                    
                    if type(guide_group[1]) == dict:
                        guide_stars[target.name] = OCSProgramProvider.parse_guide_star(guide_group[1])
        
        find_constraints = [data[key] for key in data.keys() if key.startswith(OCSProgramProvider._ConstraintsKeys.KEY)]
        constraints = OCSProgramProvider.parse_constraints(find_constraints[0]) if len(find_constraints) > 0 else None
                
        qa_states = [QAState[log_entry[OCSProgramProvider._ObsKeys.QASTATE].upper()] for log_entry in data[OCSProgramProvider._ObsKeys.LOG]]

        site = Site.GN if data[OCSProgramProvider._ObsKeys.ID].split('-')[0] == 'GN' else Site.GS
        status = ObservationStatus[data[OCSProgramProvider._ObsKeys.STATUS]]
        priority = Priority.HIGH if data[OCSProgramProvider._ObsKeys.PRIORITY] == 'HIGH' else (Priority.LOW if data[OCSProgramProvider._ObsKeys.PRIORITY] == 'LOW' else Priority.MEDIUM)
        atoms = OCSProgramProvider.parse_atoms(data[OCSProgramProvider._ObsKeys.SEQUENCE], qa_states)
 
        obs = Observation(data[OCSProgramProvider._ObsKeys.ID],
                          data[OCSProgramProvider._ObsKeys.INTERNAL_ID],
                          int(name.split('-')[1]),
                          data[OCSProgramProvider._ObsKeys.TITLE],
                          site,
                          status,
                          True if data[OCSProgramProvider._ObsKeys.PHASE2] != 'Inactive' else False,
                          priority,
                          None,
                          SetupTimeType[data[OCSProgramProvider._ObsKeys.SETUPTIME_TYPE]],
                          timedelta(milliseconds=data[OCSProgramProvider._ObsKeys.SETUPTIME]),
                          None,
                          None,
                          None,
                          OCSProgramProvider.obs_classes[data[OCSProgramProvider._ObsKeys.OBS_CLASS]],
                          targets,
                          guide_stars,
                          atoms,
                          constraints,
                          None)
        obs.exec_time = sum([atom.exec_time for atom in atoms], timedelta()) + obs.acq_overhead

        return obs
    
    @staticmethod
    def parse_time_allocation(json: dict) -> TimeAllocation:
        return TimeAllocation(TimeAccountingCode(json[OCSProgramProvider._TAKeys.CATEGORIES][0][OCSProgramProvider._TAKeys.CATEGORY]),
                              timedelta(milliseconds=json[OCSProgramProvider._TAKeys.AWARDED_TIME]),
                              timedelta(milliseconds=0),
                              timedelta(milliseconds=json[OCSProgramProvider._TAKeys.CATEGORIES][0][OCSProgramProvider._TAKeys.PROGRAM_TIME]),
                              timedelta(milliseconds=json[OCSProgramProvider._TAKeys.CATEGORIES][0][OCSProgramProvider._TAKeys.PARTNER_TIME]))

    @staticmethod
    def parse_or_group(json: dict) -> OrGroup:
        
        # Find nested OR groups/AND groups
        # TODO: is this correct if there are not nested groups in OCS natively

        observations = [OCSProgramProvider.parse_observation(json[key], key) for key in json.keys() if key.startswith('OBSERVATION_BASIC')]

        number_to_observe = len(observations)
        delay_max, delay_min = 0, 0 # TODO: What are these?
        or_group = OrGroup(json['key'], json['name'], number_to_observe, delay_min, delay_max, observations)
        return or_group

    @staticmethod
    def parse_guide_star(json: dict) -> Resource:

        # TODO: Maybe a GuideStart class should be in place to handle this better
        res = Resource(json['name'], json['tag'])
        return res
    
    @staticmethod
    def parse_root_group(json: dict) -> AndGroup:
        # Find nested OR groups/AND groups
        groups = [OCSProgramProvider.parse_and_group(json[key]) for key in json.keys() if key.startswith(OCSProgramProvider._GroupsKeys.KEY)]
        if any(key.startswith(OCSProgramProvider._GroupsKeys.ORGANIZATIONAL_FOLDER) for key in json.keys()):
            for key in json.keys():
                if key.startswith(OCSProgramProvider._GroupsKeys.ORGANIZATIONAL_FOLDER):
                    groups.append(OCSProgramProvider.parse_or_group(json[key]))
        num_to_observe = len(groups)
        root_group = AndGroup(None, None, num_to_observe, 0, 0, groups, AndOption.ANYORDER)
        return root_group

    @staticmethod
    def parse_and_group(json: dict) -> AndGroup:
        observations = [OCSProgramProvider.parse_observation(json[key], key) for key in json.keys() if key.startswith(OCSProgramProvider._ObsKeys.KEY)]

        number_to_observe = len(observations)
        delay_max, delay_min = 0, 0 # TODO: What are these?
        return AndGroup(json['key'], json['name'], number_to_observe, delay_min, delay_max, observations, AndOption.ANYORDER)

    @staticmethod
    def parse_program(json: dict) -> Program:
        print(OCSProgramProvider.obs_classes)
        too_type = TooType(json[OCSProgramProvider._ProgramKeys.TOO_TYPE]) if json[OCSProgramProvider._ProgramKeys.TOO_TYPE] != 'None' else None
        ta = OCSProgramProvider.parse_time_allocation(json)
        root_group = OCSProgramProvider.parse_root_group(json)
        id = json[OCSProgramProvider._ProgramKeys.ID]
        program_type = ProgramTypes[id.split('-')[2]]

        notes = [json[key] for key in json.keys() if key.startswith(OCSProgramProvider._ProgramKeys.NOTE)]

        start, end = OCSProgramProvider._get_program_dates(program_type, id, notes)
        
        return Program(id,
                       json[OCSProgramProvider._ProgramKeys.INTERNAL_ID],
                       Band(int(json[OCSProgramProvider._ProgramKeys.BAND])),
                       bool(json[OCSProgramProvider._ProgramKeys.THESIS]),
                       ProgramMode[json[OCSProgramProvider._ProgramKeys.MODE].upper()],
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
                too_type = TooType.RAPID if json[OCSProgramProvider._ObsKeys.TOO_OVERRIDE_RAPID] else program.too_type
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
    provider = OCSProgramProvider(os.path.join('..', 'data', 'programs.json'))
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
