import json
from datetime import datetime, timedelta

from astropy.coordinates.baseframe import _representation_deprecation
from common.api import ProgramProvider
from common.minimodel import *
from enum import Enum
from typing import NoReturn, Tuple

    
class JsonProvider(ProgramProvider):

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
        ToO = 'tooType'
        NOTES = 'INFO_SCHEDNOTE'

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

    class _TargetKeys(Enum):
        KEY = 'TELESCOPE_TARGETENV'
        BASE = 'base'
        TYPE = 'type'
        RA = 'ra'
        DEC = 'dec'
        DELTARA = 'deltara'
        DELTADEC = 'deltadec'
        EPOCH = 'epoch'
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
    
    class _MagnitudeKeys(Enum):
        NAME = 'name'
        VALUE = 'value'

    def __init__(self, path):
        self.path = path

    def load_program(self, path: str) -> dict:
        with open(path, 'r') as f:
            return json.loads(f.read())
    
    @staticmethod
    def sort_observations(obs: List[Observation]) -> List[Observation]:
        return sorted(obs, key=lambda x: x.order)
    
    @staticmethod
    def strip_constraints(constraint: str):
        return constraint.split('/')[0].split('%')[0]

    @staticmethod
    def parse_magnitude(json: dict) -> Magnitude:
        band = MagnitudeBands[json['name']]
        value = json['value']
        error = None
        return Magnitude(band, value, error)
    
    @staticmethod
    def get_program_dates(program_type: ProgramTypes, id: str, notes: list) -> Tuple[datetime, datetime]:
        start = end = None
        year = int(id[3:7])
        next_year = year + 1
        semester = id[7]
        if program_type is ProgramTypes.FT:

            for note in notes:
                if 'title' in note.keys():
                    title = note['title'].lower()
                    if 'cycle' in title or 'active' in title:
                        fields = title.strip().split(' ')
                        months = []
                        for field in fields:
                            if '-' in field:
                                months = field.split('-')
                                if len(months) == 3:
                                    break
                        if len(months) == 0:
                            for field in fields:
                                f = field.lower().strip(' ,')
                                if f in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
                                    months.append(f)
                        im1 = months[0]
                        im2 = months[-1]
                        if semester == 'B' and im1 < 6:
                            start = datetime(next_year, im1, 1)
                            end = datetime(next_year, im2, 1)
                        else:
                            start = datetime(year, im1, 1)
                            end = datetime(year, im2, 1) if im2 > im1 else datetime(next_year, im2, 1)
                        break
        else:
            if semester == 'A':
                start = datetime(int(year), 2, 1)
                end = datetime(int(year), 7, 31)
            else:
                start = datetime(int(year), 8, 1)
                end = datetime(int(next_year), 1, 31)
        return start, end

    def parse_timing_window(self, json: dict) -> TimingWindow:
        tw_arr = []
        for timing_window in json[JsonProvider._TimingWindowsKeys.TIMING_WINDOWS.value]:
            tw = TimingWindow()
            repeat = JsonProvider._TimingWindowsKeys.REPEAT.value
            tw.start = datetime.fromtimestamp(timing_window[JsonProvider._TimingWindowsKeys.START.value]/1000)
            tw.duration = timedelta(milliseconds=timing_window[JsonProvider._TimingWindowsKeys.DURATION.value])
            tw.repeat = timing_window[repeat]
            tw.period = timedelta(milliseconds=timing_window[repeat]) if timing_window[repeat] > 0 else None
            tw_arr.append(tw)
        return tw_arr
    
    def parse_constraints(self, json: dict) -> Constraints:
        return Constraints(self.cc[self.strip_constraints(json[self._ConstraintsKeys.CC.value])],
                           self.iq[self.strip_constraints(json[self._ConstraintsKeys.IQ.value])],
                           self.sb[self.strip_constraints(json[self._ConstraintsKeys.SB.value])],
                           self.wv[self.strip_constraints(json[self._ConstraintsKeys.WV.value])],
                           self.elevation_types[json[self._ConstraintsKeys.ELEVATION_TYPE.value]],
                           json[self._ConstraintsKeys.ELEVATION_MIN.value],
                           json[self._ConstraintsKeys.ELEVATION_MAX.value],
                           self.parse_timing_window(json),
                           None)
    
    @staticmethod
    def parse_atom(json: dict, id: int, qa_state: QAState) -> Observation:
        ...

    @staticmethod
    def parse_sidereal_target(json: dict) -> SiderealTarget:
        base = JsonProvider._TargetKeys.BASE.value
        magnitudes = [JsonProvider.parse_magnitude(mag) for mag in json[base][JsonProvider._TargetKeys.MAGNITUDES.value]] if 'magnitudes' in json[base] else []
        return SiderealTarget(json[base][JsonProvider._TargetKeys.NAME.value],
                              magnitudes,
                              json[base][JsonProvider._TargetKeys.TYPE.value],
                              json[base][JsonProvider._TargetKeys.RA.value],
                              json[base][JsonProvider._TargetKeys.DEC.value],
                              json[base][JsonProvider._TargetKeys.DELTARA.value] if 'deltara' in json[base] else None,
                              json[base][JsonProvider._TargetKeys.DELTADEC.value] if 'deltadec' in json[base] else None,
                              json[base][JsonProvider._TargetKeys.EPOCH.value] if 'epoch' in json[base] else None)

    @staticmethod
    def parse_nonsidereal_target(json: dict) -> NonsiderealTarget:
        base = JsonProvider._TargetKeys.BASE.value
        return NonsiderealTarget(json[base][JsonProvider._TargetKeys.DES.value],
                                 json[base][JsonProvider._TargetKeys.TAG.value],
                                 None,
                                 None)
    
    @staticmethod
    def parse_atoms(sequence: List[dict], qa_states: List[QAState]) -> List[Atom]:
       
        n_steps = len(sequence)
        n_abba = 0
        n_atom = 0
        atoms = []
        for id, step in enumerate(sequence):
            next_atom = False
            obs_class = step[JsonProvider._AtomKeys.OBS_CLASS.value]
            instrument = Resource(step[JsonProvider._AtomKeys.INSTRUMENT.value], step[JsonProvider._AtomKeys.INSTRUMENT.value])
            wavelength = float(step[JsonProvider._AtomKeys.WAVELENGTH.value])
            observed = step[JsonProvider._AtomKeys.OBSERVED.value]
            step_time = timedelta(milliseconds=step[JsonProvider._AtomKeys.TOTAL_TIME.value]/1000)
            
            #OFFSETS
            offset_p = JsonProvider._AtomKeys.OFFSET_P.value
            offset_q = JsonProvider._AtomKeys.OFFSET_Q.value
            p = float(step[offset_p]) if offset_p in step.keys() else None
            q = float(step[offset_q]) if offset_q in step.keys() else None


            # Any wavelength/filter_name change is a new atom
            if id == 0 or (id > 0 and wavelength != float(sequence[id - 1][JsonProvider._AtomKeys.WAVELENGTH.value])):
                next_atom = True

            # AB
            # ABBA
            if q is not None and n_steps >= 4 and n_steps - id > 3 and n_abba == 0:
                if (q == float(sequence[id + 3][offset_q]) and
                    q != float(sequence[id + 1][offset_q]) and
                    float(sequence[id + 1][offset_q]) == float(sequence[id + 2][offset_q])):
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
                                  None,
                                  None,
                                  instrument,
                                  wavelength))
            
            atoms[-1].exec_time += step_time

            if 'partnerCal' in obs_class:
                atoms[-1].part_time += step_time
            else:
                atoms[-1].prog_time += step_time

            if n_atom > 0 and qa_states:
                if id < len(qa_states):
                    atoms[-1].qa_state = qa_states[id-1]
                else:
                    atoms[-1].qa_state = qa_states[-1]

        return atoms
           
 
    def parse_observation(self, json: dict, name: str) -> Observation:

        targets = []
        guide_stars = {}
        for key in json.keys():
            if key.startswith(self._TargetKeys.KEY.value):
                target = None
                if json[key][self._TargetKeys.BASE.value][self._TargetKeys.TAG.value] == 'sidereal':
                    target = self.parse_sidereal_target(json[key])
                else:
                    target = self.parse_sidereal_target(json[key])
                targets.append(target)

                for guide_group in json[key]['guideGroups']:
                    
                    if type(guide_group[1]) == dict:
                        guide_stars[target.name] = self.parse_guide_star(guide_group[1])
        
        find_constraints = [json[key] for key in json.keys() if key.startswith(self._ConstraintsKeys.KEY.value)]
        constraints = self.parse_constraints(find_constraints[0]) if len(find_constraints) > 0 else None
                
        qa_states = [QAState[log_entry[self._ObsKeys.QASTATE.value].upper()] for log_entry in json[self._ObsKeys.LOG.value]]

        site = Site.GN if json[self._ObsKeys.ID.value].split('-')[0] == 'GN' else Site.GS
        status = ObservationStatus[json[self._ObsKeys.STATUS.value]]
        priority = Priority.HIGH if json[self._ObsKeys.PRIORITY.value] == 'HIGH' else (Priority.LOW if json[self._ObsKeys.PRIORITY.value] == 'LOW' else Priority.MEDIUM)
        atoms = self.parse_atoms(json[self._ObsKeys.SEQUENCE.value], qa_states)
 
        obs = Observation(json[self._ObsKeys.ID.value],
                          json[self._ObsKeys.INTERNAL_ID.value],
                          int(name.split('-')[1]),
                          json[self._ObsKeys.TITLE.value],
                          site,
                          status,
                          True if json[self._ObsKeys.PHASE2.value] != 'Inactive' else False,
                          priority,
                          None,
                          SetupTimeType[json[self._ObsKeys.SETUPTIME_TYPE.value]],
                          timedelta(milliseconds=json[self._ObsKeys.SETUPTIME.value]),
                          None,
                          None,
                          None,
                          self.obs_classes[json[self._ObsKeys.OBS_CLASS.value]],
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
        
        too_type = TooType(json[self._ProgramKeys.ToO.value]) if json[self._ProgramKeys.ToO.value] != 'None' else None
        ta = self.parse_time_allocation(json)
        root_group = self.parse_root_group(json)
        id = json[self._ProgramKeys.ID.value]
        program_type =  self.program_types[id.split('-')[2]]

        notes = [json[key] for key in json.keys() if key.startswith(self._ProgramKeys.NOTES.value)]

        start, end = self.get_program_dates(program_type, id, notes)
        
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
    def too_type_for_obs(program: Program) -> NoReturn:

        for group in program.root_group.children:
            for obs in group.children:
                if program.too_type is TooType.STANDARD:
                    obs.too_type = TooType.STANDARD
                elif program.too_type is TooType.RAPID:
                    obs.too_type = TooType.RAPID if json['tooOverrideRapid'] else TooType.STANDARD

if __name__ == '__main__':
    provider = JsonProvider('../data/programs.json')

    json = provider.load_program('./data/GN-2018B-Q-101.json')

    program = provider.parse_program(json['PROGRAM_BASIC'])
    provider.too_type_for_obs(program)    
    print(f'Program: {program.id}')

    for group in program.root_group.children:
        print(f'----- Group: {group.id}')
        for obs in group.children:
            print(f'----- ----- Observation: {obs.id}')
            for atom in obs.sequence:
                print(f'----- ----- ----- {atom}')
