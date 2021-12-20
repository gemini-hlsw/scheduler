import json
from datetime import datetime, timedelta
from common.api import ProgramProvider
from common.minimodel import *


class JsonProvider(ProgramProvider):

    # TODO: To handle Enum it could be done this way or changing the Enum to accept the 
    # string value and just use the constructor. Let me know which is better.

    obs_classes = {'partnerCal': ObservationClass.PARTNER_CAL,
                   'science': ObservationClass.SCIENCE,
                   'programCal': ObservationClass.PROG_CAL,
                   'acq': ObservationClass.ACQ,
                   'acqCal': ObservationClass.ACQ_CAL,
                   'dayCal': None}
    qa_states = {'Pass': QAState.PASS,
                'Fail': QAState.FAIL,
                'Usable': QAState.USABLE,
                'Undefined': QAState.UNDEFINED}
    
    obs_status = {'NEW': ObservationStatus.NEW,
                  'INCLUDED': ObservationStatus.INCLUDED,
                  'PROPOSED': ObservationStatus.PROPOSED,
                  'APPROVED': ObservationStatus.APPROVED,
                  'FOR_REVIEW': ObservationStatus.FOR_REVIEW,
                  'READY': ObservationStatus.READY,
                  'ONGOING': ObservationStatus.ONGOING,
                  'OBSERVED': ObservationStatus.OBSERVED,
                  'INACTIVE': ObservationStatus.INACTIVE}
    
    setuptime_types = {'FULL': SetupTimeType.FULL,
                       'REACQ': SetupTimeType.REACQ,
                       'NONE': SetupTimeType.NONE}
    
    program_modes = {'Queue': ProgramMode.QUEUE,
                     'Classical': ProgramMode.CLASSICAL,
                     'PV': ProgramMode.PV}

    def __init__(self, path):
        self.path = path

    def load_program(self, path: str) -> dict:
        with open(path, 'r') as f:
            return json.loads(f.read())
    
    @staticmethod
    def sort_observations(obs: List[Observation]) -> List[Observation]:
        return sorted(obs, key=lambda x: x.order)

    @staticmethod
    def parse_magnitude(json: dict) -> Magnitude:
        band = MagnitudeBands[json['name']]
        value = json['value']
        error = None
        mag = Magnitude(band, value, error)

        return mag

    def parse_timing_window(json: dict) -> TimingWindow:
        tw_arr = []
        for timing_window in json['timingWindows']:
            tw = TimingWindow()
            tw.start = datetime.fromtimestamp(timing_window['start']/1000.0)
            tw.duration = timedelta(milliseconds=timing_window['duration'])
            tw.repeat = timing_window['repeat']
            tw.period = timedelta(milliseconds=timing_window['repeat']) if timing_window['repeat'] > 0 else None 
            tw_arr.append(tw)
        return tw_arr
    
    @staticmethod
    def parse_constraints(json: dict) -> Constraints:
        return Constraints(json['cc'],
                           json['iq'],
                           json['sb'],
                           json['wv'],
                           json['elevationConstraintType'],
                           json['elevationConstraintMin'],
                           json['elevationConstraintMax'],
                           JsonProvider.parse_timing_window(json),
                           None)
    
    @staticmethod
    def parse_atom(json: dict, id: int, qa_state: QAState) -> Observation:
        atom = Atom()

        atom.id = id
        step_time = json['totalTime']/1000
        observe_class = json['observe:class']
        part_time = 0
        prog_time = 0
        if observe_class ==  'partnerCal':
            part_time += step_time
        else:
            prog_time += step_time
        atom.exec_time = step_time
        atom.prog_time = ...
        atom.part_time = ...
        atom.observed = json['metadata:complete']
        atom.qa_state = qa_state
        atom.guide_state = ... # missing defaultGuideOption on the atom information
        atom.required_resources = [Resource('None', json['instrument:instrument'])] 
        atom.wavelength = float(json['instrument:observingWavelength'])
        return atom

    @staticmethod
    def parse_sidereal_target(json: dict) -> SiderealTarget:
        magnitudes = [JsonProvider.parse_magnitude(mag) for mag in json['base']['magnitudes']] if 'magnitudes' in json['base'] else []
        sidereal = SiderealTarget(json['base']['name'],
                                  magnitudes,
                                  json['base']['type'],
                                  json['base']['ra'],
                                  json['base']['dec'],
                                  json['base']['deltara'] if 'deltara' in json['base'] else None,
                                  json['base']['deltadec'] if 'deltadec' in json['base'] else None,
                                  json['base']['epoch'] if 'epoch' in json['base'] else None)
       
        return sidereal

    @staticmethod
    def parse_nonsidereal_target(json: dict) -> NonsiderealTarget:
        nonsidereal = NonsiderealTarget()
        nonsidereal.des = json['base']['des']
        nonsidereal.tag = json['base']['tag']
        nonsidereal.ra = None
        nonsidereal.dec = None

        return nonsidereal
    
    @staticmethod
    def parse_atoms(sequence: List[dict], qa_states: List[QAState]) -> List[Atom]:
       
        n_steps = len(sequence)
        n_abba = 0
        n_atom = 0
        atoms = []
        for id, step in enumerate(sequence):
            next_atom = False
            obs_class = step['observe:class']
            instrument = Resource(step['instrument:instrument'], step['instrument:instrument'])
            wavelength = float(step['instrument:observingWavelength'])
            observed = step['metadata:complete']
            step_time = timedelta(milliseconds=step['totalTime']/1000)
            
            #OFFSETS
            p = float(step['telescope:p']) if 'telescope:p' in step.keys() else None
            q = float(step['telescope:q']) if 'telescope:q' in step.keys() else None


            # Any wavelength/filter_name change is a new atom
            if id == 0 or (id > 0 and wavelength != float(sequence[id-1]['instrument:observingWavelength'])):
                next_atom = True

            # AB
            # ABBA
            if q is not None and n_steps >= 4 and n_steps - id > 3 and n_abba == 0:
                print(sequence[id + 3].keys())
                if (q == float(sequence[id + 3]['telescope:q']) and
                    q != float(sequence[id + 1]['telescope:q']) and
                    float(sequence[id + 1]['telescope:q']) == float(sequence[id + 2]['telescope:q'])):
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
           
    @staticmethod
    def parse_observation(json: dict, name: str ) -> Observation:

        targets = []
        guide_stars = {}
        for key in json.keys():
            if key.startswith('TELESCOPE_TARGETENV'):
                target = None
                if json[key]['base']['tag'] == 'sidereal':
                    target = JsonProvider.parse_sidereal_target(json[key])
                else:
                    target = JsonProvider.parse_sidereal_target(json[key])
                targets.append(target)

                for guide_group in json[key]['guideGroups']:
                    
                    if type(guide_group[1]) == dict:
                        guide_stars[target.name] = JsonProvider.parse_guide_star(guide_group[1])
        
        find_constraints = [json[key] for key in json.keys() if key.startswith('SCHEDULING_CONDITIONS')]
        constraints = JsonProvider.parse_constraints(find_constraints[0]) if len(find_constraints) > 0 else None
                
        qa_states = [QAState[log_entry['qaState'].upper()] for log_entry in json['obsLog']]

        site = Site.GN if json['observationId'].split('-')[0] == 'GN' else Site.GS
        status = ObservationStatus[json['obsStatus']]
        priority = Priority.HIGH if json['priority'] == 'HIGH' else (Priority.LOW if json['priority'] == 'LOW' else Priority.MEDIUM)
        print('observationId: ', json['observationId'])
        atoms = JsonProvider.parse_atoms(json['sequence'], qa_states)

        obs = Observation(json['observationId'],
                          json['key'],
                          int(name.split('-')[1]),
                          json['title'],
                          site,
                          status,
                          None,
                          priority,
                          SetupTimeType[json['setupTimeType']],
                          timedelta(milliseconds=json['setupTime']),
                          JsonProvider.obs_classes[json['obsClass']],
                          None,
                          None,
                          None,
                          targets,
                          guide_stars,
                          atoms,
                          constraints,
                          None
                          )


        # obs.active = ...

        # obs.exec_time = ... # Total time sequence  + overhead
        # obs.program_used = ...
        # obs.partner_used = ...
       
        # obs.too_type = ... # TODO: at program level

        return obs
    
    @staticmethod
    def parse_time_allocation(json: dict) -> TimeAllocation:
        #print(json['timeAccountAllocationCategories'])
        ta = TimeAllocation(TimeAccountingCode(json['timeAccountAllocationCategories'][0]['category']),
                            timedelta(milliseconds=json['awardedTime']),
                            timedelta(milliseconds=0),
                            timedelta(milliseconds=json['timeAccountAllocationCategories'][0]['programTime']),
                            timedelta(milliseconds=json['timeAccountAllocationCategories'][0]['partnerTime']))
        return ta

    @staticmethod
    def parse_or_group(json: dict) -> OrGroup:
        
        # Find nested OR groups/AND groups
        # TODO: is this correct if there are not nested groups in OCS natively

        observations = [JsonProvider.parse_observation(json[key], key) for key in json.keys() if key.startswith('OBSERVATION_BASIC')]

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
    def parse_root_group(json: dict) -> OrGroup:
        # Find nested OR groups/AND groups
        groups = [JsonProvider.parse_and_group(json[key]) for key in json.keys() if key.startswith('GROUP_GROUP_SCHEDULING')]
        num_to_observe = len(groups)
        root_group = AndGroup(None, None, num_to_observe, 0, 0, groups, AndOption.ANYORDER)
        return root_group

    @staticmethod
    def parse_and_group(json: dict) -> AndGroup:
        observations = [JsonProvider.parse_observation(json[key], key) for key in json.keys() if key.startswith('OBSERVATION_BASIC')]

        number_to_observe = len(observations)
        delay_max, delay_min = 0, 0 # TODO: What are these?
        # or_group = AndGroup(json['key'], json['name'], number_to_observe, delay_min, delay_max, observations, AndOption.ANYORDER)
        return AndGroup(json['key'], json['name'], number_to_observe, delay_min, delay_max, observations, AndOption.ANYORDER)

    @staticmethod
    def parse_program(json: dict) -> Program:

        ta = JsonProvider.parse_time_allocation(json)
        root_group = JsonProvider.parse_root_group(json)
        program = Program(json['programId'],
                          json['key'],
                          Band(int(json['queueBand'])),
                          bool(json['isThesis']),
                          ProgramMode[json['programMode'].upper()],
                          None,
                          None,
                          ta,
                          root_group,
                          TooType(json['tooType']) if json['tooType'] != 'None' else None)

        print(program.band)
        print(program.mode)
        return program

if __name__ == '__main__':
    provider = JsonProvider('../data/programs.json')

    json = provider.load_program('./data/GN-2018B-Q-101.json')

    program = provider.parse_program(json['PROGRAM_BASIC'])