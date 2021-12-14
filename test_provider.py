import json
from datetime import datetime, timedelta
from common.api import ProgramProvider
from common.minimodel import *


class JsonProvider(ProgramProvider):
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
        band = MagnitudeBand(json['name'])
        value = json['value']
        error = ...
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
        constraints = Constraints()
        
        constraints.cc = json['cc']
        constraints.iq = json['iq']
        constraints.sb = json['sb']
        constraints.wv = json['wv']
        constraints.elevation_type = json['elevationConstraintType']
        constraints.elevation_min = json['elevationConstraintMin']
        constraints.elevation_max = json['elevationConstraintMax']
        constraints.timing_windows = JsonProvider.parse_timing_window(json['timingWindows']) # this would need parsing 
        constraints.strehl = ...
    
    @staticmethod
    def parse_atom(json: dict) -> Observation:
        atom = Atom()

        atom.id = ...
        atom.exec_time = ...
        atom.prog_time = ...
        atom.part_time = ...
        atom.observed = json['metadata:complete']
        atom.qa_state = ...
        atom.guide_state = ...
        atom.required_resources = ...
        atom.wavelength = ...
        return atom

    @staticmethod
    def parse_sidereal_target(json: dict) -> SiderealTarget:
        sidereal = SiderealTarget()
        sidereal.ra = json['ra'] # this might need transformation
        sidereal.dec = json['dec'] # this might need transformation
        sidereal.pm_ra = json['base']['deltara']
        sidereal.pm_dec = json['base']['deltadec']
        sidereal.epoch = json['base']['epoch']

        magnitudes = [JsonProvider.parse_magnitude(mag) for mag in json['magnitudes']]
        return sidereal

    @staticmethod
    def parse_nonsidereal_target(json: dict) -> NonsiderealTarget:
        nonsidereal = NonsiderealTarget()
        nonsidereal.des = json['base']['des']
        nonsidereal.tag = json['base']['tag']
        nonsidereal.ra = ...
        nonsidereal.dec = ...

        return nonsidereal

    @staticmethod
    def parse_observation(json: dict, name: str ) -> Observation:
        obs = Observation()

        obs.id = json['observationId']
        obs.internal_id = json['key']
        obs.order = int(name.split('-')[1])
        obs.title = json['title']
        obs.site = ...
        obs.status = json['obsStatus']
        obs.active = ...
        obs.priority = json['priority']
        obs.setuptime_type = SetupTimeType(json['setupTimeType'])
        obs.acq_overhead = ...
        obs.obs_class = ObservationClass(json['obsClass'])
        obs.exec_time = ...
        obs.program_used = ...
        obs.partner_used = ...
        targets_in_json = [json[key] for key in json.keys() if key.startswith('TELESCOPE_TARGETENV')]
        targets = [JsonProvider.parse_sidereal_target(target) if target['base']['tag'] == 'sidereal' else JsonProvider.parse_sidereal_target(target) for target in targets_in_json]
        obs.targets = targets
        obs.guide_stars = ...
        obs.sequence = [JsonProvider.parse_atom(seq) for seq in json['sequence']]
        find_constraints = [json[key] for key in json.keys() if key.startswith('SCHEDULING_CONDITIONSt')]
        obs.constraints = JsonProvider.parse_constraints(find_constraints[0])
        obs.too_type = ...

        return obs
    
    @staticmethod
    def parse_time_allocation(json: dict) -> TimeAllocation:
        print(json['timeAccountAllocationCategories'])
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
        observations = [JsonProvider.parse_observation(json[key], key) for key in json.keys() if key.startwith('OBSERVATION_BASIC')]

        number_to_observe = len(observations)
        delay_max, delay_min = 0, 0 # TODO: What are these?
        or_group = OrGroup(json['key'], json['name'], number_to_observe, delay_min, delay_max, observations)
        return or_group

    @staticmethod
    def parse_guide_star(json: dict) -> GuideStar:
        ...
    
    @staticmethod
    def parse_root_group(json: dict) -> OrGroup:
        # Find nested OR groups/AND groups
        groups = [JsonProvider.parse_or_group(key) for key in json.keys() if key.startswith('GROUP_GROUP_SCHEDULING')]
        num_to_observe = len(groups)
        root_group = OrGroup(None, None, num_to_observe, 0, 0, groups)
        return root_group

    @staticmethod
    def parse_and_group(json: dict) -> AndGroup:
        ...

    @staticmethod
    def parse_program(json: dict) -> Program:

        ta = JsonProvider.parse_time_allocation(json)
        root_group = JsonProvider.parse_root_group(json)
        program = Program(json['programId'],
                          json['key'],
                          Band(int(json['queueBand'])),
                          bool(json['isThesis']),
                          ProgramMode(json['programMode']),
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