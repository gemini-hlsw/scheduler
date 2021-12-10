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
    def parse_magnitude(json: dict) -> Magnitude:
        ...

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
    def parse_observation(json: dict) -> Observation:
        obs = Observation()

        obs.id = json['observationId']
        obs.internal_id = json['key']
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
        obs.constraints = JsonProvider    .parse_constraints(find_constraints[0])
        obs.too_type = ...
    
    @staticmethod
    def parse_time_allocation(json: dict) -> TimeAllocation:
        ta = TimeAllocation()
        ta.category = TimeAccountingCode(json['timeAccountAllocationCategories']['category'])

        ta.program_awarded = timedelta(milliseconds=json['awardedTime'])
        ta.partner_awarded = timedelta(milliseconds=0)
       
        ta.program_used = timedelta(milliseconds=json['timeAccountAllocationCategories']['programTime'])
        ta.partner_used = timedelta(milliseconds=json['timeAccountAllocationCategories']['partnerTime'])
        print(ta.program_used)
        return ta


    @staticmethod
    def parse_or_group(json: dict) -> OrGroup:
        ...

    @staticmethod
    def parse_and_group(json: dict) -> AndGroup:
        ...

    @staticmethod
    def parse_program(json: dict) -> Program:
        print(json.keys())
        program = Program(json['programId'],
                          json['key'],
                          Band(json['queueBand']),
                          bool(json['isThesis']),
                          ProgramMode(json['programMode'],
                          None,
                          None,
                          TooType(json['tooType']) if json['tooType'] != 'None' else None))

        print(program.band)
        print(program.mode)
        return program

if __name__ == '__main__':
    provider = JsonProvider('../data/programs.json')

    json = provider.load_program('./data/GN-2018B-Q-101.json')

    program = provider.parse_program(json['PROGRAM_BASIC'])
    time_allocation = provider.parse_time_allocation(json['PROGRAM_BASIC'])
    program.time_allocation = time_allocation