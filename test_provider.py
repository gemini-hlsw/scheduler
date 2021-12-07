import json
from datetime import datetime, timedelta
from common.api import ProgramProvider
from common.minimodel import Program, TimeAllocation, Group, TooType, ProgramMode, TimeAccountingCode, Band

class JsonProvider(ProgramProvider):
    def __init__(self, path):
        self.path = path

    def load_program(self, path: str) -> dict:
        with open(self.path, 'r') as f:
            return json.loads(f)
    

    def parse_time_allocation(json: dict) -> TimeAllocation:
        ta = TimeAllocation()
        ta.category = TimeAccountingCode(json['timeAccountAllocationCategories']['category'])

        ta.program_awarded = timedelta(milliseconds=json['awardedTime'])
        ta.partner_awarded = timedelta(milliseconds=0)
       
        ta.program_used = timedelta(milliseconds=json['timeAccountAllocationCategories']['programTime'])
        ta.partner_used = timedelta(milliseconds=json['timeAccountAllocationCategories']['partnerTime'])
        print(ta.program_used)
        return ta

    def parse_program(json: dict) -> Program:
        program = Program()

        program.id = json['programId']
        program.internal_id = json['key']
        program.band = Band(json['band'])
        program.thesis = bool(json['isThesis'])
        program.mode = ProgramMode(json['programMode'])
        program.start_time = ...
        program.end_time = ...
    
        too_type = TooType(json['tooType']) if json['tooType'] != 'None' else None

        print(program.band)
        print(program.mode)
        return program

if __name__ == '__main__':
    provider = JsonProvider('../data/programs.json')

    json = provider.load_program('./data/GN-2018B-Q-101.json')

    program = provider.parse_program(json)
    time_allocation = provider.parse_time_allocation(json)
    program.time_allocation = time_allocation