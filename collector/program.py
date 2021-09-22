from typing import NoReturn
from common.structures.band import Band


class Program:
    def __init__(self,
                 idx: str,
                 mode: str, 
                 band: Band, 
                 thesis: str, 
                 time: str,
                 used_time, 
                 too_status,
                 start,
                 end) -> None:
        self.idx = idx
        self.mode = mode
        self.band = band
        self.thesis = thesis
        self.time = time
        self.used_time = used_time
        self.too_status = too_status
        self.start = start
        self.end = end
        self.observations = []
        self.groups = []

    def add_observation(self, obs_idx: int) -> NoReturn:
        if obs_idx not in self.observations:
            self.observations.append(obs_idx)

    def add_group(self, grp_idx: str) -> NoReturn:
        if grp_idx not in self.groups:
            self.groups.append(grp_idx)
