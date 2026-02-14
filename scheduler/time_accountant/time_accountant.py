from collections import defaultdict
from dataclasses import field, dataclass
from datetime import timedelta
from typing import FrozenSet, Sequence, Dict, List, Tuple, Optional, Final

import numpy as np
from lucupy.minimodel import Site, GroupID, ObservationID, NightIndex, Atom, ObservationStatus
from lucupy.types import ZeroTime
from pandas._typing import npt


__all__ = ["TimeAccountant"]

@dataclass
class AtomAccountingRecord:
    exec_time: timedelta = field(hash=False, compare=False)
    prog_time: timedelta = field(hash=False, compare=False)
    part_time: timedelta = field(hash=False, compare=False)
    program_used: timedelta = field(hash=False, compare=False)
    partner_used: timedelta = field(hash=False, compare=False)
    not_charged: timedelta = field(hash=False, compare=False)
    observed: bool

@dataclass
class ObservationAccountingRecord:
    status: ObservationStatus
    atoms_records: List[Tuple[Atom,AtomAccountingRecord]]

    @property
    def exec_time(self) -> timedelta:
        return sum(
            (atom.exec_time for atom, atom_record in self.atoms_records),
            start=timedelta()
        )

    @property
    def cum_seq(self) -> npt.NDArray[timedelta]:
        """
        Cumulative series of execution times for the unobserved atoms
        in a sequence, excluding acquisition time.
        """
        cum_seq = [atom_record.exec_time if not atom.observed else ZeroTime for atom, atom_record in self.atoms_records]
        return np.cumsum(cum_seq)


class TimeAccountant:

    def __init__(self, sites: FrozenSet[Site], night_indices: List[NightIndex]):

        self._time_accounting_table: Dict[
            Site, Dict[NightIndex, Dict[ObservationID, ObservationAccountingRecord]]
        ] = {
            site: {
                night_idx: {}
                for night_idx in night_indices
            }
            for site in sites
        }

        self._current_site: Optional[Site] = None
        self._current_night: Optional[NightIndex] = None
        self._current_data: Optional[Dict[ObservationID, ObservationAccountingRecord]] = None

    def _ensure_current(self) -> Dict[ObservationID, ObservationAccountingRecord]:
        if self._current_data is None:
            raise RuntimeError("Call set_current(site, night_index) before accessing records.")
        return self._current_data

    def set_current(self, site: Site, night_index: NightIndex) -> None:
        self._current_site = site
        self._current_night = night_index
        self._current_data = self._time_accounting_table[site][night_index]

    def add_record(
        self, obs_id: ObservationID, atom: Atom, record: AtomAccountingRecord
    ) -> None:
        data = self._ensure_current()
        atoms = data[obs_id].atoms_records
        # Check if this atom_id already exists
        for existing_atom, _ in atoms:
            if existing_atom.id == atom.id:
                raise ValueError(
                    f"Record for atom {atom.id} already exists fop obs={obs_id}"
                )
        atoms.append((atom, record))

    def get_record(
        self, obs_id: ObservationID, atom_idx: int
    ) -> AtomAccountingRecord:
        data = self._ensure_current()
        for existing_atom_id, record in data[obs_id].atoms_records:
            if existing_atom_id == atom_idx:
                return record
        raise KeyError(
            f"No record for atom {atom_idx} in obs={obs_id}"
        )
