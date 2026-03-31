# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import field, dataclass
from datetime import timedelta
from typing import FrozenSet, Dict, List, Tuple, Optional, ClassVar

import numpy as np
import numpy.typing as npt
from lucupy.minimodel import Site, GroupID, ObservationID, NightIndex, Atom, ObservationStatus, Observation
from lucupy.types import ZeroTime



__all__ = ["TimeAccountant"]

@dataclass
class AtomAccountingRecord:
    """
    Records time accounting at atom level.

    _dirty flag helps to check if any parameters where modified in the time accounting process.
    """
    exec_time: timedelta = field(hash=False, compare=False)
    prog_time: timedelta = field(hash=False, compare=False)
    part_time: timedelta = field(hash=False, compare=False)
    program_used: timedelta = field(hash=False, compare=False)
    partner_used: timedelta = field(hash=False, compare=False)
    not_charged: timedelta = field(hash=False, compare=False)
    observed: bool

    _dirty: bool = field(default=False, repr=False, init=False)
    _tracked_fields: ClassVar[frozenset] = frozenset({
        'exec_time', 'prog_time', 'part_time',
        'program_used', 'partner_used', 'not_charged', 'observed'
    })

    def __setattr__(self, name, value):
        if name in self._tracked_fields and hasattr(self, name):
            if getattr(self, name) != value:
                object.__setattr__(self, '_dirty', True)
        object.__setattr__(self, name, value)

    def clear_dirty(self):
        object.__setattr__(self, '_dirty', False)

@dataclass
class ObservationAccountingRecord:
    """
    Records Observation and atom records for an observation.
    """
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
    """
    Keeps track of how observation time is accounted for across sites and nights.

    TODO: Add time accounting method from collector in here.

    """
    def __init__(self, sites: FrozenSet[Site], night_indices: List[NightIndex]):
        """
        The structure separates the records for specific sites and night_indices.
        each ObservationAccountingRecord records a list of atoms records that hold the TA calculations.
        """

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
        """
        Ensures the context is set when accessing records.
        """
        if self._current_data is None:
            raise RuntimeError("Call set_current(site, night_index) before accessing records.")
        return self._current_data

    def set_current(self, site: Site, night_index: NightIndex) -> None:
        """
        Set night and site context for easy retrieval.
        """
        self._current_site = site
        self._current_night = night_index
        self._current_data = self._time_accounting_table[site][night_index]

    def get_record(
            self, obs: Observation, atom: Atom
    ) -> AtomAccountingRecord:
        """
        Get the AtomAccountingRecord in the current context.
        If the record does not exist, create it and return it.
        """
        data = self._ensure_current()

        # Create observation record if it doesn't exist
        if obs.id not in data:
            data[obs.id] = ObservationAccountingRecord(
                status=obs.status,
                atoms_records=[
                    (a, AtomAccountingRecord(
                        exec_time=a.exec_time,
                        prog_time=a.prog_time,
                        part_time=a.part_time,
                        program_used=a.program_used,
                        partner_used=a.partner_used,
                        not_charged=a.not_charged,
                        observed=a.observed,
                    )) for a in obs.sequence
                ]
            )
            # Mark all atoms in new observation as dirty
            for _, record in data[obs.id].atoms_records:
                object.__setattr__(record, '_dirty', True)

        # Find and return the atom record
        for existing_atom, record in data[obs.id].atoms_records:
            if existing_atom == atom:
                return record

        raise KeyError(f"Atom {atom} not found in obs={obs.id}")

    def get_dirty_observations(self) -> Dict[ObservationID, ObservationAccountingRecord]:
        """
        Retrieves all observations records with modified AtomAccountingRecords.
        It means those observations were accounted by the last plan entry.
        """
        data = self._ensure_current()
        return {
            obs_id: obs_record
            for obs_id, obs_record in data.items()
            if any(record._dirty for _, record in obs_record.atoms_records)
        }

    def clear_all_dirty(self) -> None:
        """
        Clear the dirty flag for all records.
        """
        data = self._ensure_current()
        for obs_record in data.values():
            for _, record in obs_record.atoms_records:
                record.clear_dirty()
