# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import shelve
from contextlib import contextmanager
from fcntl import flock, LOCK_SH, LOCK_EX, LOCK_UN
from typing import List, Optional, FrozenSet
from lucupy.minimodel import Site

from scheduler.graphql_mid.types import SPlans


@contextmanager
def locking(lock_path, lock_mode):
    with open(lock_path, 'w') as lock:
        flock(lock.fileno(), lock_mode)  # block until lock is acquired
        try:
            yield
        finally:
            flock(lock.fileno(), LOCK_UN)  # release


class DBManager:
    def __init__(self, db_path):
        self.db_path = db_path

    def read(self, start_date: Optional[str] = None,
             end_date: Optional[str] = None,
             site: Optional[FrozenSet[Site]] = None) -> List[SPlans]:
        with locking(f'{self.db_path}.lock', LOCK_SH):
            with shelve.open(self.db_path) as db:
                if start_date and end_date and site:
                    plans_by_site = db[start_date + end_date]
                    if Site.GS in site and Site.GN in site:
                        return plans_by_site['both']
                    elif Site.GN in site:
                        return plans_by_site['GN']
                    elif Site.GS in site:
                        return plans_by_site['GS']
                else:
                    return db['plans']

    def write(self,
              plans: List[SPlans],
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              site: Optional[FrozenSet[Site]] = None) -> None:
        with locking(f'{self.db_path}.lock', LOCK_EX):
            with shelve.open(self.db_path) as db:
                if start_date and end_date and site:
                    if Site.GS in site and Site.GN in site:
                        db[start_date + end_date] = {'both': plans}
                    elif Site.GN in site:
                        db[start_date + end_date] = {'GN': plans}
                    elif Site.GS in site:
                        db[start_date + end_date] = {'GS': plans}
                else:
                    db['plans'] = plans

    '''


    def cas(self, old_db: Dict, new_db):
        with locking(f'{self.db_path}.lock', LOCK_EX):
            with shelve.open(self.db_path) as db:
                if old_db != dict(db):
                    return False
                db.clear()
                db.update(new_db)
                return True
    '''
