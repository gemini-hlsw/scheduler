# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
from contextlib import contextmanager
from fcntl import flock, LOCK_SH, LOCK_EX, LOCK_UN
from typing import List, Dict
from app.graphql.scalars import SPlans
import shelve

@contextmanager
def locking(lock_path, lock_mode):
    with open(lock_path, 'w') as lock:
        flock(lock.fileno(), lock_mode) # block until lock is acquired
        try:
            yield
        finally:
            flock(lock.fileno(), LOCK_UN) # release

class DBManager:
    def __init__(self, db_path):
        self.db_path = db_path

    def read(self):
        with locking(f'{self.db_path}.lock', LOCK_SH):
            with shelve.open(self.db_path) as db:
                return db['plans']

    def write(self, plans: List[SPlans]):
         with locking(f'{self.db_path}.lock', LOCK_EX):
             with shelve.open(self.db_path) as db:
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