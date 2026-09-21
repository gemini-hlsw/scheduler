# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from enum import Enum
from typing import Dict, Type, Union, final

import strawberry  # noqa

from .additive import AdditiveRanker
from .base import Ranker
from .default import DefaultRanker

__all__ = [
    'RankerName',
    'ranker_class',
]


@final
@strawberry.enum
class RankerName(Enum):
    """
    The Rankers a run can score with.

    Selectable per run in VALIDATION only, so the scoring algorithm can be compared side by
    side without a redeploy. Every other mode uses `config.ranker.name`.
    """
    DEFAULT = 'default'
    ADDITIVE = 'additive'


_RANKERS: Dict[RankerName, Type[Ranker]] = {
    RankerName.DEFAULT: DefaultRanker,
    RankerName.ADDITIVE: AdditiveRanker,
}


def ranker_class(name: Union[RankerName, str]) -> Type[Ranker]:
    """
    Resolve a Ranker by name. A new Ranker plugs in above with no new branch anywhere else.

    Accepts the enum (from the GraphQL input) or a case-insensitive string (from
    config.ranker.name). Raises KeyError with the offending name if it is unknown.
    """
    if isinstance(name, RankerName):
        return _RANKERS[name]
    try:
        return _RANKERS[RankerName[str(name).upper()]]
    except KeyError:
        raise KeyError(name) from None
