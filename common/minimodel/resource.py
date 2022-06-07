from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Resource:
    """
    This is a general observatory resource.
    It can consist of a guider, an instrument, or a part of an instrument,
    or even a personnel and is used to determine what observations can be
    performed at a given time based on the resource availability.
    """
    id: str
    description: Optional[str] = None

    def __post_init__(self):
        if self.id is None or 'NONE' in self.id.upper():
            raise ValueError('Should not have any Resources equal to None or containing "None"')

    def __eq__(self, other):
        return isinstance(other, Resource) and self.id == other.id

    def __repr__(self):
        return f'Resource(id=\'{self.id}\')'
