from enum import Enum, auto
from dataclasses import dataclass
from typing import Any


@dataclass
class SRes:
    """ Dataclass to store all simulation related information """
    ID: str
    Type: Enum
    Theta: Any
    Phi: float
    Pol: tuple
    Lmb: Any
    Ref: Any
    Trn: Any
    NLayers: int


class SType(Enum):
    """ Enum to for the different Simulation types """
    WVL = auto()
    ANGLE = auto()
    OPT = auto()
