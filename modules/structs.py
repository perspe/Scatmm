from enum import Enum, auto
from dataclasses import dataclass
from typing import Any


@dataclass
class SRes:
    """ Dataclass to store all simulation related information """
    ID: str
    Type: Enum
    Layers: list
    NLayers: int
    Theta: Any
    Phi: float
    Pol: tuple
    Lmb: Any
    INC_MED: tuple
    TRN_MED: tuple
    Ref: Any
    Trn: Any


class SType(Enum):
    """ Enum to for the different Simulation types """
    WVL = auto()
    ANGLE = auto()
    OPT = auto()
