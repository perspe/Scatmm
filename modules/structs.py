from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any
import logging


@dataclass()
class SRes:
    """ Dataclass to store all simulation related information """
    ID: int # Simulation Number in the stack
    # Simulation parameters -- allowed comparison
    Type: Enum
    Layers: list
    Theta: Any
    Phi: float
    Pol: tuple
    INC_MED: tuple
    TRN_MED: tuple
    Lmb: Any
    # Simulation Results -- no comparison
    Ref: Any = field(repr=False, compare=False)
    Trn: Any = field(repr=False, compare=False)
    # Internal Variables created in post_init
    _REPR: str = field(init=False, compare=False)
    _Type: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """ Create internal variables to represent the particular class """
        if self.Type == SType.WVL:
            self._REPR = f"W{self.ID}({self.Theta},{self.Phi})"
            self._Type = ("Wavelength (nm)", self.Lmb)
        elif self.Type == SType.ANGLE:
            self._REPR = f"A{self.ID}({self.Lmb},{self.Phi})"
            self._Type = ("Angle (θ)", self.Theta)
        elif self.Type == SType.OPT:
            self._REPR = f"O{self.ID}({self.Theta},{self.Phi})"
            self._Type = ("Optimization (nm)", self.Lmb)
        else:
            raise Exception("Unknown Simulation Type")
        for layer in self.Layers:
            self._REPR += "|" + layer.name[:5] + "(" + str(
                layer.thickness) + ")"
        logging.debug(f"{self._REPR=}")

    def description(self):
        """ Return a detailed description of the Layer """
        layer_info = "----------------------\nLayers:\n"
        type = ""
        phi = ""
        i_med = ""
        t_med = ""
        pol = ""
        if self.Type == SType.WVL:
            type = "Broadband Simulation:\n\n"
            type += f"λ : {self.Lmb.min()} - {self.Lmb.max()}\n"
            type += f"θ : {self.Theta}\n"
        elif self.Type == SType.ANGLE:
            type = "Angular simulation:\n\n"
            type += f"θ : {self.Theta.min()} - {self.Theta.max()}\n"
            type += f"λ : {self.Lmb}\n"
        elif self.Type == SType.OPT:
            type = "Optimization:\n\n"
            type += f"λ : {self.Lmb.min()} - {self.Lmb.max()}\n"
            type += f"θ : {self.Theta}\n"
        else:
            raise Exception("Invalid SType")

        phi = f"Φ : {self.Phi}\n"
        pol = f"Polarization: {self.Pol} (PTM, PTE)\n"
        i_med = f"Incidence Medium: {self.INC_MED}\n"
        t_med = f"Transmission Medium: {self.TRN_MED}\n"
        for layer in self.Layers:
            layer_info += f"{layer.name} : ({layer.thickness} nm)\n"
        summary = type + phi + pol + i_med + t_med + layer_info
        return summary

    def update_ID(self, new_id: int):
        self.ID = new_id
        self.__post_init__()

    """ Aliases for the post init variables """
    @property
    def repr(self):
        return self._REPR

    def xinfo(self):
        return self._Type


class SType(Enum):
    """ Enum to for the different Simulation types """
    WVL = auto()
    ANGLE = auto()
    OPT = auto()
