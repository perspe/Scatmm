"""
File with the optimized version of the Scattering Matrix Script

Class:
    Layer1D: Define a SMM Layer for materials with constant properties
    Layer3D: Define a broadband SMM Layer material
Functions:
    smm: Calculates R and T for a specific wavelength
    smm_broadband: Calculates R and T for a wavelength range
    smm_angle: Calculates R and T for a angle range
    smm_layer: Calculate the absorption for a specific layer
"""
import logging
from typing import Any, List, Tuple, Union

import numpy as np
from numpy.linalg import inv
import numpy.typing as npt
from scipy.interpolate import interp1d

from .py_smm_base import CMatrix, CSMatrix
from .py_smm_base import SMMType as CSMMType

# Default simulation config
sim_config = {
    "theta": 0.0,
    "phi": 0.0,
    "lmb": 500.0,
    "pol": (1, 0),
    "i_med": (1.0, 1.0),
    "t_med": (1.0, 1.0)
}


class MatOutsideBounds(Exception):
    """ Exception for material outside bounds """
    def __init__(self, material, wavelength):
        self.message = f"Simulation wavelength ({wavelength}) is outside " +\
            f"the defined oustide bounds for '{material}'"
        super().__init__(self.message)


class InvalidParameter(Exception):
    pass


class Layer1D():
    """
    Layer for single wavelength materials
    Args:
        name (str): Identifier for the layer
        thickness (float - nm): Layer thickness
        lmb (nm)/n_val/k_val (array): Material info for the layer
    """
    def __init__(self, name: str, thickness: float, n_val: float,
                 k_val: float) -> None:
        self.name: str = name
        self.thickness: float = thickness
        self.n: float = n_val
        self.k: float = k_val
        logging.debug(f"Layer: {self} created...")

    def e_value(self, lmb: npt.ArrayLike) -> npt.ArrayLike:
        """ Return e_value for specific wavelength """
        res: npt.ArrayLike = np.zeros_like(lmb, dtype=np.complex128)
        res += (self.n + 1j * self.k)**2
        return res

    def __repr__(self) -> str:
        return f"{self.name}({self.thickness} nm)"

    def __eq__(self, __o) -> bool:
        """ Create a layer comparison through the properties """
        if not isinstance(__o, Layer1D):
            raise TypeError
        if self.n == __o.n and self.k == __o.k and self.thickness == __o.thickness:
            return True
        else:
            return False


class Layer3D():
    """
    Layer for single wavelength materials
    Args:
        name (str): Identifier for the layer
        thickness (float - nm): Layer thickness
        lmb (nm)/n_array/k_array (array): Arrays with the
                                          basic info for each layer
        **kwargs: Pass extra arguments for interpolation function
    """
    def __init__(self, name: str, thickness: float, lmb: npt.NDArray,
                 n_array: npt.NDArray, k_array: npt.NDArray, **kwargs) -> None:
        self.name: str = name
        self.lmb: list = [np.min(lmb), np.max(lmb)]
        self.thickness: float = thickness
        # Store initial data for comparisons
        self._og_n = n_array
        self._og_k = k_array
        self.n = interp1d(lmb, n_array, **kwargs)
        self.k = interp1d(lmb, k_array, **kwargs)
        logging.debug(f"Layer: {self} created...")

    def e_value(self, lmb: npt.ArrayLike) -> npt.ArrayLike:
        """ Calculate e_values for a range of wavelengths """
        try:
            e_data: npt.ArrayLike = (self.n(lmb) + 1j * self.k(lmb))**2
        except ValueError:
            logging.error("Invalid Wavelength Value/Range")
            if isinstance(lmb, float):
                raise MatOutsideBounds(self.name, f"{lmb}")
            elif isinstance(lmb, np.ndarray):
                raise MatOutsideBounds(self.name, f"{lmb.min()}, {lmb.max()}")
            else:
                raise Exception("Invalid wavelength value")
        return e_data

    def __repr__(self) -> str:
        return f"{self.name}({self.thickness} nm)"

    def __eq__(self, __o) -> bool:
        """ Create a layer comparison through the properties """
        if not isinstance(__o, Layer3D):
            raise TypeError
        n_bool = self._og_n == __o._og_n
        k_bool = self._og_k == __o._og_k
        if n_bool.all() and k_bool.all() and self.thickness == __o.thickness:
            return True
        else:
            return False


""" Helper functions """


def _initialize_smm(theta, phi, lmb, pol, inc_medium):
    """ Initialize the parameters necessary for the smm calculations """
    if np.size(lmb) > 1 and np.size(theta) > 1:
        raise Exception("Only wavelength or theta can be an array")
    # Wavevector for the incident wave
    k0 = 2 * np.pi / lmb
    kx = np.sqrt(inc_medium[0] * inc_medium[1]) * np.sin(theta) * np.cos(phi)
    ky = np.sqrt(inc_medium[0] * inc_medium[1]) * np.sin(theta) * np.sin(phi)

    # Free Space parameters
    Q0 = np.array([[kx * ky, 1 + ky**2], [-(1 + kx**2), -kx * ky]])
    V0 = 0 - Q0 * 1j
    V0 = CMatrix(V0[0, 0], V0[0, 1], V0[1, 0], V0[1, 1])

    # Reduced polarization vector
    if theta == 0:
        ate = np.array([0, 1, 0])
        atm = np.array([1, 0, 0])
    else:
        ate = np.array([-np.sin(phi), np.cos(phi), 0])
        atm = np.array([
            np.cos(phi) * np.cos(theta),
            np.cos(theta) * np.sin(phi), -np.sin(theta)
        ])
    # Create the composite polariztion vector
    p_vector = np.add(pol[1] * ate, pol[0] * atm)
    p = p_vector[[0, 1]]
    logging.debug(
        f"Initialization Values for SMM: {k0=}\n{kx=}\n{ky=}\n{V0=}\n{p=}")
    return k0, kx, ky, V0, np.array(p, dtype=np.complex128)


""" Main functions """
Layer_Type = Union[Layer3D, Layer1D]


def smm(layer_list: List[Layer_Type], theta: float, phi: float, lmb: float,
        pol: Tuple[complex, complex], i_med: Tuple[complex, complex],
        t_med: Tuple[complex, complex]) -> Tuple[float, float]:
    """
    SMM for a single point simulation
    Args:
        layer_list: List of Layer objects with the info for all layers
        theta/phi: Incidence angles
        lmb: Wavelength for a particular simulation
        pol: Tuple with the TM/TE polarization components
        i_med/t_med: Data for the reflection and transmission media
    Returns:
        R, T: Arrays with the Reflection and transmission for the layer setup
    """
    logging.debug(f"SMM for {theta}:{phi}:{lmb}:{pol}:{i_med}:{t_med}")
    if not isinstance(lmb, (float, int)) or not isinstance(
            theta, (float, int)) or not isinstance(phi, (int, float)):
        logging.error("Invalid input parameter...")
        raise InvalidParameter("Invalid Input parameter")
    # Inicialize necessary values
    k0, kx, ky, V0, p = _initialize_smm(theta, phi, lmb, pol, i_med)
    S_trn = CSMatrix(V0, k0, kx, ky, 0, t_med[0], t_med[1], CSMMType.TRN)
    S_ref = CSMatrix(V0, k0, kx, ky, 0, i_med[0], i_med[1], CSMMType.REF)
    S_Global = S_ref
    for layer in layer_list:
        e_value: Any = layer.e_value(lmb)
        S_Layer = CSMatrix(V0, k0, kx, ky, layer.thickness, e_value)
        S_Global = S_Global * S_Layer
    S_Global = S_Global * S_trn
    kz_ref = np.sqrt(i_med[0] * i_med[1] - kx**2 - ky**2)
    kz_trn = np.sqrt(t_med[0] * t_med[1] - kx**2 - ky**2)
    E_ref, E_trn = S_Global.ref_trn(p, kz_ref, kz_trn)
    R = E_ref
    T = (E_trn) * np.real((kz_trn * i_med[1]) / (kz_ref * t_med[1]))
    return R, T


def smm_broadband(layer_list: List[Layer_Type],
                  theta: float,
                  phi: float,
                  lmb: npt.NDArray,
                  pol: Tuple[complex, complex],
                  i_med: Tuple[complex, complex],
                  t_med: Tuple[complex, complex],
                  override_thick=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMM for broadband simulation
    Args:
        layer_list: List of Layer objects with the info for all layers
        theta/phi: Incidence angles
        lmb: Array with wavelengths for simulation
        pol: Tuple with the TM/TE polarization components
        i_med/t_med: Data for the reflection and transmission media
        override_thick: Override the built thickness of the layers
    Returns:
        R, T: Arrays with the Reflection and transmission for the layer setup
    """
    logging.debug(
        f"SMM Broadband for {theta}:{phi}:{pol}:{i_med}:{t_med}:{override_thick}"
    )
    if not isinstance(theta,
                      (int, float)) or not isinstance(phi, (int, float)):
        raise InvalidParameter("Invalid Input parameter")
    if override_thick is not None:
        if len(override_thick) == len(layer_list):
            for thick_i, layer in zip(override_thick, layer_list):
                layer.thickness = thick_i
        else:
            logging.warning("Invalid Thickness provided")
            raise InvalidParameter(
                "Override Thickness does not match Layer List")
    k0, kx, ky, V0, p = _initialize_smm(theta, phi, lmb, pol, i_med)
    kz_ref = np.sqrt(i_med[0] * i_med[1] - kx**2 - ky**2)
    kz_trn = np.sqrt(t_med[0] * t_med[1] - kx**2 - ky**2)
    # This is a simplification to determine all the values in beforehand
    layer_data = np.array([layer_i.e_value(lmb) for layer_i in layer_list])
    R = []
    T = []
    # Loop through all wavelengths and layers
    for k0_i, layer_data in zip(k0, layer_data.T):
        S_trn = CSMatrix(V0, k0_i, kx, ky, 0, t_med[0], t_med[1], CSMMType.TRN)
        S_ref = CSMatrix(V0, k0_i, kx, ky, 0, i_med[0], i_med[1], CSMMType.REF)
        S_Global_i = S_ref
        for index, layer in enumerate(layer_list):
            S_Layer = CSMatrix(V0, k0_i, kx, ky, layer.thickness,
                               layer_data[index])
            S_Global_i = S_Global_i * S_Layer
        S_Global_i = S_Global_i * S_trn
        E_ref, E_trn = S_Global_i.ref_trn(p, kz_ref, kz_trn)
        R.append(E_ref)
        T.append((E_trn) * np.real((kz_trn * i_med[1]) / (kz_ref * t_med[1])))
    return np.array(R), np.array(T)


def smm_angle(layer_list: List[Layer_Type],
              theta: np.ndarray,
              phi: float,
              lmb: float,
              pol: Tuple[complex, complex],
              i_med: Tuple[complex, complex],
              t_med: Tuple[complex, complex],
              override_thick=None):
    """
    SMM for broad angle simulations
    Args:
        layer_list: List of layers for the device
        theta: array with the incidence angles
        phi: Polar incidence angle
        lmb: Single wavelength for the simulation
        pol: Incident polarization
        i_med/t_med (tuple): Incidence and transmission region properties
    Returns:
        R, T: Arrays with the Reflection and transmission for the layer setup
    """
    logging.debug(
        f"SMM Angle for {phi}:{lmb}:{pol}:{i_med}:{t_med}:{override_thick}")
    if not isinstance(lmb, (float, int)) or not isinstance(phi, (int, float)):
        raise InvalidParameter("Invalid Input parameter")
    if override_thick is not None:
        if len(override_thick) == len(layer_list):
            for thick_i, layer in zip(override_thick, layer_list):
                layer.thickness = thick_i
        else:
            raise InvalidParameter(
                "Override Thickness does not match Layer List")
    R, T = [], []
    for theta_i in theta:
        k0, kx, ky, V0, p = _initialize_smm(theta_i, phi, lmb, pol, i_med)
        S_trn = CSMatrix(V0, k0, kx, ky, 0, t_med[0], t_med[1], CSMMType.TRN)
        S_ref = CSMatrix(V0, k0, kx, ky, 0, i_med[0], i_med[1], CSMMType.REF)
        S_Global = S_ref
        for layer in layer_list:
            e_value: Any = layer.e_value(lmb)
            S_Layer = CSMatrix(V0, k0, kx, ky, layer.thickness, e_value)
            S_Global = S_Global * S_Layer
        S_Global = S_Global * S_trn
        kz_ref = np.sqrt(i_med[0] * i_med[1] - kx**2 - ky**2)
        kz_trn = np.sqrt(t_med[0] * t_med[1] - kx**2 - ky**2)
        E_ref, E_trn = S_Global.ref_trn(p, kz_ref, kz_trn)
        R.append(E_ref)
        T.append((E_trn) * np.real((kz_trn * i_med[1]) / (kz_ref * t_med[1])))
    return np.array(R), np.array(T)


def smm_layer(layer_list: List[Layer_Type],
              layer_i: int,
              theta: float,
              phi: float,
              lmb: npt.NDArray,
              pol: Tuple[complex, complex],
              i_med: Tuple[complex, complex],
              t_med: Tuple[complex, complex],
              override_thick=None):
    """
    Determine absorption for a particular layer
    Args:
        layer_list: List of Layer objects with the info for all layers
        layer_i: Index for the layer to calculate the absorption
        theta/phi: Incidence angles
        lmb: Array with wavelengths for simulation
        pol: Tuple with the TM/TE polarization components
        i_med/t_med: Data for the reflection and transmission media
    Returns:
        Abs: Absorption fonumpy typing unionr a particular layer
    """
    logging.debug(
        f"SMM Layer for {theta}:{phi}:{pol}:{i_med}:{t_med}:{override_thick}")
    if not isinstance(theta,
                      (int, float)) or not isinstance(phi, (int, float)):
        raise InvalidParameter("Invalid Input parameter")
    if layer_i == 0 or layer_i > len(layer_list):
        raise InvalidParameter("Invalid Layer Index")
    layer_i -= 1
    if override_thick is not None:
        if len(override_thick) == len(layer_list):
            for thick_i, layer in zip(override_thick, layer_list):
                layer.thickness = thick_i
        else:
            raise InvalidParameter(
                "Override Thickness does not match Layer List")
    k0, kx, ky, V0, p = _initialize_smm(theta, phi, lmb, pol, i_med)
    kz_ref = np.sqrt(i_med[0] * i_med[1] - kx**2 - ky**2)
    kz_trn = np.sqrt(t_med[0] * t_med[1] - kx**2 - ky**2)
    # This is a simplification to determine all the values in beforehand
    layer_data = np.array([layer_i.e_value(lmb) for layer_i in layer_list])
    non_invertable_matrix = False
    Abs = []
    # Loop through all wavelengths and layers
    for k0_i, layer_data in zip(k0, layer_data.T):
        S_trn = CSMatrix(V0, k0_i, kx, ky, 0, t_med[0], t_med[1], CSMMType.TRN)
        S_ref = CSMatrix(V0, k0_i, kx, ky, 0, i_med[0], i_med[1], CSMMType.REF)
        S_ref_11, S_ref_12, S_ref_21, S_ref_22 = S_ref.return_SMatrix()
        S_Global_i = S_ref
        S_Global_After = S_ref
        S_Global_Before = S_ref
        for layer_index, layer in enumerate(layer_list):
            if layer_index == layer_i:
                S_Global_Before = S_Global_i.return_SMatrix()
            S_Layer = CSMatrix(V0, k0_i, kx, ky, layer.thickness,
                               layer_data[layer_index])
            S_Global_i = S_Global_i * S_Layer
            if layer_index == layer_i:
                S_Global_After = S_Global_i.return_SMatrix()
        S_Pre_Trn = S_Global_i.return_SMatrix()
        S_Pre_Trn_11, S_Pre_Trn_12, S_Pre_Trn_21, S_Pre_Trn_22 = S_Pre_Trn
        S_Global_i = S_Global_i * S_trn
        # Determine the total absorption of the device
        E_ref, E_trn = S_Global_i.ref_trn(p, kz_ref, kz_trn)
        R = E_ref
        T = (E_trn) * np.real((kz_trn * i_med[1]) / (kz_ref * t_med[1]))
        E_ref, _, E_trn, _ = S_Global_i.fields(p, kz_ref, kz_trn)
        S_Global_Before_11, S_Global_Before_12, S_Global_Before_21, S_Global_Before_22 = S_Global_Before
        S_Global_After_11, S_Global_After_12, S_Global_After_21, S_Global_After_22 = S_Global_After
        c_ref_m = inv(S_ref_12) @ (E_ref - S_ref_11 @ p)
        c_ref_p = S_ref_21 @ p + S_ref_22 @ c_ref_m
        # Avoid S_Pre_Trn singular matrix for inv
        if np.any(S_Pre_Trn_12):
            c_trn_m = inv(S_Pre_Trn_12) @ (E_ref - S_Pre_Trn_11 @ p)
        else:
            non_invertable_matrix = True
            c_trn_m = np.array([0, 0])
        c_trn_p = S_Pre_Trn_21 @ p + S_Pre_Trn_22 @ c_trn_m
        sum_c_trn_p = np.sum(np.abs(c_trn_p)**2)
        sum_c_trn_m = np.sum(np.abs(c_trn_m)**2)
        sum_c_ref_p = np.sum(np.abs(c_ref_p)**2)
        sum_c_ref_m = np.sum(np.abs(c_ref_m)**2)
        int_power = sum_c_ref_p - sum_c_ref_m - sum_c_trn_p + sum_c_trn_m
        # Determine the mode coefficients just before the wanted layer
        if np.any(S_Global_Before_12):
            c_left_m = inv(S_Global_Before_12) @ (E_ref -
                                                  S_Global_Before_11 @ p)
        else:
            non_invertable_matrix = True
            c_left_m = np.array([0, 0])
        c_left_p = S_Global_Before_21 @ p + S_Global_Before_22 @ c_left_m
        # Determine the mode coefficients just after the wanted layer
        if np.any(S_Global_After_12):
            c_right_m = inv(S_Global_After_12) @ (E_ref -
                                                  S_Global_After_11 @ p)
        else:
            non_invertable_matrix = True
            c_right_m = np.array([0, 0])
        c_right_p = S_Global_After_21 @ p + S_Global_After_22 @ c_right_m
        # Determine the %abs for a particular layer in regard to the total abs
        sum_left_p = np.sum(np.abs(c_left_p)**2)
        sum_left_m = np.sum(np.abs(c_left_m)**2)
        sum_right_p = np.sum(np.abs(c_right_p)**2)
        sum_right_m = np.sum(np.abs(c_right_m)**2)
        i_abs = (sum_left_p + sum_right_m - sum_left_m -
                 sum_right_p) / int_power
        Abs.append(i_abs * (1 - R - T))
    if non_invertable_matrix:
        logging.warning(f"Non Invertable Matrix Found in.. Considered 0")
    return np.array(Abs)
