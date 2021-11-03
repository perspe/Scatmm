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
from enum import Enum, auto
import logging
from typing import Any, List, Tuple, Union

import numpy as np
from numpy.linalg import inv
import numpy.typing as npt
from scipy.interpolate import interp1d

sim_config = {
        "theta": 0.0,
        "phi": 0.0,
        "lmb": 500.0,
        "pol": (1, 0),
        "i_med": (1, 1),
        "t_med": (1, 1)
}

class MatOutsideBounds(Exception):
    """ Exception for material outside bounds """
    def __init__(self, material, wavelength):
        self.message = f"Simulation wavelength ({wavelength}) is outside " +\
            f"the defined oustide bounds for '{material}'"
        super().__init__(self.message)


class InvalidParameter(Exception):
    pass


# Alias to associate type for scattering matrix
S_Matrix = npt.NDArray[np.complex128]
C_Matrix = npt.NDArray[np.complex128]


class SMMType(Enum):
    """ Type to differenciate the different smm types """
    TRN = auto()
    REF = auto()
    NORM = auto()


class SMMBase:
    """ Base class to store the SMM elements and do fundamental operations """
    def __init__(self, S_11: S_Matrix, S_12: S_Matrix, S_21: S_Matrix,
                 S_22: S_Matrix) -> None:
        self.S_11: S_Matrix = S_11
        self.S_12: S_Matrix = S_12
        self.S_21: S_Matrix = S_21
        self.S_22: S_Matrix = S_22
        # logging.debug(f"Initializing scattering matrix:\n {self}")

    def __mul__(self, other):
        """ Implement Syntax Sugar to perform the Redheffer Product """
        D = self.S_12 @ inv(np.eye(2) - other.S_11 @ self.S_22)
        F = other.S_21 @ inv(np.eye(2) - self.S_22 @ other.S_11)
        S_11_AB = self.S_11 + D @ other.S_11 @ self.S_21
        S_12_AB = D @ other.S_12
        S_21_AB = F @ self.S_21
        S_22_AB = other.S_22 + F @ self.S_22 @ other.S_12
        return SMMBase(S_11_AB, S_12_AB, S_21_AB, S_22_AB)

    def __imul__(self, other):
        """ Syntax Sugar for *= Operation """
        return self * other

    def __repr__(self) -> str:
        """ SMM representation """
        line_1 = f"{self.S_11[0, 0]} {self.S_11[0, 1]} | "\
            f"{self.S_12[0, 0]} {self.S_12[0, 1]}\n"
        line_2 = f"{self.S_11[1, 0]} {self.S_11[1, 1]} | "\
            f"{self.S_12[1, 0]} {self.S_12[1, 1]}\n"
        line_3 = "-" * len(line_1) + "\n"
        line_4 = f"{self.S_21[0, 0]} {self.S_21[0, 1]} | "\
            f"{self.S_22[0, 0]} {self.S_22[0, 1]}\n"
        line_5 = f"{self.S_21[1, 0]} {self.S_21[1, 1]} | "\
            f"{self.S_22[1, 0]} {self.S_22[1, 1]}\n"
        return line_1 + line_2 + line_3 + line_4 + line_5


class SMatrix(SMMBase):
    """ Calculate and store the SMatrix for normal layer or ref/trn layer """
    def __init__(self,
                 V0: C_Matrix,
                 k0: float,
                 kx: float,
                 ky: float,
                 thickness: float,
                 e: complex,
                 u: complex = 1,
                 type: SMMType = SMMType.NORM):
        """ Initialize variables """
        self.thickness: float = thickness
        self.e: complex = e
        self.u: complex = u
        self.k0: float = k0
        self.kx: float = kx
        self.ky: float = ky
        self._mat_properties()
        if type == SMMType.NORM:
            S_11, S_12, S_21, S_22 = self._smm_norm(V0)
        elif type == SMMType.TRN:
            S_11, S_12, S_21, S_22 = self._smm_trn(V0)
        elif type == SMMType.REF:
            S_11, S_12, S_21, S_22 = self._smm_ref(V0)
        else:
            logging.error("Invalid SMatrix Type")
            raise Exception("Invalid SMM type. Allowed: NORM/TRN/REF")
        super().__init__(S_11, S_12, S_21, S_22)

    def _mat_properties(self) -> None:
        """ Determine the material properties for a particular layer"""
        mu_eps = self.e * self.u
        Q_i = (1 / self.u) * np.array([[
            self.kx * self.ky, mu_eps - self.kx**2
        ], [self.ky**2 - mu_eps, -self.kx * self.ky]])
        Omega_i = 1j * np.sqrt(mu_eps - self.kx**2 - self.ky**2) * np.eye(2)
        Vi = Q_i @ inv(Omega_i)
        self.Vi: C_Matrix = Vi
        self.Omega_i: C_Matrix = Omega_i

    def _smm_norm(self, V0: C_Matrix):
        """ Scattering matrix for a standard layer """
        iVi_V0 = inv(self.Vi) @ V0
        Ai = np.eye(2) + iVi_V0
        Bi = np.eye(2) - iVi_V0
        Xi = np.eye(2) * np.exp(self.Omega_i * self.k0 * self.thickness)
        iAi = inv(Ai)
        X_BiA_X = Xi @ Bi @ iAi @ Xi
        inv_fact = inv(Ai - X_BiA_X @ Bi)
        S_11 = inv_fact @ (X_BiA_X @ Ai - Bi)
        S_12 = inv_fact @ Xi @ (Ai - Bi @ iAi @ Bi)
        return S_11, S_12, S_12, S_11

    def _smm_trn(self, V0: C_Matrix):
        """ Calculate the smm for the transmission layer """
        iV_0_V_trn = inv(V0) @ self.Vi
        A_trn = np.eye(2) + iV_0_V_trn
        B_trn = np.eye(2) - iV_0_V_trn
        iA_trn = inv(A_trn)
        S_11 = B_trn @ iA_trn
        S_12 = 0.5 * (A_trn - B_trn @ iA_trn @ B_trn)
        S_21 = 2 * iA_trn
        S_22 = -iA_trn @ B_trn
        return S_11, S_12, S_21, S_22

    def _smm_ref(self, V0: C_Matrix):
        """Calculate the smm for the reflection layer"""
        iV_0_V_ref = inv(V0) @ self.Vi
        A_ref = np.eye(2) + iV_0_V_ref
        B_ref = np.eye(2) - iV_0_V_ref
        iA_ref = inv(A_ref)
        S_11 = -iA_ref @ B_ref
        S_12 = 2 * iA_ref
        S_21 = 0.5 * (A_ref - B_ref @ iA_ref @ B_ref)
        S_22 = B_ref @ iA_ref
        return S_11, S_12, S_21, S_22


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
    def __init__(self, name: str, thickness: float,
                 lmb: npt.NDArray,
                 n_array: npt.NDArray,
                 k_array: npt.NDArray, **kwargs) -> None:
        self.name: str = name
        self.lmb: list = [np.min(lmb), np.max(lmb)]
        self.thickness: float = thickness
        self.n = interp1d(lmb, n_array, **kwargs)
        self.k = interp1d(lmb, k_array, **kwargs)
        logging.debug(f"Layer: {self} created...")

    def e_value(self, lmb: npt.ArrayLike) -> npt.ArrayLike:
        """ Calculate e_values for a range of wavelengths """
        try:
            e_data: npt.ArrayLike = (self.n(lmb) + 1j * self.k(lmb))**2
        except ValueError:
            if isinstance(lmb, float):
                raise MatOutsideBounds(self.name, f"{lmb}")
            elif isinstance(lmb, np.ndarray):
                raise MatOutsideBounds(self.name, f"{lmb.min()}, {lmb.max()}")
            else:
                raise Exception("Invalid wavelength value")
        return e_data

    def __repr__(self) -> str:
        return f"{self.name}({self.thickness} nm)"


Layer_Type = Union[Layer1D, Layer3D]


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
    logging.debug(f"Initialization Values for SMM: {k0}:{kx}:{ky}\n{V0}\n{p}")
    return k0, kx, ky, V0, p


def _e_fields(S_Global, inc_p_l, inc_p_r, kx, ky, kz_ref, kz_trn):
    """ Determine the Electric Fields resulting from the SMM calculation """
    E_ref = S_Global.S_11 @ inc_p_l + S_Global.S_12 @ inc_p_r
    E_trn = S_Global.S_21 @ inc_p_l + S_Global.S_22 @ inc_p_r
    Ez_ref = -(kx * E_ref[0] + ky * E_ref[1]) / kz_ref
    Ez_trn = -(kx * E_trn[0] + ky * E_trn[1]) / kz_trn
    logging.debug(f"E Fields: {E_ref}:{E_trn}:{Ez_ref}:{Ez_trn}")
    return E_ref, E_trn, Ez_ref, Ez_trn


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
    logging.info(f"SMM... Start...")
    if not isinstance(lmb, (float, int)) or not isinstance(
            theta, (float, int)) or not isinstance(phi, (int, float)):
        logging.warning("Invalid input parameter...")
        raise InvalidParameter("Invalid Input parameter")
    # Inicialize necessary values
    k0, kx, ky, V0, p = _initialize_smm(theta, phi, lmb, pol, i_med)
    S_trn = SMatrix(V0, k0, kx, ky, 0, t_med[0], t_med[1], SMMType.TRN)
    S_ref = SMatrix(V0, k0, kx, ky, 0, i_med[0], i_med[1], SMMType.REF)
    S_Global = S_ref
    logging.debug("Looping through layers...")
    for layer in layer_list:
        e_value: Any = layer.e_value(lmb)
        S_Layer = SMatrix(V0, k0, kx, ky, layer.thickness, e_value)
        S_Global *= S_Layer
    S_Global = S_Global * S_trn
    kz_ref = np.sqrt(i_med[0] * i_med[1] - kx**2 - ky**2)
    kz_trn = np.sqrt(t_med[0] * t_med[1] - kx**2 - ky**2)
    E_ref, E_trn, Ez_ref, Ez_trn = _e_fields(S_Global, p, [0, 0], kx, ky,
                                             kz_ref, kz_trn)
    R = abs(E_ref[0])**2 + abs(E_ref[1])**2 + abs(Ez_ref)**2
    T = (abs(E_trn[0])**2 + abs(E_trn[1])**2 + abs(Ez_trn)**2) * np.real(
        (kz_trn * i_med[1]) / (kz_ref * t_med[1]))
    logging.info("Done...")
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
    logging.debug(f"SMM for {theta}:{phi}:{pol}:{i_med}:{t_med}")
    logging.debug(f"WVL: {lmb}")
    logging.info(f"SMM... Start...")
    if not isinstance(theta, (int, float)) or not isinstance(phi, (int, float)):
        raise InvalidParameter("Invalid Input parameter")
    if override_thick is not None:
        logging.info("Overriding Thicknesses")
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
    logging.debug("Looping through layers")
    for k0_i, layer_data in zip(k0, layer_data.T):
        S_trn = SMatrix(V0, k0_i, kx, ky, 0, t_med[0], t_med[1], SMMType.TRN)
        S_ref = SMatrix(V0, k0_i, kx, ky, 0, i_med[0], i_med[1], SMMType.REF)
        S_Global_i = S_ref
        for index, layer in enumerate(layer_list):
            S_Layer = SMatrix(V0, k0_i, kx, ky, layer.thickness,
                              layer_data[index])
            S_Global_i *= S_Layer
        S_Global_i = S_Global_i * S_trn
        E_ref, E_trn, Ez_ref, Ez_trn = _e_fields(S_Global_i, p, [0, 0], kx, ky,
                                                 kz_ref, kz_trn)
        R.append(abs(E_ref[0])**2 + abs(E_ref[1])**2 + abs(Ez_ref)**2)
        T.append(
            (abs(E_trn[0])**2 + abs(E_trn[1])**2 + abs(Ez_trn)**2) * np.real(
                (kz_trn * i_med[1]) / (kz_ref * t_med[1])))
    logging.info("Done...")
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
    logging.debug(f"SMM for {phi}:{lmb}:{pol}:{i_med}:{t_med}")
    logging.debug(f"Angle: {theta}")
    logging.info(f"SMM... Start...")
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
        S_trn = SMatrix(V0, k0, kx, ky, 0, t_med[0], t_med[1], SMMType.TRN)
        S_ref = SMatrix(V0, k0, kx, ky, 0, i_med[0], i_med[1], SMMType.REF)
        S_Global = S_ref
        for layer in layer_list:
            e_value: Any = layer.e_value(lmb)
            S_Layer = SMatrix(V0, k0, kx, ky, layer.thickness, e_value)
            S_Global *= S_Layer
        S_Global = S_ref * S_Global * S_trn
        kz_ref = np.sqrt(i_med[0] * i_med[1] - kx**2 - ky**2)
        kz_trn = np.sqrt(t_med[0] * t_med[1] - kx**2 - ky**2)
        E_ref, E_trn, Ez_ref, Ez_trn = _e_fields(S_Global, p, [0, 0], kx, ky,
                                                 kz_ref, kz_trn)
        R.append(abs(E_ref[0])**2 + abs(E_ref[1])**2 + abs(Ez_ref)**2)
        T.append(
            (abs(E_trn[0])**2 + abs(E_trn[1])**2 + abs(Ez_trn)**2) * np.real(
                (kz_trn * i_med[1]) / (kz_ref * t_med[1])))
    logging.info("Done..")
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
    logging.debug(f"SMM for {theta}:{phi}:{pol}:{i_med}:{t_med}")
    logging.debug(f"WVL: {lmb}")
    logging.info(f"SMM... Start...")
    if not isinstance(theta, (int, float)) or not isinstance(phi, (int, float)):
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
    Abs = []
    # Loop through all wavelengths and layers
    for k0_i, layer_data in zip(k0, layer_data.T):
        S_trn = SMatrix(V0, k0_i, kx, ky, 0, t_med[0], t_med[1], SMMType.TRN)
        S_ref = SMatrix(V0, k0_i, kx, ky, 0, i_med[0], i_med[1], SMMType.REF)
        S_Global_i = S_ref
        S_Global_After = S_ref
        S_Global_Before = S_ref
        for layer_index, layer in enumerate(layer_list):
            if layer_index == layer_i:
                logging.debug("SMatrix just before wanted layer")
                S_Global_Before = S_Global_i
            S_Layer = SMatrix(V0, k0_i, kx, ky, layer.thickness,
                              layer_data[layer_index])
            S_Global_i *= S_Layer
            if layer_index == layer_i:
                logging.debug("SMatrix just after wanted layer")
                S_Global_After = S_Global_i
        S_Pre_Trn = S_Global_i
        S_Global_i = S_Global_i * S_trn
        # Determine the total absorption of the device
        E_ref, E_trn, Ez_ref, Ez_trn = _e_fields(S_Global_i, p, [0, 0], kx, ky,
                                                 kz_ref, kz_trn)
        R = abs(E_ref[0])**2 + abs(E_ref[1])**2 + abs(Ez_ref)**2
        T = (abs(E_trn[0])**2 + abs(E_trn[1])**2 + abs(Ez_trn)**2) * np.real(
            (kz_trn * i_med[1]) / (kz_ref * t_med[1]))

        # Determine the total power inside the device
        logging.debug("Calculating power inside the device")
        c_ref_m = inv(S_ref.S_12) @ (E_ref - S_ref.S_11 @ p)
        c_ref_p = S_ref.S_21 @ p + S_ref.S_22 @ c_ref_m
        # Avoid S_Pre_Trn singular matrix for inv
        if np.any(S_Pre_Trn.S_12):
            c_trn_m = inv(S_Pre_Trn.S_12) @ (E_ref - S_Pre_Trn.S_11 @ p)
        else:
            c_trn_m = np.array([0, 0])
        c_trn_p = S_Pre_Trn.S_21 @ p + S_Pre_Trn.S_22 @ c_trn_m
        sum_c_trn_p = np.sum(np.abs(c_trn_p)**2)
        sum_c_trn_m = np.sum(np.abs(c_trn_m)**2)
        sum_c_ref_p = np.sum(np.abs(c_ref_p)**2)
        sum_c_ref_m = np.sum(np.abs(c_ref_m)**2)
        int_power = sum_c_ref_p - sum_c_ref_m - sum_c_trn_p + sum_c_trn_m
        # Determine the mode coefficients just before the wanted layer
        if np.any(S_Global_Before.S_12):
            c_left_m = inv(
                S_Global_Before.S_12) @ (E_ref - S_Global_Before.S_11 @ p)
        else:
            c_left_m = np.array([0, 0])
        c_left_p = S_Global_Before.S_21 @ p + S_Global_Before.S_22 @ c_left_m
        # Determine the mode coefficients just after the wanted layer
        if np.any(S_Global_After.S_12):
            c_right_m = inv(
                S_Global_After.S_12) @ (E_ref - S_Global_After.S_11 @ p)
        else:
            c_right_m = np.array([0, 0])
        c_right_p = S_Global_After.S_21 @ p + S_Global_After.S_22 @ c_right_m
        # Determine the %abs for a particular layer in regard to the total abs
        sum_left_p = np.sum(np.abs(c_left_p)**2)
        sum_left_m = np.sum(np.abs(c_left_m)**2)
        sum_right_p = np.sum(np.abs(c_right_p)**2)
        sum_right_m = np.sum(np.abs(c_right_m)**2)
        i_abs = (sum_left_p + sum_right_m - sum_left_m -
                 sum_right_p) / int_power
        Abs.append(i_abs * (1 - R - T))
    logging.info("Done...")
    return np.array(Abs)


if __name__ == '__main__':
    n_comp = 5
    lmb = np.linspace(0.5, 1.5, 100)
    theta = np.radians(np.linspace(0, 89))
    phi = np.radians(23)
    p = (1, 0)
    inc_medium = (1, 1)
    trn_medium = (1, 1)
    layer1 = Layer1D("l1", 0.1, 1.5, 0.1)
    layer2 = Layer1D("l2", 0.1, 1.3, 0.03)
    layer3 = Layer1D("l3", 2, 2.5, 0.1)
    # Test angular results
    R, T = smm_angle([layer1, layer2, layer3], theta, phi, lmb[0], p,
                     inc_medium, trn_medium)
    # print(R, T)
    # # Test absorption
    # abs1 = smm_layer([layer1, layer2, layer3], 1,
    #                  theta[3], phi, lmb, p, inc_medium, trn_medium)
    # abs2 = smm_layer([layer1, layer2, layer3], 2,
    #                  theta[3], phi, lmb, p, inc_medium, trn_medium)
    # abs3 = smm_layer([layer1, layer2, layer3], 3,
    #                  theta, phi, lmb, p, inc_medium, trn_medium)
    # print(abs1+abs2+abs3)
    R, T = smm([layer1, layer2, layer3], theta[3], phi, lmb[0], p, inc_medium,
               trn_medium)
    # print(1-R-T)
