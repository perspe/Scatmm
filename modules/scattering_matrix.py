"""
File with the optimized version of the Scattering Matrix Script

Has 2 main functions:
    smm - Calculates R and T for a specific wavelength
    smm_broadband - Calculates R and T for a wavelength range
"""
# TODO: Rebuild this portion of the code  <11-10-21, Miguel> #

# Basic modules
import time
from enum import Enum, auto
import numpy as np
from numpy.linalg import inv
from scipy.interpolate import interp1d


class SMMType(Enum):
    """ Type to differenciate the different smm types """
    TRN = auto()
    REF = auto()
    NORM = auto()


class SMMBase():
    """ Base class to store the SMM elements and do necessary operations """

    def __init__(self, S_11, S_12, S_21, S_22):
        self.S_11 = S_11
        self.S_12 = S_12
        self.S_21 = S_21
        self.S_22 = S_22

    def __mul__(self, other):
        """ Implement Syntax Sugar to perform the Redhaffer Product """
        D = self.S_12 @ inv(np.eye(2) - other.S_11 @ self.S_22)
        F = other.S_21 @ inv(np.eye(2) - self.S_22 @ other.S_11)
        S_11_AB = self.S_11 + D @ other.S_11 @ self.S_21
        S_12_AB = D @ other.S_12
        S_21_AB = F @ self.S_21
        S_22_AB = other.S_22 + F @ self.S_22 @ other.S_12
        return SMMBase(S_11_AB, S_12_AB, S_21_AB, S_22_AB)

    def __imul__(self, other):
        """ Implement Syntax Sugar to perform the Redhaffer Product """
        return self * other

    def __repr__(self):
        """ SMM representation """
        line_1 = f"{self.S_11[0, 0]} {self.S_11[0, 1]} | "\
            f"{self.S_12[0, 0]} {self.S_12[0, 1]}\n"
        line_2 = f"{self.S_11[1, 0]} {self.S_11[1, 1]} | "\
            f"{self.S_12[1, 0]} {self.S_12[1, 1]}\n"
        line_3 = "-"*len(line_1)+"\n"
        line_4 = f"{self.S_21[0, 0]} {self.S_21[0, 1]} | "\
            f"{self.S_22[0, 0]} {self.S_22[0, 1]}\n"
        line_5 = f"{self.S_21[1, 0]} {self.S_21[1, 1]} | "\
            f"{self.S_22[1, 0]} {self.S_22[1, 1]}\n"
        return line_1+line_2+line_3+line_4+line_5


class SMM(SMMBase):
    """ Base class to perform SMM calculations """

    def __init__(self, V0, k0, kx, ky, thickness, e, u=1, type=SMMType.NORM):
        self.thickness = thickness
        self.e = e
        self.u = u
        self.k0 = k0
        self.kx = kx
        self.ky = ky
        if type == SMMType.NORM:
            S_11, S_12, S_21, S_22 = self._smm_norm(V0)
        elif type == SMMType.TRN:
            S_11, S_12, S_21, S_22 = self._smm_trn(V0)
        elif type == SMMType.REF:
            S_11, S_12, S_21, S_22 = self._smm_ref(V0)
        else:
            raise Exception("Invalid SMM type. Allowed: NORM/TRN/REF")
        super().__init__(S_11, S_12, S_21, S_22)

    def _mat_properties(self):
        """ Determine the material properties for a particular layer"""
        mu_eps = self.e * self.u
        Q_i = (1 / self.u) * np.array([[self.kx * self.ky,
                                        mu_eps - self.kx**2],
                                       [self.ky**2 - mu_eps,
                                        - self.kx * self.ky]])
        Omega_i = 1j * np.sqrt(mu_eps - self.kx**2 - self.ky**2) * np.eye(2)
        V_i = Q_i @ inv(Omega_i)
        return Omega_i, V_i

    def _smm_norm(self, V0):
        """ Calculate the normal smm """
        Omega_i, Vi = self._mat_properties()
        iVi_V0 = inv(Vi) @ V0
        Ai = np.eye(2) + iVi_V0
        Bi = np.eye(2) - iVi_V0
        Xi = np.eye(2) * np.exp(Omega_i * self.k0 * self.thickness)
        iAi = inv(Ai)
        X_BiA_X = Xi @ Bi @ iAi @ Xi
        inv_fact = inv(Ai - X_BiA_X @ Bi)
        S_11 = inv_fact @ (X_BiA_X @ Ai - Bi)
        S_12 = inv_fact @ Xi @ (Ai - Bi @ iAi @ Bi)
        return S_11, S_12, S_12, S_11

    def _smm_trn(self, V0):
        """ Calculate the smm for the transmission region """
        _, V_trn = self._mat_properties()
        iV_0_V_trn = inv(V0) @ V_trn
        A_trn = np.eye(2) + iV_0_V_trn
        B_trn = np.eye(2) - iV_0_V_trn
        iA_trn = inv(A_trn)
        S_11 = B_trn @ iA_trn
        S_12 = 0.5 * (A_trn - B_trn @ iA_trn @ B_trn)
        S_21 = 2 * iA_trn
        S_22 = -iA_trn @ B_trn
        return S_11, S_12, S_21, S_22

    def _smm_ref(self, V0):
        """Calculate the smm for the reflection region"""
        _, V_ref = self._mat_properties()
        iV_0_V_ref = inv(V0) @ V_ref
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
        name (str): Name for the layer
        thickness (float - nm): Layer thickness
        lmb (nm)/n_array/k_array (array): Arrays with the
                                          basic info for each layer
        u_array: Permeability data (1 as default)
    """

    def __init__(self, thickness, lmb, n_val, k_val):
        self.thickness = thickness
        self.lmb = lmb
        self.n = n_val
        self.k = k_val

    def e_value(self, lmb):
        """ Calculate e_values for a range of wavelengths """
        if lmb != self.lmb:
            raise Exception(
                "Material defined outside provided wavelength value")
        return (self.n + 1j*self.k)**2


class Layer3D():
    """
    Layer for single wavelength materials
    Args:
        name (str): Name for the layer
        thickness (float - nm): Layer thickness
        lmb (nm)/n_array/k_array (array): Arrays with the
                                          basic info for each layer
        u_array: Permeability data (1 as default)
    """

    def __init__(self, thickness, lmb, n_array, k_array, **kwargs):
        self.lmb = [np.min(lmb), np.max(lmb)]
        self.thickness = thickness
        self.n = interp1d(lmb, n_array, **kwargs)
        self.k = interp1d(lmb, k_array, **kwargs)

    def e_value(self, lmb):
        """ Calculate e_values for a range of wavelengths """
        try:
            e_data = (self.n(lmb) + 1j * self.k(lmb))**2
        except ValueError:
            raise Exception(
                "Material defined outside provided wavelength range")
        return e_data


def _initialize_smm(theta, phi, lmb, pol, inc_medium, trn_medium):
    """ Initialize the parameters necessary for the smm calculation """
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

    # Initialize global SMM
    SGlobal = SMMBase(np.zeros((2, 2)), np.eye(2), np.eye(2), np.zeros((2, 2)))

    return k0, kx, ky, V0, p, SGlobal


def _e_fields(S_Global, p, kx, ky, kz_ref, kz_trn):
    """ Determine the Electric Fields resulting from the SMM calculation """
    E_ref, E_trn = S_Global.S_11 @ p, S_Global.S_21 @ p
    E_z_ref = -(kx * E_ref[0] + ky * E_ref[1]) / kz_ref
    E_z_trn = -(kx * E_trn[0] + ky * E_trn[1]) / kz_trn
    return E_ref, E_trn, E_z_ref, E_z_trn


def smm_broadband(layer_list, theta, phi, lmb, pol, i_med, t_med):
    """
    SMM for broadband simulation
    Args:
        layer_list: List of Layer objects with the info for all layers
        theta/phi: Incidence angles
        lmb: Array with wavelengths for simulation
        pol: Tuple with the TM/TE polarization components
        i_med/t_med: Data for the reflection and transmission media
    Returns:
        R, T: Arrays with the Reflection and transmission for the layer setup
    """
    k0, kx, ky, V0, p, S_Global = _initialize_smm(
        theta, phi, lmb, pol, i_med, t_med)
    kz_ref = np.sqrt(i_med[0]*i_med[1] - kx**2 - ky**2)
    kz_trn = np.sqrt(t_med[0]*t_med[1] - kx**2 - ky**2)
    # This is a simplification to determine all the values in beforehand
    layer_data = np.array([layer_i.e_value(lmb) for layer_i in layer_list])
    R = []
    T = []
    for lmb_i, k0_i, layer_data in zip(lmb, k0, layer_data.T):
        S_trn = SMM(V0, k0_i, kx, ky, 0, t_med[0], t_med[1], SMMType.TRN)
        S_ref = SMM(V0, k0_i, kx, ky, 0, i_med[0], i_med[1], SMMType.REF)
        S_Global_i = S_Global
        for index, layer in enumerate(layer_list):
            S_Layer = SMM(V0, k0_i, kx, ky, layer.thickness, layer_data[index])
            S_Global_i *= S_Layer
        S_Global_i = S_ref * S_Global_i * S_trn
        E_ref, E_trn, Ez_ref, Ez_trn = _e_fields(
            S_Global_i, p, kx, ky, kz_ref, kz_trn)
        R.append(abs(E_ref[0])**2 + abs(E_ref[1])**2 + abs(Ez_ref)**2)
        T.append(
                (abs(E_trn[0])**2 + abs(E_trn[1])**2 + abs(Ez_trn)**2) * np.real(
                    (kz_trn * i_med[1]) / (kz_ref * t_med[1])))
    return np.array(R), np.array(T)


def smm_angle(layer_list, theta, phi, lmb, pol, ref_medium, trn_medium):
    """ SMM for broad angle simulations """
    pass


def smm(layer_list, lmb, theta, phi, pol, i_med, t_med):
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
    # Inicialize necessary values
    k0, kx, ky, V0, p, S_Global = _initialize_smm(
        theta, phi, lmb, pol, i_med, t_med)
    S_trn = SMM(V0, k0, kx, ky, 0,
                t_med[0], t_med[1], SMMType.TRN)
    S_ref = SMM(V0, k0, kx, ky, 0,
                i_med[0], i_med[1], SMMType.REF)
    for layer in layer_list:
        S_Layer = SMM(V0, k0, kx, ky, layer.thickness, layer.e_value(lmb))
        S_Global *= S_Layer
    S_Global = S_ref * S_Global * S_trn
    kz_ref = np.sqrt(i_med[0]*i_med[1] - kx**2 - ky**2)
    kz_trn = np.sqrt(t_med[0]*t_med[1] - kx**2 - ky**2)
    E_ref, E_trn, Ez_ref, Ez_trn = _e_fields(
        S_Global, p, kx, ky, kz_ref, kz_trn)
    R = abs(E_ref[0])**2 + abs(E_ref[1])**2 + abs(Ez_ref)**2
    T = (abs(E_trn[0])**2 + abs(E_trn[1])**2 + abs(Ez_trn)**2) * np.real(
        (kz_trn * i_med[1]) / (kz_ref * t_med[1]))
    return R, T


""" Separator between before and after """


def redhaff_prod(S_11_A, S_12_A, S_21_A, S_22_A, S_11_B, S_12_B, S_21_B,
                 S_22_B):
    """Redhaffer star product between 2 scattering matrixes

    Args:
        S_A: scattering matrix A elements (11, 12, 21, 22)
    S_B: Scattering matrix B elements (11, 12, 21, 22)

    Return:
        S_AB: Scattering matrix elements of the multiplication (11,12,21,22)
    """
    D = S_12_A @ inv(np.eye(2) - S_11_B @ S_22_A)
    F = S_21_B @ inv(np.eye(2) - S_22_A @ S_11_B)
    S_11_AB = S_11_A + D @ S_11_B @ S_21_A
    S_12_AB = D @ S_12_B
    S_21_AB = F @ S_21_A
    S_22_AB = S_22_B + F @ S_22_A @ S_12_B

    return S_11_AB, S_12_AB, S_21_AB, S_22_AB


def material_elements(mu_r, eps_r, k_x, k_y):
    """Elements for determining scattering matrix elements

    Args:
    mu_r (float): material magnectic permeability
    eps_r (complex): material dielectric permeability
    k_x (float): wavevector x component
    k_y (float): wavevector y component

    Return:
    k_i_z_norm,Omega_i,V_i
    """
    mu_eps = mu_r * eps_r
    Q_i = (1 / mu_r) * np.array([[k_x * k_y, mu_eps - k_x**2],
                                 [k_y**2 - mu_eps, -k_x * k_y]])
    Omega_i = 1j * np.sqrt(mu_eps - k_x**2 - k_y**2) * np.eye(2)
    V_i = Q_i @ inv(Omega_i)
    return Omega_i, V_i


def smm_elements(V_0, V_i, lam_i, k_0, Li):
    """Determine scattering matrix elements for a single layer

    Args:
    V_0, V_i (matrix): V matrix for free space and material layer
    lam_i (complex): eigenvalues for specific layer
    k_0 (float): wavevector magnitude
    Li (float): layer thickness

    Return:
    S11, S12: matrix elements S11 and S22 of the scattering matrix
    Note: S11 = S22 and S12 = S21
    """
    iV_i_V0 = inv(V_i) @ V_0
    A_i = np.eye(2) + iV_i_V0
    B_i = np.eye(2) - iV_i_V0
    X_i = np.eye(2) * np.exp(lam_i * k_0 * Li)
    iA_i = inv(A_i)
    X_B_iA_X = X_i @ B_i @ iA_i @ X_i
    i_fact = inv(A_i - X_B_iA_X @ B_i)
    S_11 = i_fact @ (X_B_iA_X @ A_i - B_i)
    S_12 = i_fact @ X_i @ (A_i - B_i @ iA_i @ B_i)
    return S_11, S_12


def smm_ref(V_0, V_ref):
    """Calculate the scattering matrix element for the reflection region"""
    iV_0_V_ref = inv(V_0) @ V_ref
    A_ref = np.eye(2) + iV_0_V_ref
    B_ref = np.eye(2) - iV_0_V_ref
    iA_ref = inv(A_ref)
    S_11 = -iA_ref @ B_ref
    S_12 = 2 * iA_ref
    S_21 = 0.5 * (A_ref - B_ref @ iA_ref @ B_ref)
    S_22 = B_ref @ iA_ref
    return S_11, S_12, S_21, S_22


def smm_trn(V_0, V_trn):
    """Calculate the scattering matrix element for the transmission region"""
    iV_0_V_trn = inv(V_0) @ V_trn
    A_trn = np.eye(2) + iV_0_V_trn
    B_trn = np.eye(2) - iV_0_V_trn
    iA_trn = inv(A_trn)
    S_11 = B_trn @ iA_trn
    S_12 = 0.5 * (A_trn - B_trn @ iA_trn @ B_trn)
    S_21 = 2 * iA_trn
    S_22 = -iA_trn @ B_trn
    return S_11, S_12, S_21, S_22


def smm_old(theta,
            phi,
            e,
            thickness,
            lmb,
            pol=(1, 0),
            inc_medium=(1, 1),
            trn_medium=(1, 1),
            **kwargs):
    """Calculate reflection and transmission for a multi-layer
    configuration of materials for a single wavelength

    Args:
    theta, phi (float): angles relating to the incident wave
                         (azimutal and polar angles)
    e (array): 1D array with complex dielectric constant for
                        the different layers
    thickness (array): thicknesses of the different layers
    lmb (float): wavelength
    pol (tuple): TM and TE polarizaton default (1, 0)
    inc_medium (tuple): override incident medium refractive index
                        (default is (e = 1,u =1))
    trn_medium (tuple): override transmision medium refractive index
                        (default is (e = 1,u =1))
    **kwargs:
        u (list): list with u values, if not provided defaults to 1

    Return:
    R, T: reflection and transmission
    """

    # Check for input for u_matrix, otherwise set default as 1
    if kwargs.get("u"):
        u = np.array(kwargs.get('u'))
    else:
        u = np.ones_like(e)

    # Inicialize necessary values
    k0 = 2 * np.pi / lmb  # Absolute value of wavevector
    kx = np.sqrt(inc_medium[0] * inc_medium[1]) * np.sin(theta) * np.cos(phi)
    ky = np.sqrt(inc_medium[0] * inc_medium[1]) * np.sin(theta) * np.sin(phi)
    k_z_0_ref = np.sqrt(inc_medium[0]*inc_medium[1] - kx**2 - ky**2)
    k_z_0_trn = np.sqrt(trn_medium[0]*trn_medium[1] - kx**2 - ky**2)

    # Free Space parameters
    Q_0 = np.array([[kx * ky, 1 + ky**2], [-(1 + kx**2), -kx * ky]])
    V_0 = 0 - Q_0 * 1j

    # Define the polarization vectors
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
    p = p_vector[[0, 1]]  # Reduced polarization vector

    # Inicialize global scattering matrix
    S_global_11, S_global_12, = np.zeros((2, 2)), np.eye(2)
    S_global_21, S_global_22 = np.eye(2), np.zeros((2, 2))

    # Update the Scattering Matrix according to input layers
    for t_i, e_i, u_i in zip(thickness, e, u):
        Omega_i, V_i = material_elements(u_i, e_i, kx, ky)
        S_11, S_12 = smm_elements(V_0, V_i, Omega_i, k0, t_i)
        S_global_11, S_global_12, S_global_21, S_global_22 = redhaff_prod(
            S_global_11, S_global_12, S_global_21, S_global_22, S_11, S_12,
            S_12, S_11)

    # Add the Reflection matrix component
    _, V_ref = material_elements(inc_medium[1], inc_medium[0], kx, ky)
    S_11_ref, S_12_ref, S_21_ref, S_22_ref = smm_ref(V_0, V_ref)
    S_global_11, S_global_12, S_global_21, S_global_22 = redhaff_prod(
        S_11_ref, S_12_ref, S_21_ref, S_22_ref, S_global_11, S_global_12,
        S_global_21, S_global_22)

    _, V_trn = material_elements(trn_medium[1], trn_medium[0], kx, ky)
    S_11_trn, S_12_trn, S_21_trn, S_22_trn = smm_trn(V_0, V_trn)
    S_global_11, S_global_12, S_global_21, S_global_22 = redhaff_prod(
        S_global_11, S_global_12, S_global_21, S_global_22, S_11_trn, S_12_trn,
        S_21_trn, S_22_trn)

    # Update field components from incident power
    E_ref, E_trn = S_global_11 @ p, S_global_21 @ p

    # Field longitudinal component
    E_z_ref = -(kx * E_ref[0] + ky * E_ref[1]) / k_z_0_ref
    E_z_trn = -(kx * E_trn[0] + ky * E_trn[1]) / k_z_0_trn

    # Calculate R and T
    R = abs(E_ref[0])**2 + abs(E_ref[1])**2 + abs(E_z_ref)**2
    T = (abs(E_trn[0])**2 + abs(E_trn[1])**2 + abs(E_z_trn)**2) * np.real(
        (k_z_0_trn * inc_medium[1]) / (k_z_0_ref * trn_medium[1]))
    return R, T


def smm_broadband_old(theta,
                      phi,
                      e_matrix,
                      thickness,
                      wav,
                      pol=(1, 0),
                      inc_medium=(1, 1),
                      trn_medium=(1, 1),
                      **kwargs):
    """Calculate reflection and transmission profiles for a range
    of wavelengths

    Args:
    theta, phi (float): angles relating to the incident
                        wave (azimutal and polar angles)
    e_matrix (matrix): matrix with complex dielectric
                        constant for the different layers and wavelengths
    thickness (list): thicknesses of the different layers
    wav (list): list with wavelengths to simulate
    pol (tuple): TM and TE polarizaton (ptm, pte)
    inc_medium (tuple): override incident medium
                        refractive index (default is (e = 1,u =1))
    trn_medium (tuple): override transmision medium
                        refractive index (default is (e = 1,u =1))

    **kwargs:
        u_matrix (matrix): optional matrix with magnetic
                           permeability values, otherwise defaults to 1

    Return:
    R, T: broadband reflection and transmission
    """
    # @dask.delayed
    # def wrapper_func(e_values, u_values, l_value):
    #     R, T = smm(theta,
    #                phi,
    #                e_values,
    #                thickness,
    #                l_value,
    #                pol=pol,
    #                inc_medium=inc_medium,
    #                trn_medium=trn_medium,
    #                u=tuple(u_values))
    #     return R, T

    # Check for overrides in **kwargs input
    if kwargs.get("u_matrix"):  # u values override
        u_matrix = np.array(kwargs.get('u_matrix'))
    else:
        u_matrix = np.ones_like(e_matrix)
    R, T = np.ones_like(wav), np.ones_like(wav)
    for (e, u, (it, lmb)) in zip(e_matrix, u_matrix, enumerate(wav)):
        R[it], T[it] = smm_old(theta,
                               phi,
                               e,
                               thickness,
                               lmb,
                               pol=pol,
                               inc_medium=inc_medium,
                               trn_medium=trn_medium,
                               u=tuple(u))
    return R, T


if __name__ == '__main__':
    n_comp = 15000
    lmb = np.linspace(1, 4, n_comp)
    theta = np.radians(57)
    phi = np.radians(23)
    p = (1, 0)
    n_array = np.random.random((n_comp, 3))*1.5
    k_array = np.random.random((n_comp, 3))
    e_array = (n_array+1j*k_array)**2
    thick = np.array([0.25, 0.5, 0.1])
    inc_medium = (1, 1)
    trn_medium = (1, 1)
    startTime = time.time()
    smm_broadband_old(theta, phi, e_array, thick,
                      lmb, p, inc_medium, trn_medium)
    exec_time = (time.time() - startTime)
    print(f"Run time {exec_time}")
    print("Method 2")
    startTime = time.time()
    layer1 = Layer3D("layer_1", 0.25, lmb, n_array[:, 0], k_array[:, 0])
    layer2 = Layer3D("layer_2", 0.5, lmb, n_array[:, 1], k_array[:, 1])
    layer3 = Layer3D("layer_3", 0.1, lmb, n_array[:, 2], k_array[:, 2])
    R, T = smm_broadband([layer1, layer2, layer3],
                         theta, phi, lmb, p, inc_medium, trn_medium)

    exec_time = (time.time() - startTime)
    print(f"Run time {exec_time}")
