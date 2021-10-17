"""
File with the optimized version of the Scattering Matrix Script

Has 2 main functions:
    smm - Calculates R and T for a specific wavelength
    smm_broadband - Calculates R and T for a wavelength range
"""
# TODO: Rebuild this portion of the code  <11-10-21, Miguel> #

# Basic modules
from enum import Enum, auto
import numpy as np
from numpy.linalg import inv
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import scipy.constants as scc

c_nm = scc.c * 1e-9


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
        """ Implement Syntax Sugar to perform the Redheffer Product """
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
        self.Vi = V_i
        self.Omega_i = Omega_i
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

    def __init__(self, name, thickness, lmb, n_val, k_val):
        self.name = name
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

    def __init__(self, name, thickness, lmb, n_array, k_array, **kwargs):
        self.name = name
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


def _e_fields(S_Global, inc_p_l, inc_p_r, kx, ky, kz_ref, kz_trn):
    """ Determine the Electric Fields resulting from the SMM calculation """
    E_ref = S_Global.S_11 @ inc_p_l + S_Global.S_12 @ inc_p_r
    E_trn = S_Global.S_21 @ inc_p_l + S_Global.S_22 @ inc_p_r
    Ez_ref = -(kx * E_ref[0] + ky * E_ref[1]) / kz_ref
    Ez_trn = -(kx * E_trn[0] + ky * E_trn[1]) / kz_trn
    return E_ref, E_trn, Ez_ref, Ez_trn


def _int_field(A, c_int, Omega_Layer, k0, z):
    """ Calculate the internal fields from the internal mode coefficients """
    mode_layer = np.zeros((4, 4), dtype=np.complex128)
    mode_layer[:2, :2] = np.eye(2)*np.exp(Omega_Layer * k0 * z)
    mode_layer[2:, 2:] = np.eye(2)*np.exp(-Omega_Layer * k0 * z)
    return A@mode_layer@c_int


def _field_power(fields, kx, ky, kz):
    """ Calculate |E|^2 """
    Ex = fields[0]
    Ey = fields[1]
    Ez = -(kx*Ex + ky*Ey)/kz
    return abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2


def smm_broadband(layer_list, theta, phi, lmb, pol, i_med, t_med,
                  override_thick=None):
    """
    SMM for broadband simulation
    Args:
        layer_list: List of Layer objects with the info for all layers
        theta/phi: Incidence angles
        lmb: Array with wavelengths for simulation
        pol: Tuple with the TM/TE polarization components
        i_med/t_med: Data for the reflection and transmission media
        **override_thick: Override the built thickness of the layers
    Returns:
        R, T: Arrays with the Reflection and transmission for the layer setup
    """
    if override_thick is not None:
        if len(override_thick) == len(layer_list):
            for thick_i, layer in zip(override_thick, layer_list):
                layer.thickness = thick_i
        else:
            raise Exception("Override Thickness does not match Layer List")
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
        S_Global_i = S_ref
        for index, layer in enumerate(layer_list):
            S_Layer = SMM(V0, k0_i, kx, ky, layer.thickness, layer_data[index])
            S_Global_i *= S_Layer
        S_Global_i = S_Global_i * S_trn
        E_ref, E_trn, Ez_ref, Ez_trn = _e_fields(
            S_Global_i, p, [0, 0], kx, ky, kz_ref, kz_trn)
        R.append(abs(E_ref[0])**2 + abs(E_ref[1])**2 + abs(Ez_ref)**2)
        T.append(
                (
                    abs(E_trn[0])**2 + abs(E_trn[1])**2 + abs(Ez_trn)**2
                ) * np.real((kz_trn * i_med[1]) / (kz_ref * t_med[1])))
    return np.array(R), np.array(T)


def smm_abs_layer_i(layer_list, layer_i, theta, phi, lmb, pol, i_med, t_med):
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
        R, T: Arrays with the Reflection and transmission for the layer setup
    """
    if layer_i == 0 or layer_i > len(layer_list):
        raise Exception("Invalid Index")
    layer_i -= 1
    k0, kx, ky, V0, p, S_Global = _initialize_smm(
        theta, phi, lmb, pol, i_med, t_med)
    kz_ref = np.sqrt(i_med[0]*i_med[1] - kx**2 - ky**2)
    kz_trn = np.sqrt(t_med[0]*t_med[1] - kx**2 - ky**2)
    # kz_vac = np.sqrt(1 - kx**2 - ky**2)
    # This is a simplification to determine all the values in beforehand
    layer_data = np.array([layer_i.e_value(lmb) for layer_i in layer_list])
    thick_list = np.array([layer_i.thickness for layer_i in layer_list])
    thick_list = np.cumsum(thick_list)
    abs = []
    for lmb_i, k0_i, layer_data in zip(lmb, k0, layer_data.T):
        w = 2*np.pi*c_nm/lmb_i
        S_trn = SMM(V0, k0, kx, ky, 0,
                    t_med[0], t_med[1], SMMType.TRN)
        S_ref = SMM(V0, k0, kx, ky, 0,
                    i_med[0], i_med[1], SMMType.REF)
        S_Global_lmb = S_ref
        for layer_index, layer in enumerate(layer_list):
            S_Layer = SMM(V0, k0_i, kx, ky, layer.thickness,
                          layer_data[layer_index])
            if layer_index == layer_i:
                S_Global_Before = S_Global_lmb
                V_Layer = S_Layer.Vi
                Omega_Layer = S_Layer.Omega_i
                k_layer = np.sqrt(layer.e_value(lmb_i) - kx**2 - ky**2)
                layer_k = layer.k(lmb_i)
            S_Global_lmb *= S_Layer
        S_Global_lmb *= S_trn
        E_ref, _, _, _ = _e_fields(
            S_Global_lmb, p, [0, 0], kx, ky, kz_ref, kz_trn)
        # Determine the external mode coefficients just before the wanted layer
        c_left_m = inv(S_Global_Before.S_12)@(E_ref - S_Global_Before.S_11@p)
        c_left_p = S_Global_Before.S_21@p + S_Global_Before.S_22@c_left_m
        # Calculate the internal fields
        a_matrix = np.eye(2) + inv(V_Layer)@V0
        b_matrix = np.eye(2) - inv(V_Layer)@V0
        c_int_p = 1/2 * (a_matrix@c_left_p + b_matrix@c_left_m)
        c_int_m = 1/2 * (b_matrix@c_left_p + a_matrix@c_left_m)
        if layer_i == 0:
            z = np.linspace(0, thick_list[layer_i], 100)
        else:
            z = np.linspace(thick_list[layer_i-1], thick_list[layer_i], 100)
        # field = []
        # ex = []
        # for z_i in z:
        #     fields = _int_field(A, c_int, Omega_Layer, k0_i, z_i)
        #     ex.append(fields[0])
        #     power = _field_power(fields, kx, ky, k_layer)
        #     field.append(0.5*power*layer_k*w)
        # abs.append(trapz(np.array(field), z))
        # plt.plot(z, ex)
        # plt.show()
    return np.array(abs)


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
        S_Global, p, [0, 0], kx, ky, kz_ref, kz_trn)
    R = abs(E_ref[0])**2 + abs(E_ref[1])**2 + abs(Ez_ref)**2
    T = (abs(E_trn[0])**2 + abs(E_trn[1])**2 + abs(Ez_trn)**2) * np.real(
        (kz_trn * i_med[1]) / (kz_ref * t_med[1]))
    return R, T


if __name__ == '__main__':
    n_comp = 5
    lmb = np.linspace(0.5, 1.6, n_comp)
    theta = np.radians(57)
    phi = np.radians(23)
    p = (1, 0)
    n_array = np.random.random((n_comp, 3))*1.5
    k_array = np.zeros_like(n_array)
    # k_array = np.array([[0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0],
    #                     [0.5, 0.5, 0.5]])
    e_array = (n_array+1j*k_array)**2
    thick = np.array([3, 0.5, 0.1])
    inc_medium = (1, 1)
    trn_medium = (1, 1)
    layer1 = Layer3D(3, lmb, n_array[:, 0], k_array[:, 0])
    layer2 = Layer3D(0.5, lmb, n_array[:, 1], k_array[:, 1])
    layer3 = Layer3D(0.1, lmb, n_array[:, 2], k_array[:, 2])
    abs1 = smm_abs_layer_i([layer1, layer2, layer3], 1,
                           theta, phi, lmb, p, inc_medium, trn_medium)
    print(abs1)
    abs2 = smm_abs_layer_i([layer1, layer2, layer3], 2,
                           theta, phi, lmb, p, inc_medium, trn_medium)
    print(abs2)
    abs3 = smm_abs_layer_i([layer1, layer2, layer3], 3,
                           theta, phi, lmb, p, inc_medium, trn_medium)
    print(abs3)
    print("-------------------------------")
    print(abs1+abs2+abs3)
    R, T = smm_broadband([layer1, layer2, layer3],
                         theta, phi, lmb, p, inc_medium, trn_medium)
    print(1-R-T)
