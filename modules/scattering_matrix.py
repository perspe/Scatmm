"""
File with the optimized version of the Scattering Matrix Script

Has 2 main functions:
    smm - Calculates R and T for a specific wavelength
    smm_broadband - Calculates R and T for a wavelength range
"""

# Basic modules
import time
import numpy as np

from numpy.linalg import inv


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


def smm(theta,
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
    k_z_0_ref = np.sqrt(1 - kx**2 - ky**2)
    k_z_0_trn = np.sqrt(1 - kx**2 - ky**2)

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


def smm_broadband(theta,
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
        R[it], T[it] = smm(theta,
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
    startTime = time.time()
    lam0 = np.linspace(1, 3, 150)
    theta = np.radians(57)
    phi = np.radians(23)
    p = (1j / np.sqrt(2), 1 / np.sqrt(2))
    e = np.array([[1.3, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [2.1, 2.2, 1.9, 1.6], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [2.1, 2.2, 1.9, 1.6], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [2.1, 2.2, 1.9, 1.6], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [2.1, 2.2, 1.9, 1.6],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [2.1, 2.2, 1.9, 1.6], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [2.1, 2.2, 1.9, 1.6],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.2, 2 + 0.1j, 4, 3],
                  [2.1, 2.2, 1.9, 1.6], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.2, 2 + 0.1j, 4, 3],
                  [1.3, 2 + 0.1j, 3, 4], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [1.3, 2 + 0.1j, 3, 4],
                  [1.5, 1.5 + 0.2j, 2.5, 1.2], [1.5, 1.5 + 0.2j, 2.5, 1.2],
                  [1.2, 2 + 0.1j, 4, 3], [2.1, 2.2, 1.9, 1.6]])
    thick = np.array([0.25, 0.5, 0.1, 0.2]) * lam0[0]
    smm_broadband(theta,
                  phi,
                  e,
                  thick,
                  lam0,
                  pol=p,
                  inc_medium=(1, 1.2),
                  trn_medium=(1, 1.6))
    exec_time = (time.time() - startTime)
    print(f"Run time {exec_time}")
