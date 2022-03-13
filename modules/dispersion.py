""" Implementation of severall dispersion formulas """
from typing import Tuple
import numpy as np
import numpy.typing as npt
import logging
import scipy.constants as scc

# Alias to convert nm to ev and ev to nm
_nm_to_ev = (scc.h * scc.c) / (scc.e * 1e-9)

DispType = Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]


def const(e_array: npt.NDArray, n: float, k: float) -> DispType:
    """ Formula for a constant refractive index """
    arr = np.ones_like(e_array, np.float64)
    return arr * n, arr * k


def _tauc_lorentz_peak(e_array: npt.NDArray[np.floating], eg: float, e0: float,
                       a: float, c: float) -> DispType:
    """ Formula to calcuate one peak for the Tauc Lorentz formula """
    ei: npt.NDArray[np.floating] = np.zeros_like(e_array, dtype=np.float64)
    er: npt.NDArray[np.floating] = np.zeros_like(e_array, dtype=np.float64)
    # Interim variables
    e2, eg2, e02, c2 = np.power(e_array, 2), eg**2, e0**2, c**2
    gamma: float = np.sqrt(e02 - c2 / 2)
    alpha: float = np.sqrt(4 * e02 - c2)
    zeta4: npt.NDArray[np.floating] = np.power(e2 - gamma**2,
                                               2) + alpha**2 * c2 / 4
    ain = (e2 - e02) * e2 + eg2 * c2 - e02 * (e02 + 3 * eg2)
    atan = (e2 - e02) * (e02 + eg2) + eg2 * c2
    # Proceed to calculation
    eg_mask: npt.NDArray[np.bool_] = e_array > eg
    # Imaginary part
    ei[eg_mask] = (1 / e_array[eg_mask]) * (a * e0 * c *
                                            (e_array[eg_mask] - eg)**2) / (
                                                (e2[eg_mask] - e02)**2 +
                                                c2 * e2[eg_mask])
    # Real part
    er += (a * c * ain / (2 * np.pi * zeta4 * alpha * e0)) * np.log(
        (e02 + eg2 + alpha * eg) / (e02 + eg2 - alpha * eg))
    er -= (a * atan) / (np.pi * zeta4 * e0) * (np.pi - np.arctan(
        (2 * eg + alpha) / c) + np.arctan((alpha - 2 * eg) / c))
    er += (4 * a * e0 * eg * (e2 - gamma**2) /
           (np.pi * zeta4 * alpha)) * (np.arctan(
               (alpha + 2 * eg) / c) + np.arctan((alpha - 2 * eg) / c))
    er -= (a * e0 * c * (e2 + eg2) / (np.pi * zeta4 * e_array)) * np.log(
        np.abs(e_array - eg) / (e_array + eg))
    er += (2 * a * e0 * c * eg / (np.pi * zeta4)) * np.log(
        np.abs(e_array - eg) * (e_array + eg) /
        (np.sqrt((e02 - eg2)**2 + eg2 * c2)))
    return er, ei


def tauc_lorentz_1(e_array: npt.NDArray[np.floating], einf: float, eg: float,
                   e0: float, a: float, c: float) -> DispType:
    """ Tauc Lorentz Equation with one peak """
    er, ei = _tauc_lorentz_peak(e_array, eg, e0, a, c)
    er += einf
    e_complex: npt.NDArray[np.complexfloating] = er + 1j * ei
    n: npt.NDArray[np.complexfloating] = np.sqrt(e_complex)
    return np.real(n), np.imag(n)


def tauc_lorentz_2(e_array: npt.NDArray[np.floating], einf: float, eg: float,
                   e01: float, a1: float, c1: float, e02: float, a2: float,
                   c2: float) -> DispType:
    """ Tauc Lorentz Equation with two peaks """
    er1, ei1 = _tauc_lorentz_peak(e_array, eg, e01, a1, c1)
    er2, ei2 = _tauc_lorentz_peak(e_array, eg, e02, a2, c2)
    er = er1 + er2 + einf
    ei = ei1 + ei2
    e_complex = er + 1j * ei
    n = np.sqrt(e_complex)
    return np.real(n), np.imag(n)


def tauc_lorentz_n(e_array: npt.NDArray[np.floating], n_peak: int, einf: float,
                   eg: float, *args: Tuple[float]) -> DispType:
    """ Tauc Lorentx Equation for multiple peaks
    Args:
        e (array): energy array
        n_peak (int): Number of peaks
        einf (float): einf variable
        eg (float): eg variable
        *args (e_array0, a, c)*n_peak: remaining peak variables
    """
    n_peak = int(n_peak)
    logging.debug(f"{n_peak=}::{einf=}::{eg=}::{args=}")
    if len(args) != 3 * n_peak:
        logging.error(args, n_peak)
        raise Exception("Number of arguments not compatible with n_peak")
    er = np.zeros_like(e_array, dtype=np.float64)
    ei = np.zeros_like(e_array, dtype=np.float64)
    for i in range(n_peak):
        e0i, ai, ci = args[i * 3 + 0], args[i * 3 + 1], args[i * 3 + 2]
        er_i, ei_i = _tauc_lorentz_peak(e_array, eg, e0i, ai, ci)
        er += er_i
        ei += ei_i
    er += einf
    e_complex = er + 1j * ei
    n = np.sqrt(e_complex)
    return np.real(n), np.imag(n)


def cauchy(e_array: npt.NDArray[np.floating], a: float, b: float, c: float):
    """ Standard non-absorber cauchy formula
    Args:
        e (array): energy array
        a (float): dimensionless parameter when lmb→ inf
        b (float): curvature and amplitude of the refractive index in the visible
        c (float): curvature for small wavelengths
    """
    logging.debug(f"{a=}::{b=}::{c=}")
    lmb = _nm_to_ev * (1 / e_array)
    n = a + 1e4 * b / lmb**2 + 1e9 * c / lmb**4
    k = np.zeros_like(n)
    return n, k


def cauchy_abs(e_array: npt.NDArray[np.floating], a: float, b: float, c: float,
               d: float, e: float, f: float):
    """ Standard non-absorber cauchy formula
    Args:
        e (array): energy array
        a (float): dimensionless parameter when lmb→ inf
        b (float): curvature and amplitude of the refractive index in the visible
        c (float): curvature for small wavelengths
        d/e/f (float): similar to a/b/c but for extinction coefficient
    """
    logging.debug(f"{a=}::{b=}::{c=}::{d=}::{e=}::{f=}")
    lmb = _nm_to_ev * (1 / e_array)
    n = a + 1e4 * b / lmb**2 + 1e9 * c / lmb**4
    k = 1e-5 * d + 1e4 * e / lmb**2 + 1e9 * f / lmb**4
    return n, k


def sellmeier_abs(e_array: npt.NDArray[np.floating], a: float, b: float,
                  c: float, d: float, e: float) -> DispType:
    """ Sellmeier Absorber formula
    Args:
        a (float): related with refractive index amplitude when lmb→ inf
        b (float): curvature of the refractive index
        c (float): strenght of the absorption curve
        d/e (float): related with the intensity of the absorption coefficient
    """
    logging.debug(f"{a=}::{b=}::{c=}::{d=}::{e=}")
    lmb = _nm_to_ev * (1 / e_array)
    n = np.sqrt((1 + a) / (1 + (1e4 * b / lmb**2)))
    k = c / (1e-2 * n * d * lmb + 1e2 * e / lmb + 1 / lmb**3)
    return n, k


def _new_amorphous_peak(e_array: npt.NDArray[np.floating], fj: float,
                        gammaj: float, wj: float, wg: float) -> DispType:
    k = np.zeros_like(e_array, dtype=np.float64)
    n = np.zeros_like(e_array, dtype=np.float64)
    w_mask = e_array > wg
    k[w_mask] = fj * (e_array[w_mask] - wg)**2 / (
        (e_array[w_mask] - wj)**2 + gammaj**2)
    b: float = (fj / gammaj) * (gammaj**2 - (wj - wg)**2)
    c: float = 2 * fj * gammaj * (wj - wg)
    n = (b * (e_array - wj) + c) / ((e_array - wj)**2 + gammaj**2)
    return n, k


def new_amorphous_n(e_array: npt.NDArray[np.floating], n_peak: int,
                    ninf: float, wg: float, *args: Tuple[float]) -> DispType:
    logging.debug(f"{n_peak=}::{ninf=}::{wg=}::{args=}")
    n_peak = int(n_peak)
    if len(args) != 3 * n_peak:
        raise Exception("Invalid number of variable arguments")
    n = np.zeros_like(e_array, dtype=np.float64)
    k = np.zeros_like(e_array, dtype=np.float64)
    for i in range(n_peak):
        fj, gammaj, wj = args[i * 3 + 0], args[i * 3 + 1], args[i * 3 + 2]
        n_i, k_i = _new_amorphous_peak(e_array, fj, gammaj, wj, wg)
        n += n_i
        k += k_i
    n += ninf
    return n, k
