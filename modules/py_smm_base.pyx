# distutils: language = c++
# distutils: sources = smm_base.cpp
from libcpp.complex cimport complex
cimport cython
import numpy as np
# from numpy cimport array

""" Define C++ Classes """

cdef extern from "smm_base.h":
    cpdef enum SMMType:
        BASE,
        TRN,
        REF

    cdef cppclass Matrix:
        complex[double] M11;
        complex[double] M12;
        complex[double] M21;
        complex[double] M22;
        Matrix();
        Matrix(complex[double] M11, complex[double] M12, complex[double] M21,
             complex[double] M22);
        Matrix operator*(const Matrix &matrix) const;
        Matrix inv() const;

    cdef cppclass SMMBase:
          Matrix S11;
          Matrix S12;
          Matrix S21;
          Matrix S22;
          SMMBase();
          SMMBase(Matrix S_11, Matrix S_12, Matrix S_21, Matrix S_22);
          SMMBase operator*(const SMMBase &other);

    cdef cppclass SMatrix(SMMBase):
        SMatrix();
        SMatrix(Matrix V0, double k0, complex[double] kx, complex[double] ky,
                double thickness, complex[double] e, complex[double] u,
                SMMType stype);
        SMatrix operator*(const SMatrix &other)

# Wrapper for the Matrix class
cdef class CMatrix:
    cdef:
        Matrix _cmatrix
    # Initialize the results
    def __cinit__(self, double complex M11=1+0j, double complex M12= 0+0j,
            double complex M21=0+0j, double complex M22=1+0j):
        _M11 = <complex[double]> M11
        _M12 = <complex[double]> M12
        _M21 = <complex[double]> M21
        _M22 = <complex[double]> M22
        self._cmatrix = Matrix(_M11, _M12, _M21, _M22)

    cdef void update_CMatrix(self, Matrix new):
        self._cmatrix = new

    def __mul__(CMatrix x, CMatrix y):
        cdef Matrix res = x._cmatrix * y._cmatrix
        x.update_CMatrix(res)
        return x
    
    def __repr__(self):
        return f"{self._cmatrix.M11.real()} + {self._cmatrix.M11.imag()}j|" +\
               f"{self._cmatrix.M12.real()} + {self._cmatrix.M12.imag()}j\n" +\
               f"{self._cmatrix.M21.real()} + {self._cmatrix.M21.imag()}j|" +\
               f"{self._cmatrix.M22.real()} + {self._cmatrix.M22.imag()}j"

# Wraper for the SMatrix class
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CSMatrix:
    cdef:
        SMatrix _smatrix
        double k0, thickness
        double complex kx, ky
        double complex e, u
    def __init__(self, CMatrix V0=CMatrix(), double k0=0, double complex kx=0,
                 double complex ky=0, double thickness=1, double complex e=1,
                 double complex u=1, SMMType stype=BASE):
        self.k0 = k0
        self.kx = kx
        self.ky = ky
        self.thickness = thickness
        self.e = e
        self.u = u
        _e = <complex[double]> e
        _u = <complex[double]> u
        _kx = <complex[double]> kx
        _ky = <complex[double]> ky
        self._smatrix = SMatrix(V0._cmatrix, k0, _kx, _ky, thickness, _e, _u, stype)

    cdef void _update_smatrix(self, SMatrix new):
        self._smatrix = new

    cdef tuple _fields(self, double complex[:] inc_p_l, double complex kz_ref,
                       double complex kz_trn):
        cdef:
            double E_ref, E_trn
            complex[double] E_ref_0, E_ref_1, E_z_ref
            complex[double] E_trn_0, E_trn_1, E_z_trn

        E_ref_0 = self._smatrix.S11.M11 * <complex[double]>inc_p_l[0] +\
                self._smatrix.S11.M12 * <complex[double]>inc_p_l[1]
        E_ref_1 = self._smatrix.S11.M21 * <complex[double]>inc_p_l[0] +\
                self._smatrix.S11.M22 * <complex[double]>inc_p_l[1]
        E_trn_0 = self._smatrix.S21.M11 * <complex[double]>inc_p_l[0] +\
                self._smatrix.S21.M12 * <complex[double]>inc_p_l[1]
        E_trn_1 = self._smatrix.S21.M21 * <complex[double]>inc_p_l[0] +\
                self._smatrix.S21.M22 * <complex[double]>inc_p_l[1]

        E_z_ref = -1*(<complex[double]>self.kx * E_ref_0 +\
                <complex[double]>self.ky*E_ref_1)/<complex[double]>kz_ref
        E_z_trn = -1*(<complex[double]>self.kx * E_trn_0 +\
                <complex[double]>self.ky*E_trn_1)/<complex[double]>kz_trn
        return E_ref_0, E_ref_1, E_z_ref, E_trn_0, E_trn_1, E_z_trn

    cpdef tuple fields(self, double complex[:] inc_p_l, double complex kz_ref,
                       double complex kz_trn):
        cdef:
            double complex E_ref_0, E_ref_1, E_z_ref, E_trn_0, E_trn_1, E_z_trn

        E_ref_0, E_ref_1, E_z_ref, E_trn_0, E_trn_1, E_z_trn = self._fields(inc_p_l, kz_ref, kz_trn)
        return [E_ref_0, E_ref_1], E_z_ref, [E_trn_0, E_trn_1], E_z_trn

    cpdef tuple ref_trn(self, double complex[:] inc_p_l, double complex kz_ref,
                        double complex kz_trn):
        cdef:
            double E_ref, E_trn
            double complex E_ref_0, E_ref_1, E_z_ref, E_trn_0, E_trn_1, E_z_trn

        E_ref_0, E_ref_1, E_z_ref, E_trn_0, E_trn_1, E_z_trn = self._fields(inc_p_l, kz_ref, kz_trn)
        E_ref = abs(E_ref_0)**2 + abs(E_ref_1)**2 + abs(E_z_ref)**2
        E_trn = abs(E_trn_0)**2 + abs(E_trn_1)**2 + abs(E_z_trn)**2
        return E_ref, E_trn


    cpdef tuple return_SMatrix(self):
        S11 = np.array([[self._smatrix.S11.M11, self._smatrix.S11.M12],
            [self._smatrix.S11.M21, self._smatrix.S11.M22]])
        S12 = np.array([[self._smatrix.S12.M11, self._smatrix.S12.M12],
            [self._smatrix.S12.M21, self._smatrix.S12.M22]])
        S21 = np.array([[self._smatrix.S21.M11, self._smatrix.S21.M12],
            [self._smatrix.S21.M21, self._smatrix.S21.M22]])
        S22 = np.array([[self._smatrix.S22.M11, self._smatrix.S22.M12],
            [self._smatrix.S22.M21, self._smatrix.S22.M22]])
        return S11, S12, S21, S22

    def __mul__(CSMatrix x,CSMatrix y):
        cdef SMatrix res = x._smatrix * y._smatrix
        x._update_smatrix(res)
        return x
