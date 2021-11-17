#include "smm_base.h"
#include "dbg.h"
#include <array>
#include <complex.h>
#include <complex>

using namespace std;
using namespace std::complex_literals; // to use i for the complex number

void debug_array(const Matrix &matrix, const char *name) {
  debug("Matrix %s: %f+i%f|%f+i%f|%f+i%f|%f+i%f", name, matrix.M11.real(),
        matrix.M11.imag(), matrix.M12.real(), matrix.M12.imag(),
        matrix.M21.real(), matrix.M21.imag(), matrix.M22.real(),
        matrix.M22.imag());
}

// Defining the matrix internal functions
Matrix Matrix::operator*(const Matrix &other) const {
  Matrix M_res;
  M_res.M11 = this->M11 * other.M11 + this->M12 * other.M21;
  M_res.M12 = this->M11 * other.M12 + this->M12 * other.M22;
  M_res.M21 = this->M21 * other.M11 + this->M22 * other.M21;
  M_res.M22 = this->M21 * other.M12 + this->M22 * other.M22;
  return M_res;
}

Matrix Matrix::operator*(const complex<double> &alpha) const {
  Matrix M_res;
  M_res.M11 = alpha * this->M11;
  M_res.M12 = alpha * this->M12;
  M_res.M21 = alpha * this->M21;
  M_res.M22 = alpha * this->M22;
  return M_res;
}

Matrix Matrix::operator+(const Matrix &other) const {
  Matrix M_res;
  M_res.M11 = this->M11 + other.M11;
  M_res.M12 = this->M12 + other.M12;
  M_res.M21 = this->M21 + other.M21;
  M_res.M22 = this->M22 + other.M22;
  return M_res;
}

Matrix Matrix::operator-(const Matrix &other) const {
  Matrix M_res;
  M_res.M11 = this->M11 - other.M11;
  M_res.M12 = this->M12 - other.M12;
  M_res.M21 = this->M21 - other.M21;
  M_res.M22 = this->M22 - other.M22;
  return M_res;
}

Matrix Matrix::operator-() const {
  Matrix M_res;
  M_res.M11 = -this->M11;
  M_res.M12 = -this->M12;
  M_res.M21 = -this->M21;
  M_res.M22 = -this->M22;
  return M_res;
}

Matrix Matrix::inv() const {
  Matrix M_res;
  complex<double> det = this->M11 * this->M22 - this->M12 * this->M21;
  M_res.M11 = this->M22 / det;
  M_res.M12 = -this->M12 / det;
  M_res.M21 = -this->M21 / det;
  M_res.M22 = this->M11 / det;
  return M_res;
}

Matrix mexp(const Matrix &M1) {
  Matrix result;
  result.M11 = exp(M1.M11);
  result.M12 = 0;
  result.M21 = 0;
  result.M22 = exp(M1.M22);
  return result;
}

SMMBase SMMBase::operator*(const SMMBase &other) {
  SMMBase return_res;
  Matrix eye;
  Matrix D = this->S12 * (eye - other.S11 * this->S22).inv();
  debug_array(D, "D");
  Matrix F = other.S21 * (eye - this->S22 * other.S11).inv();
  debug_array(F, "F");
  return_res.S11 = this->S11 + D * other.S11 * this->S21;
  return_res.S12 = D * other.S12;
  return_res.S21 = F * this->S21;
  return_res.S22 = other.S22 + F * this->S22 * other.S12;
  debug_array(return_res.S11, "SMatrix __Mul__: S11");
  debug_array(return_res.S12, "SMatrix __Mul__: S12");
  debug_array(return_res.S21, "SMatrix __Mul__: S21");
  debug_array(return_res.S22, "SMatrix __Mul__: S22");
  return return_res;
}

SMMBase SMMBase::operator*=(const SMMBase &other) {
  SMMBase return_res = *this * other;
  return return_res;
}

void SMatrix::mat_elements() {
  complex<double> mu_eps = e * u;
  Matrix eye;
  Matrix Qi{kx * ky / u, e - kx * kx / u, ky * ky / u - e, -kx * ky / u};
  this->Omega_i = eye * 1i * sqrt(mu_eps - kx * kx - ky * ky);
  Vi = Qi * this->Omega_i.inv();
  debug_array(Vi, "Vi");
}

void SMatrix::smm_norm() {
  Matrix eye, Ai, Bi, iVi_V0, iA_i, Xi, X_BiA_X, inv_fact;
  iVi_V0 = this->Vi.inv() * this->V0;
  debug_array(iVi_V0, "iVi_V0");
  Ai = eye + iVi_V0;
  Bi = eye - iVi_V0;
  Xi = mexp(this->Omega_i * this->k0 * this->thickness);
  debug_array(Xi, "Xi");
  iA_i = Ai.inv();
  X_BiA_X = Xi * Bi * iA_i * Xi;
  inv_fact = (Ai - X_BiA_X * Bi).inv();
  Matrix S_11 = inv_fact * (X_BiA_X * Ai - Bi);
  Matrix S_12 = inv_fact * Xi * (Ai - Bi * iA_i * Bi);
  Matrix S_21 = S_12;
  Matrix S_22 = S_11;
  debug_array(S_11, "SMatrix Init: S11");
  debug_array(S_12, "SMatrix Init: S12");
  debug_array(S_21, "SMatrix Init: S21");
  debug_array(S_22, "SMatrix Init: S22");
  this->S11 = S_11;
  this->S12 = S_12;
  this->S21 = S_21;
  this->S22 = S_22;
};

void SMatrix::smm_trn() {
  Matrix eye, iV0_Vtrn, A_trn, B_trn, iA_trn;
  iV0_Vtrn = this->V0.inv() * this->Vi;
  A_trn = eye + iV0_Vtrn;
  B_trn = eye - iV0_Vtrn;
  iA_trn = A_trn.inv();
  Matrix S_11 = B_trn * iA_trn;
  Matrix S_12 = (A_trn - B_trn * iA_trn * B_trn) * 0.5;
  Matrix S_21 = iA_trn * 2;
  Matrix S_22 = -iA_trn * B_trn;
  debug_array(S_11, "SMatrix Init: S11");
  debug_array(S_12, "SMatrix Init: S12");
  debug_array(S_21, "SMatrix Init: S21");
  debug_array(S_22, "SMatrix Init: S22");
  this->S11 = S_11;
  this->S12 = S_12;
  this->S21 = S_21;
  this->S22 = S_22;
}

void SMatrix::smm_ref() {
  Matrix eye, iV0_Vref, A_ref, B_ref, iA_ref;
  iV0_Vref = this->V0.inv() * this->Vi;
  A_ref = eye + iV0_Vref;
  B_ref = eye - iV0_Vref;
  iA_ref = A_ref.inv();
  Matrix S_11 = -iA_ref * B_ref;
  Matrix S_12 = iA_ref * 2;
  Matrix S_21 = (A_ref - B_ref * iA_ref * B_ref) * 0.5;
  Matrix S_22 = B_ref * iA_ref;
  debug_array(S_11, "SMatrix Init: S11");
  debug_array(S_12, "SMatrix Init: S12");
  debug_array(S_21, "SMatrix Init: S21");
  debug_array(S_22, "SMatrix Init: S22");
  this->S11 = S_11;
  this->S12 = S_12;
  this->S21 = S_21;
  this->S22 = S_22;
}

SMatrix::SMatrix(Matrix V0, double k0, complex<double> kx, complex<double> ky,
                 double thickness, complex<double> e, complex<double> u,
                 SMMType stype)
    : V0(V0), k0(k0), kx(kx), ky(ky), thickness(thickness), e(e), u(u),
      stype(stype) {
  this->mat_elements();
  if (stype == BASE) {
    debug("Base SMatrix");
    smm_norm();
  } else if (stype == REF) {
    debug("Ref SMatrix");
    smm_ref();
  } else if (stype == TRN) {
    debug("Trn SMatrix");
    smm_trn();
  }
}

SMatrix::SMatrix(Matrix V0, double k0, complex<double> kx, complex<double> ky,
                 double thickness, complex<double> e, SMMType stype)
    : V0(V0), k0(k0), kx(kx), ky(ky), thickness(thickness), e(e), u(1),
      stype(stype) {
  this->mat_elements();
  if (stype == BASE) {
    debug("Base SMatrix");
    smm_norm();
  } else if (stype == REF) {
    debug("Ref SMatrix");
    smm_ref();
  } else if (stype == TRN) {
    debug("Trn SMatrix");
    smm_trn();
  }
}

SMatrix::SMatrix(Matrix V0, double k0, complex<double> kx, complex<double> ky,
                 double thickness, complex<double> e)
    : V0(V0), k0(k0), kx(kx), ky(ky), thickness(thickness), e(e), u(1),
      stype(BASE) {
  this->mat_elements();
  if (stype == BASE) {
    debug("Base SMatrix");
    smm_norm();
  } else if (stype == REF) {
    debug("Ref SMatrix");
    smm_ref();
  } else if (stype == TRN) {
    debug("Trn SMatrix");
    smm_trn();
  }
}

SMatrix SMatrix::operator*(const SMatrix &other) {
  SMatrix res = SMMBase(this->S11, this->S12, this->S21, this->S22) *
                SMMBase(other.S11, other.S12, other.S21, other.S22);
  return res;
}
