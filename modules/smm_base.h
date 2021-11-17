#ifndef __SMM_BASE_h__
#define __SMM_BASE_h__

#include "dbg.h"
#include <array>
#include <complex>

using namespace std;

// Main Structures
class Matrix {
public:
  complex<double> M11 = 1;
  complex<double> M12 = 0;
  complex<double> M21 = 0;
  complex<double> M22 = 1;
  Matrix(){};
  Matrix(complex<double> M11, complex<double> M12, complex<double> M21,
         complex<double> M22)
      : M11(M11), M12(M12), M21(M21), M22(M22){};
  Matrix operator*(const Matrix &matrix) const;
  Matrix operator*(const complex<double> &alpha) const;
  Matrix operator+(const Matrix &matrix) const;
  Matrix operator-(const Matrix &matrix) const;
  Matrix operator-() const;
  Matrix inv() const;
};

// Helper functions
void debug_array(const Matrix &matrix, const char *name);
// Matrix functions
Matrix mexp(const Matrix &M1);

typedef enum SMMType { BASE, REF, TRN } SMMType;

class SMMBase {
public:
  Matrix S11{0, 0, 0, 0};
  Matrix S12{1, 0, 0, 1};
  Matrix S21{1, 0, 0, 1};
  Matrix S22{0, 0, 0, 0};
  SMMBase(){};
  SMMBase(Matrix S_11, Matrix S_12, Matrix S_21, Matrix S_22)
      : S11(S_11), S12(S_12), S21(S_21), S22(S_22){};
  SMMBase operator*(const SMMBase &other);
  SMMBase operator*=(const SMMBase &other);
};

class SMatrix : public SMMBase {
public:
  SMatrix() : SMMBase(){};
  SMatrix(Matrix V0, double k0, complex<double> kx, complex<double> ky,
          double thickness, complex<double> e, complex<double> u,
          SMMType stype);
  SMatrix(Matrix V0, double k0, complex<double> kx, complex<double> ky,
          double thickness, complex<double> e, SMMType stype);
  SMatrix(Matrix V0, double k0, complex<double> kx, complex<double> ky,
          double thickness, complex<double> e);
  SMatrix(SMMBase SMM) : SMMBase(SMM){};
  SMatrix operator*(const SMatrix &other);

private:
  Matrix V0;
  double k0;
  complex<double> kx, ky;
  double thickness;
  complex<double> e, u;
  SMMType stype;
  Matrix Vi, Omega_i;
  void mat_elements();
  void smm_norm();
  void smm_trn();
  void smm_ref();
};

#endif /* ifndef __SMM_BASE_h__ */
