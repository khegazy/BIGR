#include "spherical_j_cpp.h"

using namespace std;


void spherical_j(
    double* x, int* x_shape, int x_dim,
    int* l, int l_len, double* out) {
  /* 
   * Used recursive relations from: https://dlmf.nist.gov/10.51
   * Require l to contain all even values <= max(l) | Every jn contribution must be saved
   * in out during the calculation.
  */ 

  unsigned long long int x_len = 1;
  for (int i=0; i<x_dim; i++) x_len *= x_shape[i];

  // Find maximum L
  int l_max = -1;
  for (int i=0; i<l_len; i++) l_max = (l[i] > l_max) ? l[i] : l_max;
  int N = l_max/2 + 1;

  // Pointers to different n
  std::vector<int>* inds = new std::vector<int>[N];
  for (int n=0; n<N; n++) {
    for (int i=0; i<l_len; i++) {
      if (n*2 == l[i]) inds[n].push_back(i);
    }
  }
  double** out_p = new double*[N];
  unsigned long long int shift = inds[0][0]*x_len;
  out_p[0] = out + shift;
  for (int i=1; i<N; i++) {
    shift = x_len*(inds[i][0]-inds[i-1][0]);
    out_p[i] = out_p[i-1] + shift;
  }

  // Pre calculations
  double jnm2, jnm2_, jn, jn_, x_sq;
  double n, fm, sgn;
  double tpi = 2*3.14159265359;
  double cut1 = tpi/4;
  double cut2 = 3*tpi/4;
  jnm2 = 0; jn = 0;
  double tnp1p1, tnm1p1;
  double* c1;
  double* c2;
  double* cnm2;
  if (N > 2) {
    c1 = new double[N-2];
    c2 = new double[N-2];
    cnm2 = new double[N-2];
    for (int i=2; i<N; i++) {
      n = (i-1)*2;
      tnp1p1 = 2*n + 3;
      tnm1p1 = 2*n - 1;
      c1[i-2] = (tnp1p1 - 2)*tnp1p1;
      cnm2[i-2] = tnp1p1/tnm1p1;
      c2[i-2] = cnm2[i-2] + 1;
    }
  }
  else {
    c1 = NULL;
    c2 = NULL;
    cnm2 = NULL;
  }

  // Calculate spherical bessel functions
  for (unsigned long long int ind=0; ind<x_len; ind++) {
    x_sq = x[ind]*x[ind];
    /////  Calculating j0 and j2  ////
    if (x[ind] == 0) {
      jnm2 = 1;
      jn = 0;
    }
    else {
      jnm2_ = sin(x[ind]);
      jnm2 = jnm2_/x[ind];
      fm = fmod(x[ind], tpi);
      sgn = ((fm > cut1) && (fm < cut2)) ? -1 : 1;
      jn  = (3/(x_sq) - 1)*jnm2 - sgn*3*sqrt(1-jnm2_*jnm2_)/x_sq;
    }

    // Filling j0
    out_p[0][ind] = jnm2;
    // Filling j2
    out_p[1][ind] = jn;

    for (int i=2; i<N; i++) {
      jn_ = (c1[i-2]/x_sq - c2[i-2])*jn - cnm2[i-2]*jnm2;
      out_p[i][ind] = jn_;
      jnm2 = jn;
      jn = jn_;
    }
  }

  // Filling x=0
  if ((x[0] == 0) || isnan(out[0])) {
    int x0_shift = x_len/x_shape[0];
    double x0 = 0;
    for (int i=0; i<N; i++) {
      x0 = (l[i] == 0) ? 1 : 0;
      for (int id=0; id<x_shape[0]; id++) {
        for (int ig=0; ig<x_shape[3]; ig++) {
          out_p[i][id*x0_shift+ig] = x0;
        }
      }
    }
  }

  // Refilling
  for (int i=0; i<N; i++) {
    if (inds[i].size() > 1) {
      for (uint j=1; j<inds[i].size(); j++) {
        shift = (inds[i][j]-inds[i][0])*x_len;
        for (unsigned long long int k=0; k<x_len; k++) {
          out_p[i][shift+k] = out_p[i][k];
        }
      }
    }
  }
  /*
  for (unsigned long long int ind=0; ind<x_len; ind++) {
    for (int i=0; i<N; i++) {
      if (isnan(out[inds[i][0]*x_len+ind]) || ind==0) {
        if (i > 2) {
          jnm2 = out[inds[i-2][0]*x_len+ind];
        }
        else {
          jnm2 = -1;
        }
        if (i > 1) {
          jn = out[inds[i-1][0]*x_len+ind];
        }
        else {
          jn = -1;
        }

        cout<<"NAN: "<<out[inds[i][0]*x_len+ind]<<" "<<ind<<"/"<<inds[i][0]*x_len<<"/"<<i;
        cout<<"  "<<jnm2<<" "<<jn<<" "<<c1[i-2]<<" "<<c1[i-2]/x_sq<<" "<<c2[i-2]<<" "<<cnm2[i-2]<<endl;
      }
    }
    }
  */
  // Free dynamic memory
  delete[] out_p;
  delete[] inds;
  if (N > 2) {
    delete[] c1;
    delete[] c2;
    delete[] cnm2;
  }

  return;
}


void calc_coeffs_helper(
    double* x, int* x_shape, int x_dim,
    int* l, int l_len,
    double* C_real, double* C_imag,
    double* Y_real, double* Y_imag,
    double* scat_amps, double* weights, double* out) {

  // Find maximum L
  int l_max = -1;
  for (int i=0; i<l_len; i++) l_max = (l[i] > l_max) ? l[i] : l_max;
  int N = l_max/2 + 1;

  // Calculate Spherical Bessel Functions
  unsigned long long int x_len = 1;
  int ox_len = x_shape[1]*x_shape[2];
  int y_len = l_len;
  for (int i=0; i<x_dim; i++) {
    x_len *= x_shape[i];
    if (i != 1) {
      y_len *= x_shape[i];
    }
  }
  int* jn_lmk = new int[N];
  for (int i=0; i<N; i++) jn_lmk[i] = i*2;

  double* jn = new double[N*x_len]; //684628000
  spherical_j(x, x_shape, x_dim, jn_lmk, N, jn);
  double** jn_p = new double*[l_len];
  unsigned long long int shift = x_len*l[0]/2;
  jn_p[0] = jn + shift;
  for (int i=1; i<l_len; i++) {
    shift = x_len*(l[i]-l[i-1])/2;
    jn_p[i] = jn_p[i-1] + shift;
  }

  long int indO, indY, indY_, indj, indj_;
  for (int il=0; il<l_len; il++) {
    indj_ = l[il]*x_len/2;
    indj = 0;
    for (int id=0; id<x_shape[0]; id++) {
      indO = 0;
      indY_ = il*x_shape[0]*x_shape[2]*x_shape[3] + id*x_shape[2]*x_shape[3];
      for (int iq=0; iq<x_shape[1]; iq++) {
        indY = indY_;
        for (int iw=0; iw<x_shape[2]; iw++) {
          for (int ig=0; ig<x_shape[3]; ig++) {
            out[il*ox_len+indO] += 
              scat_amps[id*x_shape[1]+iq]*weights[iw*x_shape[3]+ig]
              *(C_real[il]*Y_real[indY] - C_imag[il]*Y_imag[indY])
              *jn_p[il][indj];//[indj_+indj];
            indY++;
            indj++;
          }
          indO++;
        }
      }
    }
  }

  delete[] jn_p;
  delete[] jn_lmk;
  delete[] jn;

  return;
}
