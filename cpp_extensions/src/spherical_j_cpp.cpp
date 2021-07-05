#include "spherical_j_cpp.h"

using namespace std;


void spherical_j(
    double* x, int x_len,
    int* l, int l_len, double* out) {
  /* 
   * Require l to contain all even values <= max(l) | Every jn contribution must be saved
   * in out during the calculation.
  */ 

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

  // Pre calculations
  double n, jnm2, jnm2_, jn, x_sq, fm, sgn;
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
  for (int ind=0; ind<x_len; ind++) {

    x_sq = x[ind]*x[ind];
    /////  Calculating j0 and j2  ////
    jnm2_ = sin(x[ind]);
    jnm2 = jnm2_/x[ind];
    fm = fmod(x[ind], tpi);
    sgn = ((fm > cut1) && (fm < cut2)) ? -1 : 1;
    jn  = (3/(x_sq) - 1)*jnm2 - sgn*3*sqrt(1-jnm2_*jnm2_)/x_sq;

    // Filling j0
    out[ind] = jnm2;
    // Filling j2
    out[x_len+ind] = jn;

    for (int i=2; i<N; i++) {
      out[inds[i][0]*x_len+ind] = (c1[i-2]/x_sq - c2[i-2])*jn - cnm2[i-2]*jnm2;
      jnm2 = jn;
      jn = out[inds[i][0]*x_len+ind];
    }
  }

  // Refilling
  for (int i=0; i<N; i++) {
    if (inds[i].size() > 1) {
      for (uint j=1; j<inds[i].size(); j++) {
        for (int k=0; k<x_len; k++) {
          out[inds[i][j]*x_len+k] = out[inds[i][0]*x_len+k];
        }
      }
    }
  }
  
  // Free dynamic memory
  delete[] inds;
  if (N > 2) {
    delete[] c1;
    delete[] c2;
    delete[] cnm2;
  }

  return;
}
