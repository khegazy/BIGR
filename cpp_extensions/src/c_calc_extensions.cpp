#include "c_calc_extensions.h"

#include <iostream>
#include <iomanip>

#include <math.h>
#include <vector>


void calculate_c(
    double* x,
    int* x_shape,
    int* l,
    int l_len,
    double* c_prefactor_real,
    double* c_prefactor_imag,
    double* Ylk_real, 
    double* Ylk_imag,
    double* scat_amps, 
    double* mol_weights, 
    double* c_calc)
{

  // Find maximum L
  int l_max = -1;
  for (int i=0; i<l_len; i++) l_max = (l[i] > l_max) ? l[i] : l_max;

  // Calculate shape of output and relabel common dimensions
  int N_bessel = l_max/2 + 1; // Number of spherical bessel functions
  int &N_dists = x_shape[0];
  int &N_qbins = x_shape[1];
  int &N_batch = x_shape[2];
  int &N_mols  = x_shape[3];

  // Calculate jn recursive relation coefficients once and use for all jn calls
  double* recursive_coeffs[3];
  recursive_coefficients(N_bessel, recursive_coeffs); // Must free dynamic memory
  double* jn = new double[N_bessel*N_qbins*N_dists];  // allocate memory for jn

  // Calculate the C coefficients
  //   Nested loops minimize index calculations and improve readability.
  //   The loops cover the minimum number of calculations needed and are 
  //   further optimized at compile time
  long int Y_idx, weight_idx;  // index variables for input values
  long int j_idx_shift, amps_idx_shift, c_idx_shift, x_shift;
  for (int batch_idx=0; batch_idx<N_batch; batch_idx++) { // Batches
    for (int mol_idx=0; mol_idx<N_mols; mol_idx++) { // Molecules (ensemble)
      weight_idx = batch_idx*N_mols+mol_idx;
      x_shift = batch_idx*N_dists*N_qbins*N_mols;
      x_shift += mol_idx*N_dists*N_qbins;
      spherical_j(x+x_shift, N_dists*N_qbins, l_max, jn, recursive_coeffs);
      for (int dist_idx=0; dist_idx<N_dists; dist_idx++) { // Distances in mol
        amps_idx_shift = dist_idx*N_qbins;
        for (int l_idx=0; l_idx<l_len; l_idx++) { // LMK quantum numbers
          c_idx_shift = l_idx*N_batch*N_qbins;
          c_idx_shift += batch_idx*N_qbins;
          Y_idx = l_idx*N_dists*N_batch*N_mols;
          Y_idx += batch_idx*N_mols*N_dists;
          Y_idx += mol_idx*N_dists;
          Y_idx += dist_idx;
          j_idx_shift = (l[l_idx]/2)*N_dists*N_qbins;
          j_idx_shift += dist_idx*N_qbins;
          for (int q_idx=0; q_idx<N_qbins; q_idx++) { // q bins on detector
            c_calc[c_idx_shift+q_idx] += 
                scat_amps[amps_idx_shift+q_idx]
                *mol_weights[weight_idx]
                *(c_prefactor_real[l_idx]*Ylk_real[Y_idx]
                    -c_prefactor_imag[l_idx]*Ylk_imag[Y_idx])
                *jn[j_idx_shift+q_idx];
          }
        }
      }
    }
  }

  // Free dynamic memory
  delete[] jn;
  for (int i=0; i<3; i++) delete[] recursive_coeffs[i];

  return;
}


void recursive_coefficients(int N_orders, double** coeffs)
{
  // Dynamically initialize memory for recursive coefficients
  for (int i=0; i<3; i++) coeffs[i] = new double[N_orders];

  double n; // Order of spherical bessel function
  for (int n_idx=0; n_idx<N_orders; n_idx++) { // Function order
    n = (n_idx)*2; // Only considering even orders
    coeffs[0][n_idx] = (2*n + 1)*(2*n + 3);
    coeffs[1][n_idx] = -1*(4*n + 2)/(2*n - 1);
    coeffs[2][n_idx] = -1*(2*n + 3)/(2*n - 1);
  }

  return;
}


void spherical_j(
    double* x,
    unsigned long long int x_len,
    int n_max,
    double* output,
    double** inp_recursive_coeffs)
{
  int N_orders = n_max/2 + 1; // Number of bessel function orders to calculate
  // Variables used to find the sign of cos(x)
  double two_pi = 2*3.14159265359;
  double pi_half = two_pi/4;
  double three_half_pi = 3*two_pi/4;
  double x_mod, cos_sign;

  // Common variables and calculate recursive variables if not provided
  double jn, jn_prev, sinx, x_sq;
  double** recursive_coeffs = nullptr;
  if (inp_recursive_coeffs != nullptr) {
    recursive_coeffs = inp_recursive_coeffs;
  }
  else {
    recursive_coeffs = new double*[3];
    recursive_coefficients(N_orders, recursive_coeffs);
  }
 
  // Calculate spherical bessel functions
  for (unsigned long long int x_idx=0; x_idx<x_len; x_idx++) { // x values
    // Heavily used or relatively expensive calculations, calculate once
    x_sq = x[x_idx]*x[x_idx];
    sinx = sin(x[x_idx]);  // sin(x) and cos(x) are the slowest part

    // Calculate j0(x)
    if (x_sq < 1e-16) { // j0(x) = 1 in limit of x->0
      jn_prev = 1;
    }
    else {
      jn_prev = sinx/x[x_idx];
    }
    output[x_idx] = jn_prev;

    // Calculate j2(x)
    if (N_orders >= 2) {
      if (x_sq < 1e-16) { // j2(x) = 0 in limit of x->0
        jn = 0;
      }
      else {
        // Use |cos| = sqrt(1 - sin^2) for speedup and find the sign
        x_mod = fmod(x[x_idx], two_pi);
        cos_sign = ((x_mod > pi_half) && (x_mod < three_half_pi)) ? -1 : 1;
        jn  = (3/x_sq - 1)*jn_prev - cos_sign*3*sqrt(1-sinx*sinx)/x_sq;
      }
      output[x_len+x_idx] = jn;
    }

    // Calculate jn(x) where n > 2 and n is even
    for (int n_idx=2; n_idx<N_orders; n_idx++) { // order (n)
      output[n_idx*x_len+x_idx] = 
          (recursive_coeffs[0][n_idx-1]/x_sq + recursive_coeffs[1][n_idx-1])*jn
          + recursive_coeffs[2][n_idx-1]*jn_prev;
      jn_prev = jn;
      jn = output[n_idx*x_len+x_idx];
    }
  }

  // Free dynamic memory
  if (!inp_recursive_coeffs) {
    for (int i=0; i<3; i++) delete[] recursive_coeffs[i];
    delete[] recursive_coeffs;
  }

  return;
}
