#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>

extern "C" void spherical_j(
    double* x, int* x_shape, int x_dim,
    int* l, int l_len, double* out); 

extern "C" void calc_coeffs_helper(
    double* x, int* x_shape, int x_dim,
    int* l, int l_len,
    double* C_real, double* C_imag,
    double* Y_real, double* Y_imag,
    double* scat_amps, double* weights, double* out); 
