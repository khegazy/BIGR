#ifndef BIGR_INCLUDE_SPHERICAL_J_H
#define BIGR_INCLUDE_SPHERICAL_J_H


/** 
 * Recursively calculates the even spherical bessel functions on x and runs
 * 50X faster than the numpy implementation. This calculation becomes unstable 
 * at small values of x. These instabilities becomes larger at higher bessel 
 * function orders. Argument x must be a flattened C array with original 
 * dimensions of (d1, d2, ..., N_qbins) in row major form.
 *
 * The standard recursive relations (from https://dlmf.nist.gov/10.51) are
 * used to derive the following relation between orders separated by 2
 *
 * j_{n+2}(x) = [(2n+1)*(2n+3)/x^2 - (4n+2)/(2n-1)]j_n - [(2n+3)/(2n-1)]j_{n-2}
 * 
 * @param x (double*): Pointer to array of evaluation points (deltaR*q).
 * @param x_len (int): The number of locations in x (N_qbins).
 * @param l_max (int): The largest order of functions to evaluate.
 * @param output (double*): Pointer to the contiguous output matrix of
 *               of shape [l_max/2+1, N_qbins] that is edited with result.
 */
extern "C" void spherical_j(
    double* x,
    unsigned long long int x_len,
    int n_max,
    double* output,
    int N_qbins=-1,
    double** inp_recursive_coeffs=nullptr);


/**
 * Calculates the recursive coefficients needed for the spherical_j() function
 * and saves them in dynamic 1D arrays of length N_anis. The function calling
 * this is responsible for deleting these arrays.
 *
 * coeffs[0][n/2] = (2n+1)*(2n+3)
 * coeffs[1][n/2] = -(4n+2)/(2n-1)
 * coeffs[2][n/2] = -(2n+3)/(2n-1)
 *
 * @param N_anis (int): Number of even anisotropy orders to calculate.
 * @param coeffs (double*[3]): Array of pointers for dynamically made arrays of 
 *               coefficients. This array is filled with the results.
 */
void recursive_coefficients(int N_anis, double** coeffs); 


/**
 * Calculates the C coefficients using the faster C++ implementation. Spherical
 * Harmonic (Y), prefactors, scattering amplitudes, and ensemble weights for 
 * the molecules must be provided for this function to sum over and avoid 
 * running out of memory.
 *
 * @param x (double*): Pointer to evaluation points (deltaR*q) in a contiguous 
 *          matrix with shape [N_batch, N_mols, N_dists, N_qbins].
 * @param x_shape (int*): Array of sizes for each dimension of x.
 * @param l (int*): The angular momentum quantum numbers (anisotropy).
 * @param l_len (int): The number of angular momentum quantum numbers.
 * @param C_prefactor_real (double*): Real part of the C coefficient prefactors.
 * @param C_prefactor_imag (double*): Imaginary part of the C prefactors.
 * @param Y_real (double*): Real part of the spherical harmonic factor.
 * @param Y_imag (double*): Imaginary part of the spherical harmonic factor.
 * @param scat_amps (double*): Precalculated scattering amplitudes. 
 * @param c_coeffs (double*): Pointer to the contiguous output matrix of shape
 *                 [l_len, N_batch, N_qbins] to be filled with the results.
 */
extern "C" void calc_c_coeffs(
    double* x,
    int* x_shape,
    int* l,
    int l_len,
    double* C_prefactor_real,
    double* C_prefactor_imag,
    double* Y_real,
    double* Y_imag,
    double* scat_amps,
    double* weights,
    double* c_coeffs); 

#endif // BIGR_INCLUDE_SPHERICAL_J_H
