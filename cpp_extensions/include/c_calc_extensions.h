#ifndef BIGR_INCLUDE_C_CALC_H
#define BIGR_INCLUDE_C_CALC_H


/**
 * Recursively calculates the even Spherical Bessel functions on x and runs
 * 50X faster than the scipy implementation. This calculation becomes unstable
 * at small values of x. These instabilities become larger at higher orders
 * (larger n). Argument x must be a flattened C array with original
 * dimensions of [d1, d2, ..., dM] in row major form.
 *
 * The standard recursive relations (from https://dlmf.nist.gov/10.51) are
 * used to derive the following relation between orders separated by 2
 *
 * j_{n+2}(x) = [(2n+1)*(2n+3)/x^2 - (4n+2)/(2n-1)]j_n - [(2n+3)/(2n-1)]j_{n-2}
 * 
 * @param x (double*): Pointer to array of evaluation points (deltaR*q) with
 *          contiguous memory in row major form.
 * @param x_len (int): The length of x.
 * @param l_max (int): The largest order of functions to evaluate.
 * @param output (double*): Pointer to the memory contiguous output matrix of
 *               of shape [l_max/2+1, d1, ..., dM] that is edited with result.
 */
extern "C" void spherical_j(
    double* x,
    unsigned long long int x_len,
    int n_max,
    double* output,
    double** inp_recursive_coeffs=nullptr);


/**
 * Calculates the recursive coefficients needed for the spherical_j() function
 * and saves them in dynamic 1D arrays of length N_orders. The function calling
 * this is responsible for deleting these arrays.
 *
 * coeffs[0][n/2] = (2n+1)*(2n+3)
 * coeffs[1][n/2] = -(4n+2)/(2n-1)
 * coeffs[2][n/2] = -(2n+3)/(2n-1)
 *
 * @param N_orders (int): Number of even anisotropy orders to calculate.
 * @param coeffs (double*[3]): Array of pointers for dynamically made arrays of 
 *               coefficients. This array is filled with the results.
 */
void recursive_coefficients(int N_orders, double** coeffs); 


/**
 * Calculates the C coefficients using the faster C++ implementation. Spherical
 * Harmonic (Ylk), prefactors, scattering amplitudes, and ensemble weights for 
 * the molecules must be provided for this function to sum over and avoid 
 * running out of memory.
 *
 * @param x (double*): Pointer to a matrix of evaluation points (deltaR*q) with 
 *          contiguous memory and shape [N_batch, N_mols, N_dists, N_qbins].
 * @param x_shape (int*): Array of sizes for each dimension of x.
 * @param l (int*): The principal angular momentum quantum number of each C.
 *          This corresponds to the order of the spherical bessel function.
 * @param l_len (int): The number of angular momentum quantum numbers.
 * @param c_prefactor_real (double*): Real part of the C coefficient prefactors.
 * @param c_prefactor_imag (double*): Imaginary part of the C prefactors.
 * @param Ylk_real (double*): Real part of the spherical harmonic factor.
 * @param Ylk_imag (double*): Imaginary part of the spherical harmonic factor.
 * @param scat_amps (double*): Precalculated scattering amplitudes. 
 * @param c_calc (double*): Pointer to the memory contiguous output matrix of
 *               shape [l_len, N_batch, N_qbins] to be filled with the results.
 */
extern "C" void calculate_c(
    double* x,
    int* x_shape,
    int* l,
    int l_len,
    double* c_prefactor_real,
    double* c_prefactor_imag,
    double* Ylk_real,
    double* Ylk_imag,
    double* scat_amps,
    double* weights,
    double* c_calc); 

#endif // BIGR_INCLUDE_C_CALC_H
