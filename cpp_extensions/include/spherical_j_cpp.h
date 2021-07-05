#include <iostream>
#include <math.h>
#include <vector>

extern "C" void spherical_j(
    double* x, int x_len,
    int* l, int l_len, double* out); 

