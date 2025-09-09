#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "gaussians.h"

double cartesian_gaussian(unsigned int ii, double A_x, double a, double x){
    double x_a = (x-A_x);

    return pow(x_a, ii) * exp(-a * pow(x_a, 2)); 
} 

double gaussian_overlap_distribution(unsigned int ii, double A_x, double a, unsigned int jj, double B_x, double b, double x){

    // gaussian overlap distribution Omega_{i,j}

    double p = a+b;
    double mu = a*b/p;
    double P_x = (a*A_x  + b * B_x) / p;
    double X_ab = A_x - B_x;
    double x_a = (x-A_x);
    double x_b = (x-B_x);
    double x_p = (x-P_x);

    return exp(-mu * pow(X_ab, 2)) * pow(x_a, ii) *  pow(x_b, jj) * exp(-p * pow(x_p, 2));
}

