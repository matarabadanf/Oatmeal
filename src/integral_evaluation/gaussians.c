#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "gaussians.h"

double cartesian_gaussian(unsigned int ii, double A_x, double a, double x){
    double x_a = (x-A_x);

    return pow(x_a, ii) * exp(-a * pow(x_a, 2)); 
} 

// int main(){

//     double a = 0.5;
//     double A_x = 0.0;
//     double x = 1.3;
//     uint ii = 1;

//     double value = cartesian_gaussian(ii, A_x, a, x);

//     printf("i = %i, x = %f : %f\n", ii, x, value);


//     return 0;
// }