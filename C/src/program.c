#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "gaussians.h"

int main(void){
    double a = 0.5;
    double A_x = 0.0;
    double x = 1.3;
    unsigned int ii = 1;
    double value = cartesian_gaussian(ii, A_x, a, x);
    printf("i = %i, x = %f : %f\n", ii, x, value);
    return 0;
}