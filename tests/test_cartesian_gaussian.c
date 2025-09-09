#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "gaussians.h"

int main(void){

    double        a[6] = {0.5,  0.2,  0.4, 0.7, 1.4,  2.20};
    double      A_x[6] = {0.0, -1.2,  2.4, 1.7, 0.6,  1.45};
    double        x[6] = {0.0,  1.2, -1.4, 0.7, 3.4, -0.10};
    unsigned int ii[6] = {  0,   1,     2,   3,   4,     5};

    double results[6] = {1.0, 0.75841, 0.04478, -0.49659, 0.00105, -0.04531};

    for (unsigned int i = 0; i < 6; i++) {
        double value = cartesian_gaussian(ii[i], A_x[i], a[i], x[i]);

        // printf("Calculated = %f, reference = %f, %f \n", value, results[i], fabs(value - results[i])); 

        if (fabs(value - results[i]) > 1e-5){
            printf("Error in a = %f, A_x = %f, x = %f, i = %u\n", a[i], A_x[i], x[i], ii[i]);
            printf("\n\n!!! FAILED TEST cartesian_gaussian_test\n\n");
            return 1;
        }
    }

    // printf("\n\n--- PASSED TEST cartesian_gaussian_test\n\n");

    return 0;
}