#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "gaussians.h"

int main(void){

    // Gaussian definition

    double        a[6] = {0.5,  0.2,  0.4, 0.7, 1.4,  2.20};
    double      A_x[6] = {0.0, -1.2,  2.4, 1.7, 0.6,  1.45};
    double        x[6] = {0.0,  1.2, -1.4, 0.7, 3.4, -0.10};
    unsigned int ii[6] = {  0,   1,     2,   3,   4,     5};

    double results[6] = {1.0, 0.75841, 0.04478, -0.49659, 0.00105, -0.04531};

    for (unsigned int i = 0; i < 6; i++) {
        double value = cartesian_gaussian(ii[i], A_x[i], a[i], x[i]);

        // printf("Calculated = %f, reference = %f, %f \n", value, results[i], fabs(value - results[i])); 

        if (fabs(value - results[i]) > 1e-5){
            // printf("Error in a = %f, A_x = %f, x = %f, i = %u\n", a[i], A_x[i], x[i], ii[i]);
            printf("\n\n!!! FAILED TEST cartesian_gaussian_test at gaussian_definition\n\n");
            return 1;
        }
    }

    // Gaussian overlap 

    double a_1= 0.5;
    double A_x_1 = -2.0;
    double b_1 = 0.2;
    double B_x_1 = 1.3;

    // combining i with j it is 01 12 20 22
    unsigned int ii_1[4] = {0, 1, 2, 2};
    unsigned int jj_1[4] = {1, 2, 0, 2};
    
    double x_1[4] = {1.005502751375687,  4.007003501750875, 0.005002501250624292, -1.9959979989994991};
    double reverence_overlap_distributions[4] = {-0.0031626191851994367,  1.4845135925484385e-07,  0.38514684461289406,  1.9811813720615314e-05};


    for (unsigned int i = 0; i < 4; i++) {
        double value = gaussian_overlap_distribution(ii_1[i], A_x_1, a_1, jj_1[i], B_x_1, b_1, x_1[i]);

        // printf("Calculated = %f, reference = %f, %f \n", value, reverence_overlap_distributions[i], fabs(value - reverence_overlap_distributions[i])); 

        if (fabs(value - reverence_overlap_distributions[i]) > 1e-5){
            printf("\n\n!!! FAILED TEST cartesian_gaussian_test at gaussian_overlap\n\n");
            return 1;
        }
    }

    return 0;
}