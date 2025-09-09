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


    double A_x_1 = -2.0;
    double B_x_1 = 1.3;

    double a_1= 0.5;
    double b_1 = 0.2;


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


    double OS_reference[36] = {0.44708262,  -1.0538376 ,   2.80339051,  -8.1134742 ,
        25.13188312, -82.42079362,   0.42153504,  -0.67427358,
         1.13771447,  -1.64258173,   0.51442066,  12.04569143,
         0.71679205,  -1.08738835,   2.11187603,  -4.90609904,
        13.74327963, -45.67741257,   1.27802542,  -1.47650553,
         2.06309186,  -2.44684721,  -0.32458981,  23.22399742,
         2.74097837,  -2.80937637,   4.36135593,  -8.39918562,
        22.15285105, -77.14250727,   6.23585224,  -4.90958611,
         5.99328894,  -5.56446149,  -4.03809881,  72.73723953}; //72.73723953

    double *OS_calc = Overlap_OS_matrix(A_x_1, B_x_1, a_1, b_1, 5, 5);

    for (unsigned int i = 0; i < 36; i++){
         if (fabs( OS_reference[i] - OS_calc[i]) > 1e-5){
             printf("\n\n!!! FAILED TEST cartesian_gaussian_test at Overlap integral\n\n");
            return 1;
         }
    }

    double kinetic_OS_reference[36] = {-1.34854717e-01,  1.67753741e-02,  3.45757237e-01, -1.60368758e+00,
        5.69430175e+00, -1.83716905e+01, -6.71014963e-03, -2.73157611e-01,
        6.75025779e-01, -1.24044899e+00,  9.33925902e-01,  6.45653844e+00,
       -2.55916720e-02, -2.59945087e-01,  2.95621609e-01,  4.78193988e-02,
       -1.85010958e+00,  6.79851582e+00,  9.05577356e-02, -6.90155538e-01,
        8.11033624e-01, -8.53989316e-01, -6.60276926e-01,  8.77829692e+00,
        1.99287199e-01, -1.01369098e+00, -1.51850524e-01,  2.84066984e+00,
       -1.24776048e+01,  3.75981478e+01,  7.67707332e-01, -2.50059475e+00,
        8.04846097e-02,  1.11953307e+00, -6.31768747e+00,  1.02080135e+01}; 

    double *kinetic_OS_calc = kinetic_OS_matrix(A_x_1, B_x_1, a_1, b_1, 5, 5);

    for (unsigned int i = 0; i < 36; i++){
         if (fabs( kinetic_OS_reference[i] - kinetic_OS_calc[i]) > 1e-5){
             printf("\n\n!!! FAILED TEST cartesian_gaussian_test at Kinetic integral\n\n");
            return 1;
         }
    }

    return 0;
}