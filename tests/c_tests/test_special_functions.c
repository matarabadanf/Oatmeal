#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "special_functions.h"

int main(void){

    unsigned int k[4] = {1, 2, 3, 15};
    double factorials[4] = {1, 2, 6, 1.3076744e+12};

    for (unsigned int i = 0; i < 4; i++) {

        double test_value = factorial(k[i]);
        //printf("%f\n", fabs(test_value / factorials[i]));
        
        if (fabs(fabs(test_value / factorials[i]) - 1) > 1e-5){

            printf("\n\n!!! FAILED TEST special_functions_test at gamma func\n\n");
            return 1;
        }
    }

    double x_sample_boys[17] = {0.1, 0.2, 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 6, 8, 10, 12, 14, 30};
    double boys_0[17] = {0.96764331, 0.93715003, 0.85562439, 0.80849581, 0.74682413,
       0.66335095, 0.59814401, 0.54629197, 0.50434356, 0.46984704,
       0.4410407 , 0.36160815, 0.31330869, 0.28024739, 0.25583143,
       0.23685408, 0.16180216};

    for (unsigned int i = 0; i < 17; i++) {

        double test_value = boys_hypergeom(0, x_sample_boys[i], 100);
        // printf("%f, %f\n", test_value, fabs(test_value / boys_0[i]));
        
        if (fabs(fabs(test_value /  boys_0[i]) - 1) > 1e-3){

            printf("\n\n!!! FAILED TEST special_functions_test at boys func n = 0\n\n");
            return 1;
        }
    }

    double boys_1[17] = {0.31402947, 0.29604819, 0.24909373, 0.22279322, 0.18947235,
       0.14674026, 0.11570218, 0.09284139, 0.07575942, 0.06280709,
       0.05284063, 0.02992745, 0.01956083, 0.0140101 , 0.01065939,
       0.00845904, 0.0026967};

    for (unsigned int i = 0; i < 16; i++) {

        double test_value = boys_hypergeom(1, x_sample_boys[i], 100);
        // printf("%f, %f\n", test_value, fabs(test_value / boys_1[i]));
        
        if (fabs(fabs(test_value /  boys_1[i]) - 1) > 1e-3){

            printf("\n\n!!! FAILED TEST special_functions_test at boys func n = 1\n\n");
            return 1;
        }
    }

    double boys_9[17] = {4.80805503e-02, 4.39263813e-02, 3.35116308e-02, 2.79907955e-02,
       2.13802797e-02, 1.36697004e-02, 8.75984046e-03, 5.62721746e-03,
       3.62431065e-03, 2.34083440e-03, 1.51640251e-03, 2.76360502e-04,
       5.38480667e-05, 1.14193418e-05, 2.68247023e-06, 7.07184865e-07,
       5.53260170e-10};

    for (unsigned int i = 0; i < 16; i++) {

        double test_value = boys_hypergeom(9, x_sample_boys[i], 100);
        // printf("%f, %f\n", test_value, fabs(test_value / boys_9[i]));
        
        if (fabs(fabs(test_value /  boys_9[i]) - 1) > 1e-3){

            printf("\n\n!!! FAILED TEST special_functions_test at boys func n = 9\n\n");
            return 1;
        }
    }


    return 0;
}