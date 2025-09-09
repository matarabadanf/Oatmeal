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

#define MATRIX_VAL(i, j, n_cols, matrix) matrix[ (i) * n_cols + (j) ]

double *Overlap_OS_matrix(double Ax, double Bx, double a, double b, unsigned int ii, unsigned int jj){

    unsigned int max_dim = MAX(ii, jj) + 1;

    // printf("\nMax_dim is %d, I = %d, j=%d", max_dim, ii, jj );

    double *angular_momentum_matrix = malloc(max_dim * max_dim * sizeof(double));

    double p = a + b;
    double mu = (a * b)/p;
    double X_ab = (Bx-Ax);
    double X_pa = b/p * X_ab;
    double X_pb = -a/p * X_ab;


    // ground case
    MATRIX_VAL(0, 0, max_dim, angular_momentum_matrix) = pow(PI/p, 0.5) * exp(-mu * pow(X_ab, 2));

    // i row
    for (int i = 1; i < max_dim; i++){

        if (i-2 < 0){
            MATRIX_VAL(i, 0, max_dim, angular_momentum_matrix) = X_pa * MATRIX_VAL(i-1, 0, max_dim, angular_momentum_matrix);

            // printf("\n\n\n%f",  MATRIX_VAL(i, 0, max_dim, angular_momentum_matrix));
        }

        else{
            MATRIX_VAL(i, 0, max_dim, angular_momentum_matrix) = X_pa * MATRIX_VAL(i-1, 0, max_dim, angular_momentum_matrix) +
             1/(2*p) *((i-1) * MATRIX_VAL(i-2, 0, max_dim, angular_momentum_matrix));
            
            // printf("\n\n\n\n%f",  MATRIX_VAL(i, 0, max_dim, angular_momentum_matrix));
        }
    }

    // j_row
    for (int j = 1; j < max_dim; j++){

        if (j-2 < 0){
            MATRIX_VAL(0, j, max_dim, angular_momentum_matrix) = X_pb * MATRIX_VAL(0, j-1, max_dim, angular_momentum_matrix);

            // printf("\n\n\n%f",  MATRIX_VAL(0, j, max_dim, angular_momentum_matrix));
        }

        else{
            MATRIX_VAL(0, j, max_dim, angular_momentum_matrix) = X_pb * MATRIX_VAL(0, j-1, max_dim, angular_momentum_matrix) +
             1/(2*p) *((j-1) * MATRIX_VAL(0, j-2, max_dim, angular_momentum_matrix));
            
            // printf("\n\n\n\n%f",  MATRIX_VAL(0, j, max_dim, angular_momentum_matrix));
        }
    }


    // general case 
    for (int total = 1; total < max_dim; total++) {
        for (int i = total; i < max_dim; i++) {
            double term1 = X_pa * MATRIX_VAL(i-1, total, max_dim, angular_momentum_matrix);
            double term2 = 0.0;
            if (i-2 >= 0)
                term2 += (i-1) * MATRIX_VAL(i-2, total, max_dim, angular_momentum_matrix);
            if (total-1 >= 0)
                term2 += (total) * MATRIX_VAL(i-1, total-1, max_dim, angular_momentum_matrix);
            term2 *= 1.0/(2*p);
            MATRIX_VAL(i, total, max_dim, angular_momentum_matrix) = term1 + term2;
        }
        for (int j = total; j < max_dim; j++) {
            double term1 = X_pb * MATRIX_VAL(total, j-1, max_dim, angular_momentum_matrix);
            double term2 = 0.0;
            if (total-1 >= 0 && j-1 >= 0)
                term2 += (total) * MATRIX_VAL(total-1, j-1, max_dim, angular_momentum_matrix);
            if (j-2 >= 0)
                term2 += (j-1) * MATRIX_VAL(total, j-2, max_dim, angular_momentum_matrix);
            term2 *= 1.0/(2*p);
            MATRIX_VAL(total, j, max_dim, angular_momentum_matrix) = term1 + term2;
        }
    }

    return angular_momentum_matrix;
}

double Overlap_OS(double Ax, double Bx, double a, double b, unsigned int ii, unsigned int jj){

    unsigned int max_dim = MAX(ii, jj) + 1;
    double *ang_matrix = Overlap_OS_matrix(Ax, Bx, a, b, ii, jj);

    double result = MATRIX_VAL(ii, jj, max_dim, ang_matrix);

    free(ang_matrix);

    return result;
}