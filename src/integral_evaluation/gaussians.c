#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "gaussians.h"
#include "special_functions.h"

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

double *Overlap_OS_matrix(double Ax, double Bx, double a, double b, unsigned int ii, unsigned int jj) {

    // builds the Obara-Saika recurrence matrix for overlap integrals. 

    // Each element (using i-recurrence) is given by:
    //    S_{i+1,\,j} = X_{PA} S_{ij} + \frac{1}{2p}(iS_{i-1,\,j} + jS_{i,\,j-1})
    // or:
    //    S_{i,\,j} = X_{PA} S_{i-1,j} + \frac{1}{2p}((i-1)*S_{i-2,\,j} + jS_{i-1,\,j-1})

    unsigned int max_dim = MAX(ii, jj) + 1;
    double *S = malloc(max_dim * max_dim * sizeof(double));
    if (!S) return NULL;  // safe malloc check

    double p = a + b;
    double mu = a * b / p;
    double X_ab = Bx - Ax;
    double X_pa = b / p * X_ab;
    double X_pb = -a / p * X_ab;

    // Base case
    S[0] = sqrt(PI / p) * exp(-mu * X_ab * X_ab);

    // First column (j = 0)
    for (unsigned int i = 1; i < max_dim; i++) {
        double term2 = (i >= 2) ? (i - 1) / (2.0 * p) * S[(i - 2) * max_dim] : 0.0; // if i >= 2, then (i - 1) / (2.0 * p) * S[(i - 2) * max_dim] : else This is the ternary operator in c 
        S[i * max_dim] = X_pa * S[(i - 1) * max_dim] + term2;
    }

    // First row (i = 0)
    for (unsigned int j = 1; j < max_dim; j++) {
        double term2 = (j >= 2) ? (j - 1) / (2.0 * p) * S[j - 2] : 0.0;
        S[j] = X_pb * S[j - 1] + term2;
    }

    // General case (i > 0, j > 0) using i-recurrence 
    for (unsigned int i = 1; i < max_dim; i++) {
        for (unsigned int j = 1; j < max_dim; j++) {
            double term1 = X_pa * S[(i - 1) * max_dim + j];
            double term2 = 0.0;
            double term3 = 0.0;

            // The second and third terms are evaluated separately way, and it is necessary to check for negative indices: 
            if (i >= 2){
                term2 += (i - 1) / (2.0 * p) * S[(i - 2) * max_dim + j];
            }

            term3 = j / (2.0 * p) * S[(i - 1) * max_dim + (j - 1)];

            S[i * max_dim + j] = term1 + term2 + term3;
        }
    }

    return S;
}


double Overlap_OS(double Ax, double Bx, double a, double b, unsigned int ii, unsigned int jj){

    unsigned int max_dim = MAX(ii, jj) + 1;
    double *ang_matrix = Overlap_OS_matrix(Ax, Bx, a, b, ii, jj);

    double result = MATRIX_VAL(ii, jj, max_dim, ang_matrix);

    free(ang_matrix);

    return result;
}

double *kinetic_OS_matrix(double Ax, double Bx, double a, double b, unsigned int ii, unsigned int jj){


    unsigned int max_dim = MAX(ii, jj) + 1;
    double *S =  Overlap_OS_matrix(Ax, Bx, a, b, ii, jj);
    double *T = malloc(max_dim * max_dim * sizeof(double));
    if (!S) return NULL;  // safe malloc check
    if (!T) return NULL; 


    double p = a + b;
    double X_ab = Bx - Ax;
    double X_pa = b / p * X_ab;
    double X_pb = -a / p * X_ab;
    

    // Base case
    T[0] = (a - 2 * pow(a, 2) * (pow(X_pa, 2) + 1/(2*p))) * S[0];

    //first column
    for (unsigned int i = 1; i < max_dim; i++) {
        double term1 = X_pa * MATRIX_VAL(i-1, 0, max_dim, T); 
        double term2 = (i >= 2) ? 1/(2*p) * (i-1) * MATRIX_VAL(i-2, 0, max_dim, T): 0; 
        double term3 = b/p * 2 * a * MATRIX_VAL(i, 0, max_dim, S); 
        double term4 = (i >= 2) ?  - b/p * (i-1) * MATRIX_VAL(i-2, 0, max_dim, S) : 0; 

        MATRIX_VAL(i, 0, max_dim, T) = term1 + term2 + term3 + term4;
    }

    
    //first row
    for (unsigned int j = 1; j < max_dim; j++) {
        double term1 = X_pb * MATRIX_VAL(0, j-1, max_dim, T); 
        double term2 = (j >= 2) ? 1/(2*p) * (j-1) * MATRIX_VAL(0, j-2, max_dim, T): 0; 
        double term3 = a/p * 2 * b * MATRIX_VAL(0, j, max_dim, S); 
        double term4 = (j >= 2) ?  - a/p * (j-1) * MATRIX_VAL(0, j-2, max_dim, S) : 0; 

        MATRIX_VAL(0, j, max_dim, T) = term1 + term2 + term3 + term4;
    }


        // General case (i > 0, j > 0) using i-recurrence 
    for (unsigned int i = 1; i < max_dim; i++) {
        for (unsigned int j = 1; j < max_dim; j++) {
            double term1 = X_pa * MATRIX_VAL(i-1, j, max_dim, T); 
            double term2 = (i >= 2) ? 1/(2*p) * (i-1) * MATRIX_VAL(i-2, j, max_dim, T): 0; 
            double term2_2 = 1/(2*p) * (j) * MATRIX_VAL(i-1, j-1, max_dim, T); 
            double term3 = b/p * 2 * a * MATRIX_VAL(i, j, max_dim, S); 
            double term4 = (i >= 2) ?  - b/p * (i-1) * MATRIX_VAL(i-2, j, max_dim, S) : 0; 

            MATRIX_VAL(i, j, max_dim, T) = term1 + term2 + term2_2 + term3 + term4;
        }
    }

    return T;
}

double Kinetic_OS(double Ax, double Bx, double a, double b, unsigned int ii, unsigned int jj){

    unsigned int max_dim = MAX(ii, jj) + 1;
    double *kinetic_matrix = kinetic_OS_matrix(Ax, Bx, a, b, ii, jj);

    double result = MATRIX_VAL(ii, jj, max_dim, kinetic_matrix);

    free(kinetic_matrix);

    return result;
}

double Hermite_gaussian(double x, double p, double P_x, int t) {
    if (t == 0) {
        return exp(-p * pow(x - P_x, 2));
    } 
    else if (t < 0) {
        return 0.0;
    } 
    else {
        return 2.0 * p * ((x - P_x) * Hermite_gaussian(x, p, P_x, t - 1) - (t - 1) * Hermite_gaussian(x, p, P_x, t - 2));
    }
}

double Hermite_coefficient(int i, int j, int t, double Ax, double Bx, double a, double b){
    // Hermite coefficient E(i, j, t, Ax, Bx, a, b)
    // Compute parameters

    double X_ab = Bx - Ax;
    double p = a + b;
    double mu = (a * b) / p;
    double X_pa = b / p * X_ab;
    double X_pb = -a / p * X_ab;

    // Edge cases
    if (i < 0 || j < 0 || t < 0 || t > (i + j)) {
        return 0.0;
    } 
    else if (i == 0 && j == 0 && t == 0) {
        return exp(-mu * X_ab * X_ab);
    }

    // Recursion
    if (t > 0) {
        return ((i * Hermite_coefficient(i - 1, j, t - 1, Ax, Bx, a, b)) +
                (j * Hermite_coefficient(i, j - 1, t - 1, Ax, Bx, a, b))) / (2.0 * p * t);
    }
    else if (t == 0 && i > 0) {
        return X_pa * Hermite_coefficient(i - 1, j, t, Ax, Bx, a, b) +
               Hermite_coefficient(i - 1, j, 1, Ax, Bx, a, b);
    } 
    else if (t == 0 && j > 0) {
        return X_pb * Hermite_coefficient(i, j - 1, t, Ax, Bx, a, b) +
               Hermite_coefficient(i, j - 1, 1, Ax, Bx, a, b);
    }

    return 0;

}

double cartesian_from_hermite(double x, int i, int j, double Ax, double Bx, double a, double b) {
    // Calculates the value of the overlap using the Hermite summation
    double total = 0.0;
    double p = a + b;
    double Px = (a * Ax + b * Bx) / p;

    for (int t = 0; t <= i + j; t++) {
        total += Hermite_coefficient(i, j, t, Ax, Bx, a, b) * Hermite_gaussian(x, p, Px, t);
    }

    return total;
}