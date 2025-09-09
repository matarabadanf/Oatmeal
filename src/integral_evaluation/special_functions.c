#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "special_functions.h"

double factorial(unsigned int k){
    double factorial = tgamma(k+1);

    return factorial;
}

double pochhammer(double a, unsigned int k){
    return tgamma(a+k) / tgamma(a);
}

double kummer_confluent_series(double a, double b, double x, unsigned int k){
    double m = 0;

    for (unsigned int i = 0; i < k; i++){
        double a_k = pochhammer(a, i);
        double b_k = pochhammer(b, i);
        double k_factorial = factorial(i);
        // # print(f"series {i}: {a_k} {b_k} {k_factorial},  {a_k / (b_k * k_factorial)} ")
        m += a_k / (b_k * k_factorial) * pow(x, i);
    }
    return m;
}

double boys_hypergeom(unsigned int n, double x, unsigned int k){
    double a = n+0.5;
    double b = n+1.5;
    return kummer_confluent_series(a, b, -x, k) / (2*n+1);
}