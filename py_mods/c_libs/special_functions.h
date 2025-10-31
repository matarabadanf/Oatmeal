#ifndef SPECIAL_FUNCTIONS_H
#define SPECIAL_FUNCTIONS_H

double factorial(unsigned int k);
double pochhammer(double a, unsigned int k);
double kummer_confluent_series(double a, double b, double x, unsigned int k);
double boys_hypergeom(unsigned int n, double x, unsigned int k);
double binomial(unsigned int n, unsigned int m);

#endif