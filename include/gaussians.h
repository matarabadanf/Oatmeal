#ifndef GAUSSIANS_H
#define GAUSSIANS_H

double cartesian_gaussian(unsigned int ii, double A_x, double a, double x);
double gaussian_overlap_distribution(unsigned int ii, double A_x, double a, unsigned int jj, double B_x, double b, double x);
double Overlap_OS(double Ax, double Bx, double a, double b, unsigned int ii, unsigned int jj);
double *Overlap_OS_matrix(double Ax, double Bx, double a, double b, unsigned int ii, unsigned int jj);
double Kinetic_OS(double Ax, double Bx, double a, double b, unsigned int ii, unsigned int jj);
double *kinetic_OS_matrix(double Ax, double Bx, double a, double b, unsigned int ii, unsigned int jj);



#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define MATRIX_VAL(i, j, n_cols, matrix) matrix[ (i) * n_cols + (j) ]
#define E 2.71828182845904523
#define PI 3.14159265358979323

#endif