#ifndef GAUSSIANS_H
#define GAUSSIANS_H

double cartesian_gaussian(unsigned int ii, double A_x, double a, double x);
double gaussian_overlap_distribution(unsigned int ii, double A_x, double a, unsigned int jj, double B_x, double b, double x);

#endif