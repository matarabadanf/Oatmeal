#ifndef SCF_UTILS_H
#define SCF_UTILS_H


void calc_g_matrix_comp(
    const double complex *P,      /* input density matrix, size dim*dim */
    const double complex *eri,    /* input ERI, size dim*dim*dim*dim */
    double complex *g_mat,        /* output G matrix, size dim*dim we will use prealocated so python deals with garbage collection*/
    size_t dim
);

#define TENSOR_INDEX_SYM(i, j, k, l, dim) ((i * dim + j) * dim + k) * dim + l


#endif