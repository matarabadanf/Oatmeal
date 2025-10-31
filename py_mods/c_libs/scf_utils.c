#include <stddef.h>
#include <complex.h>
#include "scf_utils.h"

void calc_g_matrix_comp(
    const double complex *P,      /* input density matrix, size dim*dim */
    const double complex *eri,    /* input ERI, size dim*dim*dim*dim */
    double complex *g_mat,        /* output G matrix, size dim*dim we will use prealocated so python deals with garbage collection*/
    size_t dim
) {
    /*
    Calculate G matrix using: 

    G_{mu, nu} = sum_{la, si} P_{la, si} * ( <mu nu|la si> - 0.5 * <mu la|nu si> )
    
    */

    size_t nn = dim * dim;
    for (size_t i = 0; i < nn; ++i) g_mat[i] = 0.0 + 0.0*I;

    for (size_t mu = 0; mu < dim; ++mu) {
        for (size_t nu = 0; nu < dim; ++nu) {
            double complex acc = 0.0 + 0.0*I;
            for (size_t la = 0; la < dim; ++la) {
                for (size_t si = 0; si < dim; ++si) {
                    double complex P_ls = P[la * dim + si];
                    size_t idx1 = TENSOR_INDEX_SYM(mu, nu, la, si, dim); /* eri[mu,nu,la,si] */
                    size_t idx2 = TENSOR_INDEX_SYM(mu, la, nu, si, dim); /* eri[mu,la,nu,si] */
                    acc += P_ls * (eri[idx1] - 0.5 * eri[idx2]);
                }
            }
            g_mat[mu * dim + nu] = acc;
        }
    }
}
