#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "gaussians.h"
#include "special_functions.h"
#include "3d_utils.h"

void main(){
    printf("\nTest H\n");

    primitive_gaussian_t a = {{1, 1, 1}, 0.5, {{0,0,0}}, 0};



    normalize_3D_gaussian(&a, 0);
    
    
    double self_overlap = overlap_3D(&a, &a);
    printf("Self overlap is: %lf\n", self_overlap);

    printf("Normalization in each axis is: %lf, %lf, %lf\n", 
        a.normalization_constants[0],
        a.normalization_constants[1],
        a.normalization_constants[2]
    );


}