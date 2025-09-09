#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int status;

    printf("Running test_cartesian...\n");
    status = system("./test_cartesian_gaussian.exe");
    if (status != 0) {
        printf("!!! FAILED TEST cartesian_gaussian_test\n");
        return 1;
    }

    printf("Running test_special_functions...\n");
    status = system("./test_special_functions.exe");
    if (status != 0) {
        printf("!!! FAILED TEST test_special_functions\n");
        return 1;
    }

    printf("All tests passed!\n");
    return 0;
}