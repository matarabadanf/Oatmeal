#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int status;

    printf("Running test_cartesian...\n");
    status = system("./test_cartesian_gaussian.exe");
    if (status != 0) {
        printf("test_cartesian FAILED\n");
        return 1;
    }

    printf("All tests passed!\n");
    return 0;
}