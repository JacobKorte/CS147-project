#include <stdio.h>

#define BLOCK_SIZE 1024
#define GRID_SIZE 36

// terrible implementation. 4 parameters, really-
__device__ void numToIndex(int num, int numChars, int pswdLen, int* arr) {
    int iterator = pswdLen - 1;

    while(iterator >= 0)
    {
        int val = num % numChars;
        arr[iterator] = val;

        num /= numChars;
        iterator--;
    }
}

__global__ void crack_kernel(char* validChars, int numValidChars, int pswdLen) {
    int startingNum, endingNum;
    startingNum = ;

    int* startingIndex, endingIndex;
    numToIndex(startingNum, numValidChars, pswdLen, startingIndex);
    numToIndex(endingNum, numValidChars, pswdLen, endingIndex);
    
    while(false) { // replace condition later
        // create password
        // update index
    }
}

void crack(char* validChars, int numValidChars, int pswdLen) {
    dim3 blockSize (1024, 1, 1);
    dim3 gridSize (36, 1, 1);

    crack_kernel<<<gridSize, blockSize>>>(validChars, numValidChars, pswdLen);
}