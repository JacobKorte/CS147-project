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
__device__ void incrementIndex(int* index, int base) {
    int overflow = 1;
    for (int i = base - 1; i >= 0; --i){
        index[i] += overflow;
        if (index[i] >= base){
            index[i] = 0;
            overflow = 1;
        } 
        
        else {
            overflow = 0;
            break;
        }
    }
}

__device__ void createPasswordFromIndex(const int* index, const char* characterSet, int pswdLen, char* password) {
    for (int i = 0; i < pswdLen; ++i) {
        password[i] = characterSet[index[i]];
    }
}