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
    int numThreads = BLOCK_SIZE * GRID_SIZE;
    unsigned long long int totalPswds = pow(numValidChars, pswdLen);
    int workPerThread = totalPswds / numThreads; // every thread should do AT LEAST this amt of work
    int overhead = totalPswds % numThreads; // remaining amt of pswds that dont divide evenly

    // calculate starting and ending numbers, convert to workable indexes
    unsigned long long int startingNum, endingNum;
    startingNum = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * workPerThread;
    endingNum = startingNum + workPerThread - 1;

    int* startingIndex, endingIndex;
    numToIndex(startingNum, numValidChars, pswdLen, startingIndex);
    numToIndex(endingNum, numValidChars, pswdLen, endingIndex);
    
    // get to work
    while(false) { // replace condition later
        // create password
        // update index
    }

    // handle remainder passwords
    if(blockIdx.x * BLOCK_SIZE + threadIdx.x < overhead)
    {
        // gen index
        // create password
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