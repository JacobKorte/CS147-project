#include <math.h> // for pow()
#include <stdio.h>
#include <stdlib.h> // for free()

#define BLOCK_SIZE 1024
#define GRID_SIZE 36

                        // number to            number of     length of    the array
                        // create               valid         password
                        // index from           chars         (# indexes)
__device__ void numToIndex(unsigned long long int num, int numChars, const int pswdLen, int* arr) { // int[]
    int iterator = pswdLen - 1;

    while(iterator >= 0)
    {
        unsigned long long int val = num % numChars;
        arr[iterator] = val;

        num /= numChars;
        iterator--;
    }
}

__device__ bool checkPassword(char* c1, char* c2, const int pswdLen) 
{
    // doesnt work since strlen is a host() function... great
    // int c1Len = strlen(c1);
    // int c2Len = strlen(c2);

    // get lengths of c1 and c2
    int c1Len = 0;
    int c2Len = 0;
    for(int i = 0; c1[i] != '\0'; i++) { c1Len++; }
    for(int i = 0; c2[i] != '\0'; i++) { c2Len++; }

    if(c1Len != c2Len) return false;

    for(int i = 0; i < pswdLen; i++)  
    {
        if(c1[i] != c2[i]) 
        {
            return false;
        }
    }

    return true;
}

__device__ void incrementIndex(int* index, int pswdLen, int base) {
    for(int i = pswdLen - 1; i >= 0; i--)
    {
        if(index[i] == (base - 1))
        {
            index[i] = 0; // overflowing
        }
        else
        {
            index[i] += 1; return; // not overflowing
        }
    }
}

__device__ void createPasswordFromIndex(const int* index, const char* characterSet, int pswdLen, char* password) { // char[]
    int i = 0;
    
    for (i = i; i < pswdLen; i++) {
        password[i] = characterSet[index[i]];
    }
    password[i] = '\0';
}


                            //total number of passwords          // string of valid     number of valid    length of      password to
                            //unsigned long long                 // chars               chars              password       check against
__global__ void crack_kernel(unsigned long long int totalPswds, char* validChars, int numValidChars, const int pswdLen, char* password, bool* doneness, bool* printPasswords) {
    bool localPrintPasswords = *printPasswords; // put from global into a closer memory space
    int numThreads = BLOCK_SIZE * GRID_SIZE;
    int workPerThread = totalPswds / numThreads; // every thread should do AT LEAST this amt of work
    int overhead = totalPswds % numThreads; // remaining amt of pswds that dont divide evenly

    // calculate starting and ending numbers, convert to workable indexes
    unsigned long long int startingNum, endingNum;
    startingNum = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * workPerThread;
    if(workPerThread == 0) { endingNum = 0; }
    else { endingNum = startingNum + workPerThread - 1; }

    int startingIndex[64];
    int endingIndex[64];

    // where the created password is held
    char p[64];

    if(workPerThread > 0) // can skip if workPerThreads is 0 (ie: less pswds than threads working)
    {
        numToIndex(startingNum, numValidChars, pswdLen, startingIndex);
        numToIndex(endingNum, numValidChars, pswdLen, endingIndex);
        
        // get to work
        while(startingNum < endingNum && !(*doneness)) { // replace condition later
            // create password
            createPasswordFromIndex(startingIndex, validChars, pswdLen, p);
            if(localPrintPasswords) printf("%s", p);

            //check if match
            if(checkPassword(p, password, pswdLen)) {
                *doneness = true;
            }

            // update index
            startingNum++;
            incrementIndex(startingIndex, pswdLen, numValidChars);
        }
    }

    // handle remainder passwords -----------------------------------------------------------------
    if(!(*doneness) && (blockIdx.x * BLOCK_SIZE + threadIdx.x < overhead))
    {
        // gen index
        int passwordNum = GRID_SIZE * BLOCK_SIZE * workPerThread + (blockIdx.x * BLOCK_SIZE + threadIdx.x);
        int index[64];

        numToIndex(passwordNum, numValidChars, pswdLen, index);

        // create password
        createPasswordFromIndex(index, validChars, pswdLen, p);

        if(localPrintPasswords) printf("%s\n", p);

        if(checkPassword(p, password, pswdLen))
        {
            *doneness = true;
        }
    }
}

void crack(char* validChars, int numValidChars, const int pswdLen, char* password, int printPasswords, bool* device_isDone, bool* device_printPasswords) {
    dim3 blockSize (1024, 1, 1);
    dim3 gridSize (36, 1, 1);
    unsigned long long int totalPswds = pow(numValidChars, pswdLen);

    crack_kernel<<<gridSize, blockSize>>>(totalPswds, validChars, numValidChars, pswdLen, password, device_isDone, device_printPasswords);
}