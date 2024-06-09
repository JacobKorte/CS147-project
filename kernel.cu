#include <stdio.h>

#define BLOCK_SIZE 1024
#define GRID_SIZE 36
__device__ bool isDone;

                        // number to    number of     length of    the array
                        // create       valid         password
                        // index from   chars         (# indexes)
__device__ void numToIndex(int num, int numChars, const int pswdLen, int* arr) {
    int iterator = pswdLen - 1;

    while(iterator >= 0)
    {
        int val = num % numChars;
        arr[iterator] = val;

        num /= numChars;
        iterator--;
    }
}

__device__ bool checkPassword(char* c1, char* c2, const int pswdLen) 
{
    for(int i = 0; i < pswdLen; i++)  
    {
        if(c1[i] != c2[i]) 
        {
            return false;
        }
    }
    return true;
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

                            //total number of passwords          // string of valid     number of valid    length of      password to
                            //uniigned long long                 // chars               chars              password       check against
__global__ void crack_kernel(unsigned long long int totalPswds, char* validChars, int numValidChars, const int pswdLen, char* password) {
    int numThreads = BLOCK_SIZE * GRID_SIZE;
    //unsigned long long int totalPswds = pow(numValidChars, pswdLen); doesnt like this for some reason
    int workPerThread = totalPswds / numThreads; // every thread should do AT LEAST this amt of work
    int overhead = totalPswds % numThreads; // remaining amt of pswds that dont divide evenly

    // calculate starting and ending numbers, convert to workable indexes
    unsigned long long int startingNum, endingNum;
    startingNum = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * workPerThread;
    endingNum = startingNum + workPerThread - 1;

    int* startingIndex = (int*)malloc(pswdLen * sizeof(int));
    int* endingIndex = (int*)malloc(pswdLen * sizeof(int));
    numToIndex(startingNum, numValidChars, pswdLen, startingIndex);
    numToIndex(endingNum, numValidChars, pswdLen, endingIndex);

    // where the created password is held
    char* p = (char*)malloc((pswdLen + 1) * sizeof(char));
    p[pswdLen] = '\0';
    
    // get to work
    while(startingNum < endingNum && !isDone) { // replace condition later
        // create password
        createPasswordFromIndex(startingIndex, validChars, pswdLen, p);
        //check if match

        if(checkPassword(p, password, pswdLen))
            isDone = true;

        // update index
        startingNum++;
        incrementIndex(startingIndex, numValidChars);
    }

    // handle remainder passwords
    if(!isDone && (blockIdx.x * BLOCK_SIZE + threadIdx.x < overhead))
    {
        // gen index
        int passwordNum = GRID_SIZE * BLOCK_SIZE * workPerThread + (blockIdx.x * BLOCK_SIZE + threadIdx.x);
        int* index;
        numToIndex(passwordNum, numValidChars, pswdLen, startingIndex);

        // create password
        createPasswordFromIndex(index, validChars, pswdLen, p);

        if(checkPassword(p, password, pswdLen))
            isDone = true;
        
    }
}

void crack(char* validChars, int numValidChars, const int pswdLen, char* password) {
    dim3 blockSize (1024, 1, 1);
    dim3 gridSize (36, 1, 1);
    //have to do it this way because pow doesnt work on GPU
    unsigned long long int totalPswds = numValidChars; 
    for(int i = 1; i < pswdLen; i++) 
    {
        totalPswds *= numValidChars; 
    }
    int iterator = pswdLen;
    while(!isDone && iterator > 0)
    {
        // generate all the passwords of length pswdLen or less!
        crack_kernel<<<gridSize, blockSize>>>(totalPswds, validChars, numValidChars, iterator, password);
        iterator--;
    }
}

