/* ------------------------------------------------------------------------------------------------

    USAGE

    ./a.out arg1 arg2 arg3 arg4
        arg1: enter pswd length.
        arg2: 0 or 1. 0 = choose a preset, 1 = enter own valid characters
            if chosen preset, enter 0-3 as next argument.
            if chosen valid characters, enter number of valid characters 
                and the characters as next 2 arguments.
        arg3: 0 or 1. 0 = generate random pswd to find, 1 = enter own pswd
            if chosen to enter a password, enter it here as 1 string.
        arg4: 0 or 1. 0 = do not print generated pswds, 1 = print
            (printing not recommended on large amounts of pswds)
    
    example usages:
    ./a.out 6 0 3 1 abcdef 0
        (length 6, base64 preset, find pswd abcdef, do not print)
    ./a.out 10 1 4 abcd 0 1 
        (length 10, characters {a, b, c, d}, find random pswd, print)

------------------------------------------------------------------------------------------------ */

#include <math.h> // for pow()
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> // for timing the program later

#include "kernel.cu"

// generates a RANDOM password
void generatePassword(char* password, char* characterList, int length) {
    
    int numChar = strlen(characterList);
    
    srand(time(0));

    int i;
    for (i = 0; i < length; ++i) {
        // Get a random index from the characters string
        int randomIndex = rand() % numChar;
        // Append the character at the random index to the password
        password[i] = characterList[randomIndex];
    }
    password[i] = '\0'; // add null terminator
}

int main(int argc, char *argv[])
{
    char* validChars = (char*)malloc(256);
    char* presetPassword = (char*)malloc(256); // the password to check against

    int arg = 1;

    int pswdLen = atoi(argv[arg]); arg++;
    int runType = atoi(argv[arg]); arg++;
    int numValidChars;

    switch(runType)
    {
        case 0: // pick preset
        {
            int charSet = atoi(argv[arg]); arg++;

            switch(charSet)
            {
                case 0: // 0-9
                {
                    numValidChars = 10;
                    const char* str = "0123456789";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }
                case 1: // lowercase letters
                {
                    numValidChars = 26;
                    const char* str = "abcdefghijklmnopqrstuvwxyz";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }
                case 2: // base32: RFC 4648 standard (https://en.wikipedia.org/wiki/Base32)
                {
                    numValidChars = 32;
                    const char* str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }
                case 3: // base64: RFC 4648 standard (https://en.wikipedia.org/wiki/Base64)
                {
                    numValidChars = 64;
                    const char* str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }
                default:
                {
                    printf("unrecognized argument at %d: pick a character preset: 0 = decimal numbers, 1 = lowercase letters, 2 = base32, 3 = base64\n", arg);
                    exit(-1);
                    break;
                }
            }
            break;
        }
        case 1: // entered own characters
        {
            numValidChars = atoi(argv[arg]); arg++;
            const char* str = argv[arg]; arg++;
            for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }

            break;
        }
        default:
        {
            printf("unrecognized argument at %d: pick a preset (0) or enter your own (1)\n", arg);
            exit(-1);
            break;
        }
    }

    int hasGivenPassword = atoi(argv[arg]); arg++;
    
    switch(hasGivenPassword)
    {
        case 0: // generate a random password
        {
            generatePassword(presetPassword, validChars, pswdLen);
            break;
        }
        case 1: // get the password from arguments
        {
            const char* argPswd = argv[arg]; arg++;
            int presetPswdLen = strlen(argPswd);

            for(int i = 0; i < presetPswdLen; i++) { presetPassword[i] = argPswd[i]; }
            break;
        }
        default:
        {
            printf("unrecognized argument at %d: generate a random password (0) or enter your own (1)\n", arg);
            exit(-1);
            break;
        }
    }

    int printPswds = atoi(argv[arg]); arg++; // if it's 0, print, if it's 1...
    bool* d_printPswds; cudaMalloc(&d_printPswds, sizeof(bool));

    switch(printPswds)
    {
        case 0: { cudaMemset(d_printPswds, false, sizeof(bool)); break;}
        case 1: { cudaMemset(d_printPswds, true, sizeof(bool));  break;}
        default:
        {
            printf("unrecognized argument at %d: don't print passwords (0) or print passwords (1)\n", arg);
            exit(-1);
            break;
        }
    }


    // cool, free to do other stuff now
    bool h_isDone = false;
    bool* d_isDone; cudaMalloc(&d_isDone, sizeof(bool)); cudaMemset(d_isDone, false, sizeof(bool));

    int iterator = 1;

    // while(!h_isDone && iterator <= pswdLen)
    while(!h_isDone && iterator <= 1)
    {
        printf("launching kernel w/ pswdLen of %d\n", iterator);

        crack(validChars, numValidChars, pswdLen, presetPassword, printPswds, d_isDone, d_printPswds);

        cudaError_t cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) printf("Unable to launch kernel\n");

        cudaMemcpy(&h_isDone, d_isDone, sizeof(bool), cudaMemcpyDeviceToHost);

        iterator++;        
    }

    cudaFree(d_isDone);
    cudaFree(d_printPswds);

    return 0;
}