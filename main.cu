/* ------------------------------------------------------------------------------------------------

    USAGE

    ./bruteforce 0 0 validCharString numValidChars  passwordLength
    ./bruteforce 0 1 validCharString numValidChars  passwordLength password
    ./bruteforce 1 0 0-3             passwordLength
    ./bruteforce 1 1 0-3             passwordLength password

    1st version: input your own characters (eg: abcd)
    2nd version: ditto 1st, enter the password to find
    3nd version: pick a preset of characters (0-3)
        0: numbers
        1: lowercase letters
        2: base32
        3: base64
    4th version: ditto 3rd, enter the password to find

------------------------------------------------------------------------------------------------ */

#include <stdio.h>
#include <stdlib.h>
#include <time.h> // for timing the program later
#include <string.h>

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
    int runType = atoi(argv[1]);
    int hasGivenPassword = atoi(argv[2]);
    char* validChars = (char*)malloc(256);
    char* presetPassword = (char*)malloc(256); // the password to check against
    int numValidChars, pswdLen;

    // get valid characters + pswdLen
    switch(runType)
    {
        case 0:
        {
            // custom character set
            const char* str = argv[3];
            numValidChars = atoi(argv[4]);
            for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
            pswdLen = atoi(argv[5]);
            break;
        }
        case 1:
        {
            int charSet = atoi(argv[3]);
            pswdLen = atoi(argv[4]);
            switch(charSet)
            {
                case 0:
                {
                    // numbers 0-9
                    numValidChars = 10;
                    const char* str = "0123456789";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }
                case 1:
                {
                    // lowercase letters
                    numValidChars = 26;
                    const char* str = "abcdefghijklmnopqrstuvwxyz";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }
                case 2:
                {
                    // base32: RFC 4648 standard (https://en.wikipedia.org/wiki/Base32)
                    numValidChars = 32;
                    const char* str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }
                case 3:
                {
                    // base64: RFC 4648 standard (https://en.wikipedia.org/wiki/Base64)
                    numValidChars = 64;
                    const char* str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }
                default:
                {
                    printf("unrecognized argument: character preset should be a number 0-3");
                    exit(-1);
                    break;
                }
            }
            break;
        }
        default:
        {
            printf("unrecognized argument: run type should be a number 0-1");
            exit(-1);
            break;
        }
    }

    switch(hasGivenPassword)
    {
        case 0:
        {
            // generate a random password
            generatePassword(presetPassword, validChars, pswdLen);
            break;
        }
        case 1:
        {
            // get the password from arguments (idx 6)
            const char* argPswd = argv[6];
            for(int i = 0; i < pswdLen; i++) { presetPassword[i] = argPswd[i]; }
            break;
        }
        default:
        {
            printf("unrecognized argument: has preset password should be a number 0-1");
            exit(-1);
            break;
        }
    }

    // cool, free to do other stuff now

    return 0;
}