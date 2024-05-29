/* --------------------------------------------------------

    USAGE

    ./bruteforce 0 validCharString numValidChars passwordLength
    ./bruteforce 1 0-3 passwordLength
    
    1st version: input your own characters (eg: abcd)
    2nd version: pick a preset of characters (0-3)

--------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <time.h> // for timing the program later

int main(int argc, char *argv[])
{
    int runType = atoi(argv[1]);
    char* validChars;
    int numValidChars, pswdLen;

    // get valid characters + pswdLen
    switch(runType)
    {
        case 0:
        {
            // custom character set
            const char* str = argv[2];
            numValidChars = atoi(argv[3]);
            for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
            pswdLen = atoi(argv[4]);
            break;
        }
        case 1:
        {
            int charSet = atoi(argv[2]);
            pswdLen = atoi(argv[3]);
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

    // cool, free to do other stuff now

    return 0;
}