#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

void read_file(char *filename, float* objects, int numObjs, int numCoords)
{
    std::ifstream input(filename);
    int id;
    int i = 0;
    int j = 0;
    while(input >> id)
    {
        while(j < numCoords)
        {
            input>> objects[i * numCoords + j];
            j++;
        }
        j = 0;
        i++;
    }
    input.close();
}

int main(int argc, char **argv)
{
    /*int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    float *numbers = new float[n * k];
    read_file(argv[3],numbers, n, k);
    for(int i = 0; i < n; i++)
    {
        for(int  j = 0; j < k; j++)
        {
            std::cout<<numbers[i * k + j]<< ' ';
        }
        std::cout<< std::endl;
    }
    free(numbers);*/
    int *memb = new int[100];
    memset(memb, -1, sizeof(int) * 100);
    std::cout << memb[0] << std::endl;
}