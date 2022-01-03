#include <iostream>
#include <fstream>
#include <random>

int main(int argc, char** argv)
{
    int N = atoi(argv[1]);
    int n = atoi(argv[2]);
    std::ofstream out(argv[3]);
    std::random_device rd;
    std::default_random_engine generator(rd()); // rd() provides a random seed
    std::uniform_real_distribution<double> distribution(-10,10);
    for(int i = 0; i < N; i++)
    {
        out << i;
        for(int j = 0; j < n; j++)
        {
            out << ' ' << distribution(generator);
        }
        out << '\n';
    }
    out.close();
}