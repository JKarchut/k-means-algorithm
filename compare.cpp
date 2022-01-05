#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

std::pair<int, int> create_pair(int a, int b) {
    if (a <= b) return std::make_pair(a, b);
    else return std::make_pair(b, a);
}

int parse_input(char * file, std::vector<std::pair<int, int> >& data_vector) {
    int a, b;
    std::ifstream input(file);

    while(input >> a) {
        if (input.eof())
            return -1; 
        input >> b;

        data_vector.push_back(create_pair(a, b));
    }
    sort(data_vector.begin(), data_vector.end());
    return 0;
}

void check_if_equal(std::vector<std::pair<int, int> >& correct_output, std::vector<std::pair<int, int> >& solution_output) {
    if ((int)correct_output.size() != (int)solution_output.size()) {
        std::cout << "Invalid Solution" << std::endl;
        return;
    }
    size_t vectors_size = (int)correct_output.size();
    for (size_t i = 0; i < vectors_size; i++) {
        if (correct_output[i].first != solution_output[i].first || correct_output[i].second != solution_output[i].second) {
            std::cout << "Invalid Solution" << std::endl;
            return;
        }
    }
    std::cout << "OK" << std::endl;
}

int main(int argc, char ** argv) {
    if (argc < 3)
        return -1;
    std::vector<std::pair<int, int> > output1, output2;

    if(-1 == parse_input(argv[1], output1)) return -1;
    if(-1 == parse_input(argv[2], output2)) return -1;
    check_if_equal(output1, output2);
    return 0;
}