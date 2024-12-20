#include "Comparator.hh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <algorithm>

namespace {
    const char* RED = "\033[1;31m";
    const char* YELLOW = "\033[1;33m";
    const char* CYAN = "\033[1;36m";
    const char* RESET = "\033[0m";
}

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <trace1> [trace2]\n";
        return 1;
    }

    if (argc == 2) {
        Tracer tracer1(argv[1]);
        tracer1.trace_ << std::cout;
    } else {  // argc == 3
        Comparator comparator(argv[1], argv[2]);
        std::cout << comparator;
    }
    return 0;
}