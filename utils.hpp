#pragma once
#include <vector>
#include <string>
#include <cstddef>

struct MNISTData {
    std::vector<float> train_images; // flattened: n_train * 784
    std::vector<int>   train_labels;
    std::vector<float> test_images;  // flattened: n_test * 784
    std::vector<int>   test_labels;
    size_t n_train = 0;
    size_t n_test = 0;
};

// path should be the folder containing the 4 idx files: train-images.idx3-ubyte, train-labels.idx1-ubyte, ...
// limit_train / limit_test can be used to load fewer examples for quick debug
MNISTData load_mnist(const std::string& path, size_t limit_train = 60000, size_t limit_test = 10000);
