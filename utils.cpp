#include "utils.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cstdint>

static uint32_t read_be_uint32(std::ifstream& f) {
    uint32_t x = 0;
    f.read(reinterpret_cast<char*>(&x), sizeof(x));
    // file is big-endian
    uint32_t b0 = (x & 0x000000FFu);
    uint32_t b1 = (x & 0x0000FF00u) >> 8;
    uint32_t b2 = (x & 0x00FF0000u) >> 16;
    uint32_t b3 = (x & 0xFF000000u) >> 24;
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
}

static std::vector<float> load_images_flat(const std::string& filename, size_t limit) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Failed to open " + filename);

    uint32_t magic = read_be_uint32(f);
    uint32_t n_imgs = read_be_uint32(f);
    uint32_t n_rows = read_be_uint32(f);
    uint32_t n_cols = read_be_uint32(f);

    if (magic != 2051) throw std::runtime_error("Invalid image file magic: " + filename);

    size_t img_size = (size_t)n_rows * (size_t)n_cols;
    size_t n = std::min((size_t)n_imgs, limit);
    std::vector<float> flat;
    flat.reserve(n * img_size);

    std::vector<unsigned char> buffer(img_size);
    for (size_t i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(buffer.data()), img_size);
        for (size_t j = 0; j < img_size; ++j) {
            flat.push_back(static_cast<float>(buffer[j]) / 255.0f);
        }
    }
    return flat;
}

static std::vector<int> load_labels(const std::string& filename, size_t limit) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Failed to open " + filename);

    uint32_t magic = read_be_uint32(f);
    uint32_t n_items = read_be_uint32(f);
    if (magic != 2049) throw std::runtime_error("Invalid label file magic: " + filename);

    size_t n = std::min((size_t)n_items, limit);
    std::vector<int> labels;
    labels.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        unsigned char v = 0;
        f.read(reinterpret_cast<char*>(&v), 1);
        labels.push_back((int)v);
    }
    return labels;
}

MNISTData load_mnist(const std::string& path, size_t limit_train, size_t limit_test) {
    MNISTData d;
    std::string timgs = path + "/train-images.idx3-ubyte";
    std::string tlabels = path + "/train-labels.idx1-ubyte";
    std::string kimgs = path + "/t10k-images.idx3-ubyte";
    std::string klabels = path + "/t10k-labels.idx1-ubyte";

    d.train_images = load_images_flat(timgs, limit_train);
    d.train_labels = load_labels(tlabels, limit_train);
    d.test_images = load_images_flat(kimgs, limit_test);
    d.test_labels = load_labels(klabels, limit_test);

    d.n_train = d.train_labels.size();
    d.n_test = d.test_labels.size();

    std::cout << "Loaded MNIST: train=" << d.n_train << " test=" << d.n_test << "\n";
    return d;
}
