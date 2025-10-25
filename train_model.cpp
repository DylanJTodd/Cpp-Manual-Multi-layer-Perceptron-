#include "utils.hpp"
#include "mlp.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>

struct ModelMetadata {
    std::vector<int> architecture;
    float learning_rate = 0.0f;
    float final_accuracy = 0.0f;
    int total_epochs = 0;
    std::string timestamp;
};

std::string get_timestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

void write_float_array(std::ofstream& out, const std::vector<float>& arr, int indent = 0) {
    std::string indent_str(indent, ' ');
    out << indent_str << "[\n";

    for (size_t i = 0; i < arr.size(); ++i) {
        if (i % 10 == 0) out << indent_str << "  ";
        out << std::scientific << std::setprecision(8) << arr[i];
        if (i < arr.size() - 1) out << ", ";
        if ((i + 1) % 10 == 0 && i < arr.size() - 1) out << "\n";
    }

    out << "\n" << indent_str << "]";
}

bool save_model_json(MLP& model, const ModelMetadata& metadata, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return false;
    }

    std::vector<std::vector<float>> weights, biases;
    model.get_weights_copy(weights, biases);

    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"timestamp\": \"" << metadata.timestamp << "\",\n";
    out << "    \"learning_rate\": " << metadata.learning_rate << ",\n";
    out << "    \"total_epochs\": " << metadata.total_epochs << ",\n";
    out << "    \"final_accuracy\": " << metadata.final_accuracy << ",\n";
    out << "    \"architecture\": [";
    for (size_t i = 0; i < metadata.architecture.size(); ++i) {
        out << metadata.architecture[i];
        if (i < metadata.architecture.size() - 1) out << ", ";
    }
    out << "]\n";
    out << "  },\n";

    out << "  \"layers\": [\n";
    for (size_t layer = 1; layer < weights.size(); ++layer) {
        out << "    {\n";
        out << "      \"layer_index\": " << layer << ",\n";
        out << "      \"input_dim\": " << metadata.architecture[layer - 1] << ",\n";
        out << "      \"output_dim\": " << metadata.architecture[layer] << ",\n";

        out << "      \"weights\": ";
        write_float_array(out, weights[layer], 6);
        out << ",\n";

        out << "      \"biases\": ";
        write_float_array(out, biases[layer], 6);
        out << "\n";

        out << "    }";
        if (layer < weights.size() - 1) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    out.close();
    std::cout << "Model saved to " << filename << std::endl;
    return true;
}

bool load_model_json(MLP& model, ModelMetadata& metadata, const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << filename << " for reading" << std::endl;
        return false;
    }

    std::string line;
    std::vector<std::vector<float>> weights, biases;

    bool reading_metadata = false;
    bool reading_layers = false;
    int current_layer = 0;
    std::vector<float> current_weights, current_biases;
    bool reading_weights = false;
    bool reading_biases = false;

    while (std::getline(in, line)) {
        size_t pos = line.find("\"timestamp\":");
        if (pos != std::string::npos) {
            size_t start = line.find("\"", pos + 12) + 1;
            size_t end = line.find("\"", start);
            metadata.timestamp = line.substr(start, end - start);
            continue;
        }

        pos = line.find("\"learning_rate\":");
        if (pos != std::string::npos) {
            size_t start = line.find(":", pos) + 1;
            size_t end = line.find(",", start);
            if (end == std::string::npos) end = line.length();
            metadata.learning_rate = std::stof(line.substr(start, end - start));
            continue;
        }

        pos = line.find("\"total_epochs\":");
        if (pos != std::string::npos) {
            size_t start = line.find(":", pos) + 1;
            size_t end = line.find(",", start);
            if (end == std::string::npos) end = line.length();
            metadata.total_epochs = std::stoi(line.substr(start, end - start));
            continue;
        }

        pos = line.find("\"final_accuracy\":");
        if (pos != std::string::npos) {
            size_t start = line.find(":", pos) + 1;
            size_t end = line.find(",", start);
            if (end == std::string::npos) end = line.length();
            metadata.final_accuracy = std::stof(line.substr(start, end - start));
            continue;
        }

        pos = line.find("\"architecture\":");
        if (pos != std::string::npos) {
            size_t start = line.find("[", pos) + 1;
            size_t end = line.find("]", start);
            std::string arch_str = line.substr(start, end - start);
            std::istringstream iss(arch_str);
            int val;
            char comma;
            metadata.architecture.clear();
            while (iss >> val) {
                metadata.architecture.push_back(val);
                iss >> comma;
            }
            continue;
        }

        pos = line.find("\"weights\":");
        if (pos != std::string::npos) {
            reading_weights = true;
            reading_biases = false;
            current_weights.clear();
            continue;
        }

        pos = line.find("\"biases\":");
        if (pos != std::string::npos) {
            reading_biases = true;
            reading_weights = false;
            current_biases.clear();
            continue;
        }

        if (reading_weights || reading_biases) {
            std::istringstream iss(line);
            float val;
            char ch;
            while (iss >> ch) {
                if (ch == ']') {
                    if (reading_weights) {
                        weights.push_back(current_weights);
                        reading_weights = false;
                    }
                    else if (reading_biases) {
                        biases.push_back(current_biases);
                        reading_biases = false;
                    }
                    break;
                }
                else if (std::isdigit(ch) || ch == '-' || ch == '+' || ch == 'e' || ch == 'E') {
                    iss.putback(ch);
                    if (iss >> val) {
                        if (reading_weights) {
                            current_weights.push_back(val);
                        }
                        else if (reading_biases) {
                            current_biases.push_back(val);
                        }
                    }
                }
            }
        }
    }

    in.close();

    weights.insert(weights.begin(), std::vector<float>());
    biases.insert(biases.begin(), std::vector<float>());

    std::vector<std::vector<float>> temp_weights, temp_biases;
    model.get_weights_copy(temp_weights, temp_biases);

    for (size_t layer = 1; layer < weights.size(); ++layer) {
        if (layer < temp_weights.size() && weights[layer].size() == temp_weights[layer].size()) {
            temp_weights[layer] = weights[layer];
        }
        if (layer < temp_biases.size() && biases[layer].size() == temp_biases[layer].size()) {
            temp_biases[layer] = biases[layer];
        }
    }

    std::cout << "Model loaded from " << filename << std::endl;
    std::cout << "  Architecture: ";
    for (size_t i = 0; i < metadata.architecture.size(); ++i) {
        std::cout << metadata.architecture[i];
        if (i < metadata.architecture.size() - 1) std::cout << " -> ";
    }
    std::cout << "\n  Accuracy: " << (metadata.final_accuracy * 100.0f) << "%" << std::endl;

    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   OPTIMAL MODEL TRAINER" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "Loading MNIST data..." << std::endl;
    MNISTData mnist = load_mnist("mnist_data", 60000, 10000);

    std::vector<int> optimal_architecture = { 784, 512, 256, 128, 10 };
    float optimal_learning_rate = 0.001f;
    int optimal_epochs = 15;
    size_t optimal_batch_size = 64;

    std::cout << "\nOptimal Configuration:" << std::endl;
    std::cout << "  Architecture: ";
    for (size_t i = 0; i < optimal_architecture.size(); ++i) {
        std::cout << optimal_architecture[i];
        if (i < optimal_architecture.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
    std::cout << "  Learning Rate: " << optimal_learning_rate << std::endl;
    std::cout << "  Epochs: " << optimal_epochs << std::endl;
    std::cout << "  Batch Size: " << optimal_batch_size << std::endl;

    std::cout << "\nBuilding model..." << std::endl;
    MLP model(optimal_architecture, optimal_learning_rate);

    std::cout << "Starting training...\n" << std::endl;
    auto training_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= optimal_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        model.train_epoch(mnist.train_images, mnist.train_labels, optimal_batch_size);

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);

        float train_accuracy = model.evaluate(mnist.train_images, mnist.train_labels, 10000);
        float test_accuracy = model.evaluate(mnist.test_images, mnist.test_labels, mnist.n_test);

        std::cout << "Epoch " << std::setw(2) << epoch << "/" << optimal_epochs
            << " | Train: " << std::fixed << std::setprecision(2) << (train_accuracy * 100.0f) << "% "
            << "| Test: " << (test_accuracy * 100.0f) << "% "
            << "| Time: " << epoch_duration.count() << "s" << std::endl;
    }

    auto training_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(training_end - training_start);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Training complete in " << total_duration.count() << " seconds!" << std::endl;

    float final_accuracy = model.evaluate(mnist.test_images, mnist.test_labels, mnist.n_test);
    std::cout << "Final Test Accuracy: " << (final_accuracy * 100.0f) << "%" << std::endl;
    std::cout << "========================================\n" << std::endl;

    ModelMetadata metadata;
    metadata.architecture = optimal_architecture;
    metadata.learning_rate = optimal_learning_rate;
    metadata.final_accuracy = final_accuracy;
    metadata.total_epochs = optimal_epochs;
    metadata.timestamp = get_timestamp();

    std::string model_filename = "optimal_model.json";
    if (save_model_json(model, metadata, model_filename)) {
        std::cout << "\nModel successfully saved!" << std::endl;
        std::cout << "File: " << model_filename << std::endl;
        std::cout << "Size: " << optimal_architecture.size() << " layers" << std::endl;
    }

    std::cout << "\nYou can now load this model in the main application." << std::endl;

    return 0;
}