#include "utils.hpp"
#include "mlp.hpp"
#include "visualization.hpp"
#include <thread>
#include <chrono>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>

bool load_model_from_json(MLP& model, const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    std::string line;
    std::vector<std::vector<float>> weights, biases;
    std::vector<int> architecture;
    float final_accuracy = 0.0f;

    std::vector<float> current_weights, current_biases;
    bool reading_weights = false;
    bool reading_biases = false;

    while (std::getline(in, line)) {
        size_t pos = line.find("\"final_accuracy\":");
        if (pos != std::string::npos) {
            size_t start = line.find(":", pos) + 1;
            size_t end = line.find(",", start);
            if (end == std::string::npos) end = line.length();
            final_accuracy = std::stof(line.substr(start, end - start));
        }

        pos = line.find("\"architecture\":");
        if (pos != std::string::npos) {
            size_t start = line.find("[", pos) + 1;
            size_t end = line.find("]", start);
            std::string arch_str = line.substr(start, end - start);
            std::istringstream iss(arch_str);
            int val;
            char comma;
            while (iss >> val) {
                architecture.push_back(val);
                iss >> comma;
            }
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
                else if (std::isdigit(ch) || ch == '-' || ch == '+' || ch == 'e' || ch == 'E' || ch == '.') {
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

    model.set_weights(weights, biases);

    std::cout << "Model loaded from " << filename << std::endl;
    std::cout << "  Architecture: ";
    for (size_t i = 0; i < architecture.size(); ++i) {
        std::cout << architecture[i];
        if (i < architecture.size() - 1) std::cout << " -> ";
    }
    std::cout << "\n  Pre-trained Accuracy: " << (final_accuracy * 100.0f) << "%" << std::endl;

    return true;
}

int main() {
    std::cout << "Loading MNIST data..." << std::endl;
    MNISTData mnist = load_mnist("mnist_data", 60000, 10000);

    Visualizer viz(1400, 800);
    viz.set_mode(VisualizationMode::Configuration);
    viz.set_fps_limit(60);
    viz.start();

    std::cout << "Configure your network in the visualization window..." << std::endl;
    while (!viz.is_config_complete()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    TrainingConfig config = viz.get_config();
    std::vector<int> architecture = config.get_architecture();
    bool load_default = viz.should_load_default();

    std::cout << "\nConfiguration:" << std::endl;

    if (load_default) {
        std::cout << "  Mode: Loading pre-trained model" << std::endl;
        architecture = { 784, 512, 256, 128, 10 };
    }
    else {
        std::cout << "  Mode: Training new model" << std::endl;
        std::cout << "  Architecture: ";
        for (size_t i = 0; i < architecture.size(); ++i) {
            std::cout << architecture[i];
            if (i < architecture.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
        std::cout << "  Learning Rate: " << config.learning_rate << std::endl;
        std::cout << "  Epochs: " << config.epochs << std::endl;
        std::cout << "  Batch Size: " << config.batch_size << std::endl;
    }

    std::cout << "\nBuilding model..." << std::endl;
    MLP model(architecture, load_default ? 0.001f : config.learning_rate);

    if (load_default) {
        std::cout << "Loading pre-trained model..." << std::endl;
        if (!load_model_from_json(model, "optimal_model.json")) {
            std::cerr << "Failed to load default model. Please run train_optimal_model first." << std::endl;
            return 1;
        }

        std::cout << "\nEvaluating loaded model..." << std::endl;
        float loaded_accuracy = model.evaluate(mnist.test_images, mnist.test_labels, mnist.n_test);
        std::cout << "Loaded model accuracy: " << (loaded_accuracy * 100.0f) << "%" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(2));

    }
    else {
        std::cout << "Starting training..." << std::endl;
        viz.set_mode(VisualizationMode::Training);
        viz.set_fps_limit(10);

        size_t num_training_samples = mnist.n_train;
        size_t total_batches = (num_training_samples + config.batch_size - 1) / config.batch_size;

        std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::vector<size_t> sample_indices(num_training_samples);
        for (size_t i = 0; i < num_training_samples; ++i) sample_indices[i] = i;

        auto training_start = std::chrono::high_resolution_clock::now();

        int viz_update_counter = 0;
        const int viz_update_frequency = std::max(50, static_cast<int>(total_batches / 20));

        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            std::cout << "Epoch " << epoch << "/" << config.epochs << std::endl;
            std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

            for (size_t batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
                size_t batch_start = batch_idx * config.batch_size;
                size_t batch_end = std::min(batch_start + config.batch_size, num_training_samples);
                size_t current_batch_size = batch_end - batch_start;

                std::vector<float> batch_images(current_batch_size * 784);
                std::vector<int> batch_labels(current_batch_size);

                for (size_t i = 0; i < current_batch_size; ++i) {
                    size_t sample_idx = sample_indices[batch_start + i];
                    batch_labels[i] = mnist.train_labels[sample_idx];
                    std::memcpy(&batch_images[i * 784],
                        &mnist.train_images[sample_idx * 784],
                        784 * sizeof(float));
                }

                model.train_epoch(batch_images, batch_labels, current_batch_size);

                viz_update_counter++;
                if (viz_update_counter >= viz_update_frequency) {
                    viz_update_counter = 0;

                    size_t vis_idx = rand() % current_batch_size;
                    size_t vis_sample = sample_indices[batch_start + vis_idx];

                    VizSnapshot snap;
                    snap.layer_sizes = architecture;
                    snap.input_image.assign(
                        mnist.train_images.begin() + vis_sample * 784,
                        mnist.train_images.begin() + vis_sample * 784 + 784
                    );
                    snap.true_label = mnist.train_labels[vis_sample];

                    model.forward_sample_copy(snap.input_image.data(), snap.activations);

                    snap.epoch = epoch;
                    snap.batch_index = (int)batch_idx;
                    snap.loss = 0.0f;

                    viz.push_snapshot(snap);
                }

                if (batch_idx % 20 == 0) {
                    TrainingProgress prog;
                    prog.current_epoch = epoch;
                    prog.total_epochs = config.epochs;
                    prog.current_batch = (int)batch_idx;
                    prog.total_batches = (int)total_batches;
                    viz.update_training_progress(prog);
                }
            }

            float accuracy = model.evaluate(mnist.test_images, mnist.test_labels, mnist.n_test);
            std::cout << "Epoch " << epoch << " - Test accuracy: "
                << (accuracy * 100.0f) << "%" << std::endl;
        }

        auto training_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(training_end - training_start);

        std::cout << "\nTraining complete in " << duration.count() << " seconds!" << std::endl;
        float final_accuracy = model.evaluate(mnist.test_images, mnist.test_labels, mnist.n_test);
        std::cout << "Final accuracy: " << (final_accuracy * 100.0f) << "%" << std::endl;

        TrainingProgress final_prog;
        final_prog.training_complete = true;
        final_prog.accuracy = final_accuracy;
        viz.update_training_progress(final_prog);

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    std::cout << "\nSwitching to interactive drawing mode..." << std::endl;
    std::cout << "Draw digits with your mouse and see real-time predictions!" << std::endl;
    std::cout << "Press 'C' to clear the canvas" << std::endl;

    viz.set_mode(VisualizationMode::Interactive);
    viz.set_fps_limit(60);

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        std::vector<float> drawn_image;
        if (viz.get_drawn_image(drawn_image)) {
            float pixel_sum = 0.0f;
            for (float val : drawn_image) pixel_sum += val;

            if (pixel_sum > 0.1f) {
                std::vector<std::vector<float>> activations;
                model.forward_sample_copy(drawn_image.data(), activations);

                if (!activations.empty()) {
                    viz.set_prediction(activations.back());
                }
            }
        }
    }

    viz.stop();
    return 0;
}