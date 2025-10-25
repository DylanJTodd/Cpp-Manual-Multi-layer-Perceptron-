#pragma once
#include <vector>
#include <cstddef>

class MLP {
public:
    MLP(const std::vector<int>& layer_sizes, float learning_rate = 1e-3f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

    void train_epoch(const std::vector<float>& flat_images,
        const std::vector<int>& labels,
        size_t batch_size);

    float evaluate(const std::vector<float>& flat_images,
        const std::vector<int>& labels,
        size_t n_samples);

    void forward_sample_copy(const float* input,
        std::vector<std::vector<float>>& out_activations);

    void get_weights_copy(std::vector<std::vector<float>>& out_weights,
        std::vector<std::vector<float>>& out_biases);

    void set_weights(const std::vector<std::vector<float>>& in_weights,
        const std::vector<std::vector<float>>& in_biases);

private:
    size_t num_layers;
    std::vector<int> layer_sizes;

    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> biases;

    std::vector<std::vector<float>> weight_momentum, weight_velocity;
    std::vector<std::vector<float>> bias_momentum, bias_velocity;
    int adam_timestep;

    std::vector<std::vector<float>> activations;
    std::vector<std::vector<float>> pre_activations;
    std::vector<std::vector<float>> deltas;

    std::vector<std::vector<float>> weight_gradients;
    std::vector<std::vector<float>> bias_gradients;

    float learning_rate, beta1, beta2, epsilon;

    static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
    static inline float relu_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }

    void initialize_parameters();
    void ensure_batch_buffers(size_t batch_size);

    static void matrix_multiply_and_add(
        const std::vector<float>& weight_matrix, int output_dim, int input_dim,
        const std::vector<float>& input_activations, int batch_size,
        std::vector<float>& output_buffer);

    void accumulate_gradients(size_t layer_index, int batch_size);

    static void apply_softmax_inplace(std::vector<float>& logits, int output_dim, int batch_size);

    static inline size_t matrix_index(size_t row, size_t col, size_t num_cols) {
        return row * num_cols + col;
    }
};