#include "mlp.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <limits>
#include <cstring>

static std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

MLP::MLP(const std::vector<int>& layer_sizes, float learning_rate, float b1, float b2, float e)
    : layer_sizes(layer_sizes.begin(), layer_sizes.end()), learning_rate(learning_rate),
    beta1(b1), beta2(b2), epsilon(e), adam_timestep(0)
{
    if (layer_sizes.size() < 2) {
        throw std::runtime_error("Network requires at least input and output layer");
    }
    num_layers = layer_sizes.size();
    initialize_parameters();
}

void MLP::initialize_parameters() {
    weights.resize(num_layers);
    biases.resize(num_layers);
    weight_momentum.resize(num_layers);
    weight_velocity.resize(num_layers);
    bias_momentum.resize(num_layers);
    bias_velocity.resize(num_layers);
    weight_gradients.resize(num_layers);
    bias_gradients.resize(num_layers);

    for (size_t layer = 1; layer < num_layers; ++layer) {
        int input_dim = layer_sizes[layer - 1];
        int output_dim = layer_sizes[layer];
        size_t weight_count = (size_t)input_dim * (size_t)output_dim;

        weights[layer].resize(weight_count);
        weight_momentum[layer].resize(weight_count, 0.0f);
        weight_velocity[layer].resize(weight_count, 0.0f);
        weight_gradients[layer].resize(weight_count, 0.0f);

        biases[layer].resize(output_dim, 0.0f);
        bias_momentum[layer].resize(output_dim, 0.0f);
        bias_velocity[layer].resize(output_dim, 0.0f);
        bias_gradients[layer].resize(output_dim, 0.0f);

        float xavier_limit = std::sqrt(6.0f / (input_dim + output_dim));
        std::uniform_real_distribution<float> distribution(-xavier_limit, xavier_limit);
        for (size_t i = 0; i < weight_count; ++i) {
            weights[layer][i] = distribution(rng);
        }
    }

    activations.clear();
    pre_activations.clear();
    deltas.clear();
}

void MLP::ensure_batch_buffers(size_t batch_size) {
    activations.resize(num_layers);
    pre_activations.resize(num_layers);
    deltas.resize(num_layers);

    for (size_t layer = 0; layer < num_layers; ++layer) {
        size_t required_size = (size_t)layer_sizes[layer] * batch_size;
        if (activations[layer].size() < required_size) {
            activations[layer].resize(required_size);
            pre_activations[layer].resize(required_size);
            deltas[layer].resize(required_size);
        }
    }
}

void MLP::matrix_multiply_and_add(
    const std::vector<float>& weight_matrix, int output_dim, int input_dim,
    const std::vector<float>& input_activations, int batch_size,
    std::vector<float>& output_buffer)
{
    const float* W = weight_matrix.data();
    const float* A = input_activations.data();
    float* C = output_buffer.data();

    for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
        const float* weight_row = W + (size_t)out_idx * (size_t)input_dim;
        float* output_row = C + (size_t)out_idx * (size_t)batch_size;

        int batch_idx = 0;
        for (; batch_idx + 7 < batch_size; batch_idx += 8) {
            float sum0 = output_row[batch_idx];
            float sum1 = output_row[batch_idx + 1];
            float sum2 = output_row[batch_idx + 2];
            float sum3 = output_row[batch_idx + 3];
            float sum4 = output_row[batch_idx + 4];
            float sum5 = output_row[batch_idx + 5];
            float sum6 = output_row[batch_idx + 6];
            float sum7 = output_row[batch_idx + 7];

            for (int in_idx = 0; in_idx < input_dim; ++in_idx) {
                float w = weight_row[in_idx];
                const float* input_row = A + (size_t)in_idx * (size_t)batch_size;
                sum0 += w * input_row[batch_idx];
                sum1 += w * input_row[batch_idx + 1];
                sum2 += w * input_row[batch_idx + 2];
                sum3 += w * input_row[batch_idx + 3];
                sum4 += w * input_row[batch_idx + 4];
                sum5 += w * input_row[batch_idx + 5];
                sum6 += w * input_row[batch_idx + 6];
                sum7 += w * input_row[batch_idx + 7];
            }

            output_row[batch_idx] = sum0;
            output_row[batch_idx + 1] = sum1;
            output_row[batch_idx + 2] = sum2;
            output_row[batch_idx + 3] = sum3;
            output_row[batch_idx + 4] = sum4;
            output_row[batch_idx + 5] = sum5;
            output_row[batch_idx + 6] = sum6;
            output_row[batch_idx + 7] = sum7;
        }

        for (; batch_idx < batch_size; ++batch_idx) {
            float sum = output_row[batch_idx];
            for (int in_idx = 0; in_idx < input_dim; ++in_idx) {
                sum += weight_row[in_idx] * A[(size_t)in_idx * batch_size + batch_idx];
            }
            output_row[batch_idx] = sum;
        }
    }
}

void MLP::apply_softmax_inplace(std::vector<float>& logits, int output_dim, int batch_size) {
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
            float val = logits[(size_t)out_idx * batch_size + batch_idx];
            if (val > max_logit) max_logit = val;
        }

        float sum_exp = 0.0f;
        for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
            float& logit = logits[(size_t)out_idx * batch_size + batch_idx];
            logit = std::exp(logit - max_logit);
            sum_exp += logit;
        }

        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
            logits[(size_t)out_idx * batch_size + batch_idx] *= inv_sum;
        }
    }
}

void MLP::accumulate_gradients(size_t layer_index, int batch_size) {
    int output_dim = layer_sizes[layer_index];
    int input_dim = layer_sizes[layer_index - 1];

    float* grad_W = weight_gradients[layer_index].data();
    float* grad_b = bias_gradients[layer_index].data();
    const float* delta = deltas[layer_index].data();
    const float* prev_activation = activations[layer_index - 1].data();

    for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
        const float* delta_row = delta + (size_t)out_idx * (size_t)batch_size;
        float* grad_W_row = grad_W + (size_t)out_idx * (size_t)input_dim;

        float bias_grad_sum = 0.0f;
        int batch_idx = 0;
        for (; batch_idx + 3 < batch_size; batch_idx += 4) {
            bias_grad_sum += delta_row[batch_idx] + delta_row[batch_idx + 1] +
                delta_row[batch_idx + 2] + delta_row[batch_idx + 3];
        }
        for (; batch_idx < batch_size; ++batch_idx) {
            bias_grad_sum += delta_row[batch_idx];
        }
        grad_b[out_idx] += bias_grad_sum;

        for (int in_idx = 0; in_idx < input_dim; ++in_idx) {
            const float* prev_act_row = prev_activation + (size_t)in_idx * (size_t)batch_size;
            float weight_grad_acc = 0.0f;

            batch_idx = 0;
            for (; batch_idx + 7 < batch_size; batch_idx += 8) {
                weight_grad_acc +=
                    delta_row[batch_idx] * prev_act_row[batch_idx] +
                    delta_row[batch_idx + 1] * prev_act_row[batch_idx + 1] +
                    delta_row[batch_idx + 2] * prev_act_row[batch_idx + 2] +
                    delta_row[batch_idx + 3] * prev_act_row[batch_idx + 3] +
                    delta_row[batch_idx + 4] * prev_act_row[batch_idx + 4] +
                    delta_row[batch_idx + 5] * prev_act_row[batch_idx + 5] +
                    delta_row[batch_idx + 6] * prev_act_row[batch_idx + 6] +
                    delta_row[batch_idx + 7] * prev_act_row[batch_idx + 7];
            }
            for (; batch_idx < batch_size; ++batch_idx) {
                weight_grad_acc += delta_row[batch_idx] * prev_act_row[batch_idx];
            }
            grad_W_row[in_idx] += weight_grad_acc;
        }
    }
}

void MLP::train_epoch(const std::vector<float>& flat_images,
    const std::vector<int>& labels,
    size_t batch_size) {
    size_t num_samples = labels.size();
    if (num_samples == 0) return;

    int input_size = layer_sizes[0];
    const size_t image_stride = (size_t)input_size;

    std::vector<size_t> sample_indices(num_samples);
    for (size_t i = 0; i < num_samples; ++i) sample_indices[i] = i;
    std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

    ensure_batch_buffers(batch_size);

    for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
        int current_batch_size = (int)std::min(batch_size, num_samples - batch_start);

        for (size_t layer = 1; layer < num_layers; ++layer) {
            std::memset(weight_gradients[layer].data(), 0, weight_gradients[layer].size() * sizeof(float));
            std::memset(bias_gradients[layer].data(), 0, bias_gradients[layer].size() * sizeof(float));
        }

        float* input_layer = activations[0].data();
        const float* image_data = flat_images.data();
        for (int feature = 0; feature < input_size; ++feature) {
            float* dest = input_layer + (size_t)feature * batch_size;
            for (int batch_idx = 0; batch_idx < current_batch_size; ++batch_idx) {
                dest[batch_idx] = image_data[sample_indices[batch_start + batch_idx] * image_stride + feature];
            }
            std::memset(dest + current_batch_size, 0, (batch_size - current_batch_size) * sizeof(float));
        }

        for (size_t layer = 1; layer < num_layers; ++layer) {
            int output_dim = layer_sizes[layer];
            int input_dim = layer_sizes[layer - 1];
            float* Z = pre_activations[layer].data();
            float* A = activations[layer].data();
            const float* bias = biases[layer].data();

            std::memset(Z, 0, output_dim * batch_size * sizeof(float));
            matrix_multiply_and_add(weights[layer], output_dim, input_dim,
                activations[layer - 1], (int)batch_size, pre_activations[layer]);

            bool is_output_layer = (layer + 1 == num_layers);

            if (!is_output_layer) {
                for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
                    float* Z_row = Z + (size_t)out_idx * batch_size;
                    float* A_row = A + (size_t)out_idx * batch_size;
                    float bias_val = bias[out_idx];

                    int batch_idx = 0;
                    for (; batch_idx + 3 < current_batch_size; batch_idx += 4) {
                        float z0 = Z_row[batch_idx] + bias_val;
                        float z1 = Z_row[batch_idx + 1] + bias_val;
                        float z2 = Z_row[batch_idx + 2] + bias_val;
                        float z3 = Z_row[batch_idx + 3] + bias_val;
                        A_row[batch_idx] = relu(z0);
                        A_row[batch_idx + 1] = relu(z1);
                        A_row[batch_idx + 2] = relu(z2);
                        A_row[batch_idx + 3] = relu(z3);
                    }
                    for (; batch_idx < current_batch_size; ++batch_idx) {
                        A_row[batch_idx] = relu(Z_row[batch_idx] + bias_val);
                    }
                    std::memset(A_row + current_batch_size, 0, (batch_size - current_batch_size) * sizeof(float));
                }
            }
            else {
                for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
                    float* Z_row = Z + (size_t)out_idx * batch_size;
                    float* A_row = A + (size_t)out_idx * batch_size;
                    float bias_val = bias[out_idx];

                    for (int batch_idx = 0; batch_idx < current_batch_size; ++batch_idx) {
                        A_row[batch_idx] = Z_row[batch_idx] + bias_val;
                    }
                    std::memset(A_row + current_batch_size, 0, (batch_size - current_batch_size) * sizeof(float));
                }
                apply_softmax_inplace(activations[layer], output_dim, (int)batch_size);
            }
        }

        size_t output_layer_size = (size_t)layer_sizes.back();
        float* output_activations = activations.back().data();
        float* output_delta = deltas.back().data();

        std::memcpy(output_delta, output_activations, output_layer_size * batch_size * sizeof(float));
        for (int batch_idx = 0; batch_idx < current_batch_size; ++batch_idx) {
            int label = labels[sample_indices[batch_start + batch_idx]];
            output_delta[label * batch_size + batch_idx] -= 1.0f;
        }

        accumulate_gradients(num_layers - 1, current_batch_size);

        for (int layer = (int)num_layers - 2; layer >= 1; --layer) {
            int output_dim = layer_sizes[layer];
            int next_layer_dim = layer_sizes[layer + 1];
            const float* next_weights = weights[layer + 1].data();
            const float* next_delta = deltas[layer + 1].data();
            float* current_delta = deltas[layer].data();
            const float* current_z = pre_activations[layer].data();

            std::memset(current_delta, 0, output_dim * batch_size * sizeof(float));

            for (int next_idx = 0; next_idx < next_layer_dim; ++next_idx) {
                const float* weight_row = next_weights + (size_t)next_idx * (size_t)output_dim;
                const float* delta_next = next_delta + (size_t)next_idx * batch_size;

                for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
                    float weight = weight_row[out_idx];
                    float* delta_current = current_delta + (size_t)out_idx * batch_size;
                    for (int batch_idx = 0; batch_idx < current_batch_size; ++batch_idx) {
                        delta_current[batch_idx] += weight * delta_next[batch_idx];
                    }
                }
            }

            for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
                float* delta_current = current_delta + (size_t)out_idx * batch_size;
                const float* z_current = current_z + (size_t)out_idx * batch_size;
                for (int batch_idx = 0; batch_idx < current_batch_size; ++batch_idx) {
                    delta_current[batch_idx] *= relu_derivative(z_current[batch_idx]);
                }
            }

            accumulate_gradients((size_t)layer, current_batch_size);
        }

        ++adam_timestep;
        float bias_correction1 = 1.0f - std::pow(beta1, (float)adam_timestep);
        float bias_correction2 = 1.0f - std::pow(beta2, (float)adam_timestep);
        float inv_batch = 1.0f / (float)current_batch_size;
        float adjusted_lr = learning_rate / bias_correction1;
        float beta1_complement = 1.0f - beta1;
        float beta2_complement = 1.0f - beta2;

        for (size_t layer = 1; layer < num_layers; ++layer) {
            int output_dim = layer_sizes[layer];
            int input_dim = layer_sizes[layer - 1];
            size_t weight_count = (size_t)output_dim * (size_t)input_dim;

            float* W = weights[layer].data();
            float* m_W = weight_momentum[layer].data();
            float* v_W = weight_velocity[layer].data();
            float* grad_W = weight_gradients[layer].data();

            for (size_t i = 0; i < weight_count; ++i) {
                float gradient = grad_W[i] * inv_batch;
                m_W[i] = beta1 * m_W[i] + beta1_complement * gradient;
                v_W[i] = beta2 * v_W[i] + beta2_complement * gradient * gradient;
                float v_corrected = v_W[i] / bias_correction2;
                W[i] -= adjusted_lr * m_W[i] / (std::sqrt(v_corrected) + epsilon);
            }

            float* b = biases[layer].data();
            float* m_b = bias_momentum[layer].data();
            float* v_b = bias_velocity[layer].data();
            float* grad_b = bias_gradients[layer].data();

            for (int i = 0; i < output_dim; ++i) {
                float gradient = grad_b[i] * inv_batch;
                m_b[i] = beta1 * m_b[i] + beta1_complement * gradient;
                v_b[i] = beta2 * v_b[i] + beta2_complement * gradient * gradient;
                float v_corrected = v_b[i] / bias_correction2;
                b[i] -= adjusted_lr * m_b[i] / (std::sqrt(v_corrected) + epsilon);
            }
        }
    }
}

float MLP::evaluate(const std::vector<float>& flat_images,
    const std::vector<int>& labels,
    size_t n_samples) {
    if (n_samples == 0) return 0.0f;

    size_t max_samples = std::min(n_samples, labels.size());
    size_t input_size = (size_t)layer_sizes[0];
    size_t eval_batch_size = 256;
    ensure_batch_buffers(eval_batch_size);

    size_t correct_predictions = 0;
    const float* image_data = flat_images.data();

    for (size_t batch_start = 0; batch_start < max_samples; batch_start += eval_batch_size) {
        int current_batch_size = (int)std::min(eval_batch_size, max_samples - batch_start);

        float* input_layer = activations[0].data();
        for (size_t feature = 0; feature < input_size; ++feature) {
            float* dest = input_layer + feature * eval_batch_size;
            for (int batch_idx = 0; batch_idx < current_batch_size; ++batch_idx) {
                dest[batch_idx] = image_data[(batch_start + batch_idx) * input_size + feature];
            }
            std::memset(dest + current_batch_size, 0, (eval_batch_size - current_batch_size) * sizeof(float));
        }

        for (size_t layer = 1; layer < num_layers; ++layer) {
            int output_dim = layer_sizes[layer];
            int input_dim = layer_sizes[layer - 1];
            float* Z = pre_activations[layer].data();
            float* A = activations[layer].data();
            const float* bias = biases[layer].data();

            std::memset(Z, 0, output_dim * eval_batch_size * sizeof(float));
            matrix_multiply_and_add(weights[layer], output_dim, input_dim,
                activations[layer - 1], (int)eval_batch_size, pre_activations[layer]);

            bool is_output_layer = (layer + 1 == num_layers);

            if (!is_output_layer) {
                for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
                    float* Z_row = Z + (size_t)out_idx * eval_batch_size;
                    float bias_val = bias[out_idx];
                    float* A_row = A + (size_t)out_idx * eval_batch_size;
                    for (int batch_idx = 0; batch_idx < current_batch_size; ++batch_idx) {
                        A_row[batch_idx] = relu(Z_row[batch_idx] + bias_val);
                    }
                    std::memset(A_row + current_batch_size, 0, (eval_batch_size - current_batch_size) * sizeof(float));
                }
            }
            else {
                for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
                    float* Z_row = Z + (size_t)out_idx * eval_batch_size;
                    float bias_val = bias[out_idx];
                    float* A_row = A + (size_t)out_idx * eval_batch_size;
                    for (int batch_idx = 0; batch_idx < current_batch_size; ++batch_idx) {
                        A_row[batch_idx] = Z_row[batch_idx] + bias_val;
                    }
                    std::memset(A_row + current_batch_size, 0, (eval_batch_size - current_batch_size) * sizeof(float));
                }
                apply_softmax_inplace(activations[layer], layer_sizes[layer], (int)eval_batch_size);
            }
        }

        const float* output = activations.back().data();
        int num_output_classes = layer_sizes.back();

        for (int batch_idx = 0; batch_idx < current_batch_size; ++batch_idx) {
            int predicted_class = 0;
            float max_probability = output[batch_idx];

            for (int class_idx = 1; class_idx < num_output_classes; ++class_idx) {
                float prob = output[(size_t)class_idx * eval_batch_size + batch_idx];
                if (prob > max_probability) {
                    max_probability = prob;
                    predicted_class = class_idx;
                }
            }

            if (predicted_class == labels[batch_start + batch_idx]) {
                ++correct_predictions;
            }
        }
    }

    return (float)correct_predictions / (float)max_samples;
}

void MLP::forward_sample_copy(const float* input, std::vector<std::vector<float>>& out_activations) {
    out_activations.resize(num_layers);
    for (size_t layer = 0; layer < num_layers; ++layer) {
        out_activations[layer].resize(layer_sizes[layer]);
    }

    std::memcpy(out_activations[0].data(), input, layer_sizes[0] * sizeof(float));

    for (size_t layer = 1; layer < num_layers; ++layer) {
        int output_dim = layer_sizes[layer];
        int input_dim = layer_sizes[layer - 1];

        for (int out_idx = 0; out_idx < output_dim; ++out_idx) {
            float z = biases[layer][out_idx];
            const float* weight_row = weights[layer].data() + (size_t)out_idx * (size_t)input_dim;
            const float* prev_activation = out_activations[layer - 1].data();

            for (int in_idx = 0; in_idx < input_dim; ++in_idx) {
                z += weight_row[in_idx] * prev_activation[in_idx];
            }

            if (layer + 1 < num_layers) {
                out_activations[layer][out_idx] = relu(z);
            }
            else {
                out_activations[layer][out_idx] = z;
            }
        }

        if (layer + 1 == num_layers) {
            float max_val = *std::max_element(out_activations[layer].begin(), out_activations[layer].end());
            float sum = 0.0f;
            for (int i = 0; i < output_dim; ++i) {
                out_activations[layer][i] = std::exp(out_activations[layer][i] - max_val);
                sum += out_activations[layer][i];
            }
            float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            for (int i = 0; i < output_dim; ++i) {
                out_activations[layer][i] *= inv_sum;
            }
        }
    }
}

void MLP::get_weights_copy(std::vector<std::vector<float>>& out_weights,
    std::vector<std::vector<float>>& out_biases) {
    out_weights.resize(num_layers);
    out_biases.resize(num_layers);

    for (size_t layer = 0; layer < num_layers; ++layer) {
        if (layer == 0) {
            out_weights[layer].clear();
            out_biases[layer].clear();
        }
        else {
            out_weights[layer] = weights[layer];
            out_biases[layer] = biases[layer];
        }
    }
}

void MLP::set_weights(const std::vector<std::vector<float>>& in_weights,
    const std::vector<std::vector<float>>& in_biases) {
    if (in_weights.size() != num_layers || in_biases.size() != num_layers) {
        throw std::runtime_error("Weight/bias dimensions don't match model architecture");
    }

    for (size_t layer = 1; layer < num_layers; ++layer) {
        if (in_weights[layer].size() == weights[layer].size()) {
            weights[layer] = in_weights[layer];
        }
        if (in_biases[layer].size() == biases[layer].size()) {
            biases[layer] = in_biases[layer];
        }
    }
}