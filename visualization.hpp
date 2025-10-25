#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <SFML/Graphics.hpp>

enum class VisualizationMode {
    Configuration,
    Training,
    Interactive
};

enum class ActivationFunction {
    ReLU,
    Tanh,
    Sigmoid,
    LeakyReLU
};

enum class Optimizer {
    SGD,
    Momentum,
    Adam
};

enum class WeightInit {
    Xavier,
    He,
    LeCun
};

struct LayerPreset {
    const char* name;
    std::vector<int> sizes;
};

const std::vector<LayerPreset> LAYER_PRESETS = {
    {"Small (1x128)", {128}},
    {"Medium (2x256)", {256, 256}},
    {"Large (3x512)", {512, 512, 512}},
    {"Deep (4x256)", {256, 256, 256, 256}},
};

const std::vector<float> LEARNING_RATE_OPTIONS = {
    0.0001f, 0.0005f, 0.001f, 0.005f, 0.01f
};

const std::vector<int> BATCH_SIZE_OPTIONS = {
    16, 32, 64, 128, 256
};

const std::vector<int> EPOCH_OPTIONS = {
    1, 3, 5, 10, 15, 20
};

struct TrainingConfig {
    int preset_index = 1;
    std::vector<int> hidden_layers = { 256, 256 };
    float learning_rate = 0.001f;
    int epochs = 3;
    int batch_size = 64;
    ActivationFunction activation = ActivationFunction::ReLU;
    Optimizer optimizer = Optimizer::Adam;
    WeightInit weight_init = WeightInit::Xavier;
    float momentum = 0.9f;

    std::vector<int> get_architecture() const {
        std::vector<int> arch = { 784 };
        arch.insert(arch.end(), hidden_layers.begin(), hidden_layers.end());
        arch.push_back(10);
        return arch;
    }
};

struct TrainingProgress {
    int current_epoch = 0;
    int total_epochs = 0;
    int current_batch = 0;
    int total_batches = 0;
    float current_loss = 0.0f;
    float accuracy = 0.0f;
    bool training_complete = false;
};

struct VizSnapshot {
    std::vector<int> layer_sizes;
    std::vector<float> input_image;
    std::vector<std::vector<float>> activations;
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> biases;
    int true_label = -1;
    int predicted_label = -1;
    int epoch = 0;
    int batch_index = 0;
    float loss = 0.0f;
};

struct ButtonLayout {
    float x, y, width, height;

    bool contains(float mouse_x, float mouse_y) const {
        return mouse_x >= x && mouse_x <= x + width &&
            mouse_y >= y && mouse_y <= y + height;
    }
};

class Visualizer {
public:
    Visualizer(int window_w = 1600, int window_h = 900);
    ~Visualizer();

    void start();
    void stop();

    void push_snapshot(const VizSnapshot& snap);
    void update_training_progress(const TrainingProgress& progress);
    void set_mode(VisualizationMode mode);
    void set_fps_limit(unsigned fps);

    bool is_config_complete() const { return config_complete; }
    bool should_load_default() const { return load_default_model; }
    TrainingConfig get_config() const;

    bool get_drawn_image(std::vector<float>& out_image);
    void set_prediction(const std::vector<float>& probabilities);

private:
    void run();
    void draw_config_mode(sf::RenderWindow& win);
    void draw_training_mode(sf::RenderWindow& win);
    void draw_interactive_mode(sf::RenderWindow& win);
    void draw_neural_network(sf::RenderWindow& win, const VizSnapshot& snap,
        int x_offset, int y_offset, int width, int height);

    void draw_button(sf::RenderWindow& win, const ButtonLayout& layout,
        const std::string& text, bool selected, bool hovered = false);

    ButtonLayout get_preset_button_layout(size_t index) const;
    ButtonLayout get_activation_button_layout(int index) const;
    ButtonLayout get_optimizer_button_layout(int index) const;
    ButtonLayout get_weight_init_button_layout(int index) const;
    ButtonLayout get_learning_rate_button_layout(size_t index) const;
    ButtonLayout get_batch_size_button_layout(size_t index) const;
    ButtonLayout get_epoch_button_layout(size_t index) const;
    ButtonLayout get_start_button_layout() const;
    ButtonLayout get_load_default_button_layout() const;

    int win_w, win_h;
    std::thread worker;
    std::mutex mutex_snap;
    std::mutex mutex_progress;
    std::mutex mutex_drawing;
    std::mutex mutex_config;
    std::condition_variable cv;

    VizSnapshot latest;
    bool has_snapshot = false;

    TrainingProgress progress;
    VisualizationMode mode = VisualizationMode::Configuration;

    std::atomic<bool> running;
    std::atomic<bool> config_complete;
    std::atomic<bool> load_default_model;
    unsigned fps = 30;

    TrainingConfig config;
    int lr_index = 2;
    int batch_index = 2;
    int epoch_index = 1;

    static constexpr int MNIST_IMAGE_SIZE = 28;
    static constexpr float DRAWING_BRUSH_INTENSITY = 0.8f;
    static constexpr float DRAWING_BRUSH_RADIUS = 1.5f;

    std::vector<std::vector<float>> canvas;
    bool is_drawing = false;
    std::vector<float> current_prediction;
    bool has_prediction = false;

    sf::Font font;
    bool font_loaded = false;

    float animation_time = 0.0f;
};