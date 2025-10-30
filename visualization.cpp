#include "visualization.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <iostream>

Visualizer::Visualizer(int window_w, int window_h)
    : win_w(window_w), win_h(window_h), running(false), config_complete(false), load_default_model(false)
{
    canvas.resize(MNIST_IMAGE_SIZE, std::vector<float>(MNIST_IMAGE_SIZE, 0.0f));

    if (!font.loadFromFile("C:/Windows/Fonts/arial.ttf")) {
        std::cout << "Warning: Could not load font" << std::endl;
        font_loaded = false;
    }
    else {
        font_loaded = true;
    }

    config.preset_index = 1;
    config.hidden_layers = LAYER_PRESETS[1].sizes;
    config.learning_rate = LEARNING_RATE_OPTIONS[lr_index];
    config.batch_size = BATCH_SIZE_OPTIONS[batch_index];
    config.epochs = EPOCH_OPTIONS[epoch_index];
}

Visualizer::~Visualizer() {
    stop();
}

void Visualizer::start() {
    if (running) return;
    running = true;
    worker = std::thread(&Visualizer::run, this);
}

void Visualizer::stop() {
    if (!running) return;
    running = false;
    cv.notify_all();
    if (worker.joinable()) worker.join();
}

void Visualizer::push_snapshot(const VizSnapshot& snap) {
    std::lock_guard<std::mutex> lk(mutex_snap);
    latest = snap;
    has_snapshot = true;
}

void Visualizer::update_training_progress(const TrainingProgress& prog) {
    std::lock_guard<std::mutex> lk(mutex_progress);
    progress = prog;
}

void Visualizer::set_mode(VisualizationMode m) {
    mode = m;
    if (mode == VisualizationMode::Interactive) {
        std::lock_guard<std::mutex> lk(mutex_drawing);
        for (auto& row : canvas) {
            std::fill(row.begin(), row.end(), 0.0f);
        }
        has_prediction = false;
    }
}

void Visualizer::set_fps_limit(unsigned f) {
    fps = f;
}

TrainingConfig Visualizer::get_config() const {
    return config;
}

bool Visualizer::get_drawn_image(std::vector<float>& out_image) {
    std::lock_guard<std::mutex> lk(mutex_drawing);
    out_image.resize(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE);
    for (int r = 0; r < MNIST_IMAGE_SIZE; ++r) {
        for (int c = 0; c < MNIST_IMAGE_SIZE; ++c) {
            out_image[r * MNIST_IMAGE_SIZE + c] = canvas[r][c];
        }
    }
    return true;
}

void Visualizer::set_prediction(const std::vector<float>& probabilities) {
    std::lock_guard<std::mutex> lk(mutex_drawing);
    current_prediction = probabilities;
    has_prediction = true;
}

ButtonLayout Visualizer::get_preset_button_layout(size_t index) const {
    return { 50.0f + (index % 2) * 280.0f, 120.0f + (index / 2) * 60.0f, 260.0f, 45.0f };
}

ButtonLayout Visualizer::get_activation_button_layout(int index) const {
    return { 650.0f + (index % 2) * 140.0f, 120.0f + (index / 2) * 60.0f, 130.0f, 45.0f };
}

ButtonLayout Visualizer::get_optimizer_button_layout(int index) const {
    return { 1000.0f + (index % 2) * 140.0f, 120.0f + (index / 2) * 60.0f, 130.0f, 45.0f };
}

ButtonLayout Visualizer::get_weight_init_button_layout(int index) const {
    return { 1000.0f + (index % 2) * 140.0f, 250.0f + (index / 2) * 60.0f, 130.0f, 45.0f };
}

ButtonLayout Visualizer::get_learning_rate_button_layout(size_t index) const {
    return { 50.0f + index * 105.0f, 415.0f, 95.0f, 45.0f };
}

ButtonLayout Visualizer::get_batch_size_button_layout(size_t index) const {
    return { 650.0f + index * 105.0f, 415.0f, 95.0f, 45.0f };
}

ButtonLayout Visualizer::get_epoch_button_layout(size_t index) const {
    return { 50.0f + index * 105.0f, 535.0f, 95.0f, 45.0f };
}

ButtonLayout Visualizer::get_start_button_layout() const {
    return { 450.0f, 680.0f, 350.0f, 90.0f };
}

ButtonLayout Visualizer::get_load_default_button_layout() const {
    return { 850.0f, 680.0f, 350.0f, 90.0f };
}

void Visualizer::draw_button(sf::RenderWindow& win, const ButtonLayout& layout,
    const std::string& text, bool selected, bool hovered) {
    sf::RectangleShape btn(sf::Vector2f(layout.width, layout.height));
    btn.setPosition(sf::Vector2f(layout.x, layout.y));

    if (selected) {
        btn.setFillColor(sf::Color(80, 150, 255));
    }
    else if (hovered) {
        btn.setFillColor(sf::Color(70, 70, 100));
    }
    else {
        btn.setFillColor(sf::Color(50, 50, 70));
    }
    btn.setOutlineThickness(2);
    btn.setOutlineColor(selected ? sf::Color(120, 180, 255) : sf::Color(80, 80, 100));
    win.draw(btn);

    if (font_loaded && !text.empty()) {
        sf::Text label(text, font, 16);
        sf::FloatRect bounds = label.getLocalBounds();
        label.setPosition(sf::Vector2f(
            layout.x + (layout.width - bounds.width) / 2,
            layout.y + (layout.height - bounds.height) / 2 - 4
        ));
        label.setFillColor(sf::Color::White);
        win.draw(label);
    }
}

void Visualizer::run() {
    sf::RenderWindow window(sf::VideoMode(win_w, win_h), "Neural Network Trainer");
    window.setFramerateLimit(fps);
    sf::Event event;

    while (running && window.isOpen()) {
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
                running = false;
            }

            if (mode == VisualizationMode::Configuration && event.type == sf::Event::MouseButtonPressed) {
                float mouse_x = static_cast<float>(event.mouseButton.x);
                float mouse_y = static_cast<float>(event.mouseButton.y);
                std::lock_guard<std::mutex> lk(mutex_config);

                for (size_t i = 0; i < LAYER_PRESETS.size(); ++i) {
                    if (get_preset_button_layout(i).contains(mouse_x, mouse_y)) {
                        config.preset_index = static_cast<int>(i);
                        config.hidden_layers = LAYER_PRESETS[i].sizes;
                    }
                }

                for (int i = 0; i < 4; ++i) {
                    if (get_activation_button_layout(i).contains(mouse_x, mouse_y)) {
                        config.activation = static_cast<ActivationFunction>(i);
                    }
                }

                for (int i = 0; i < 3; ++i) {
                    if (get_optimizer_button_layout(i).contains(mouse_x, mouse_y)) {
                        config.optimizer = static_cast<Optimizer>(i);
                    }
                }

                for (int i = 0; i < 3; ++i) {
                    if (get_weight_init_button_layout(i).contains(mouse_x, mouse_y)) {
                        config.weight_init = static_cast<WeightInit>(i);
                    }
                }

                for (size_t i = 0; i < LEARNING_RATE_OPTIONS.size(); ++i) {
                    if (get_learning_rate_button_layout(i).contains(mouse_x, mouse_y)) {
                        lr_index = static_cast<int>(i);
                        config.learning_rate = LEARNING_RATE_OPTIONS[i];
                    }
                }

                for (size_t i = 0; i < BATCH_SIZE_OPTIONS.size(); ++i) {
                    if (get_batch_size_button_layout(i).contains(mouse_x, mouse_y)) {
                        batch_index = static_cast<int>(i);
                        config.batch_size = BATCH_SIZE_OPTIONS[i];
                    }
                }

                for (size_t i = 0; i < EPOCH_OPTIONS.size(); ++i) {
                    if (get_epoch_button_layout(i).contains(mouse_x, mouse_y)) {
                        epoch_index = static_cast<int>(i);
                        config.epochs = EPOCH_OPTIONS[i];
                    }
                }

                if (get_start_button_layout().contains(mouse_x, mouse_y)) {
                    config_complete = true;
                }

                if (get_load_default_button_layout().contains(mouse_x, mouse_y)) {
                    load_default_model = true;
                    config_complete = true;
                }
            }

            if (mode == VisualizationMode::Interactive) {
                if (event.type == sf::Event::MouseButtonPressed) {
                    is_drawing = true;
                    has_last_pos = false;
                }
                if (event.type == sf::Event::MouseButtonReleased) {
                    is_drawing = false;
                    has_last_pos = false;
                }
                if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::C) {
                    std::lock_guard<std::mutex> lk(mutex_drawing);
                    for (auto& row : canvas) {
                        std::fill(row.begin(), row.end(), 0.0f);
                    }
                    has_prediction = false;
                }
            }
        }

        if (mode == VisualizationMode::Interactive && is_drawing) {
            sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
            constexpr int canvas_pixel_size = 560;
            constexpr int canvas_offset_x = 50;
            constexpr int canvas_offset_y = 100;

            if (mouse_pos.x >= canvas_offset_x && mouse_pos.x < canvas_offset_x + canvas_pixel_size &&
                mouse_pos.y >= canvas_offset_y && mouse_pos.y < canvas_offset_y + canvas_pixel_size) {

                std::lock_guard<std::mutex> lk(mutex_drawing);

                if (has_last_pos) {
                    float x0 = (last_mouse_pos.x - canvas_offset_x) / 20.0f;
                    float y0 = (last_mouse_pos.y - canvas_offset_y) / 20.0f;
                    float x1 = (mouse_pos.x - canvas_offset_x) / 20.0f;
                    float y1 = (mouse_pos.y - canvas_offset_y) / 20.0f;

                    float distance = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
                    int num_steps = std::max(1, static_cast<int>(distance * 2));

                    for (int step = 0; step <= num_steps; ++step) {
                        float t = num_steps > 0 ? static_cast<float>(step) / num_steps : 0.0f;
                        float cell_x = x0 + t * (x1 - x0);
                        float cell_y = y0 + t * (y1 - y0);

                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                int nx = static_cast<int>(cell_x + dx);
                                int ny = static_cast<int>(cell_y + dy);
                                if (nx >= 0 && nx < MNIST_IMAGE_SIZE && ny >= 0 && ny < MNIST_IMAGE_SIZE) {
                                    float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy));
                                    float intensity = (dist < 1.0f) ? 1.0f :
                                        std::max(0.0f, 1.0f - (dist - 1.0f));
                                    canvas[ny][nx] = std::min(1.0f, canvas[ny][nx] + intensity * DRAWING_BRUSH_INTENSITY);
                                }
                            }
                        }
                    }
                }
                else {
                    float cell_x = (mouse_pos.x - canvas_offset_x) / 20.0f;
                    float cell_y = (mouse_pos.y - canvas_offset_y) / 20.0f;

                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int nx = static_cast<int>(cell_x + dx);
                            int ny = static_cast<int>(cell_y + dy);
                            if (nx >= 0 && nx < MNIST_IMAGE_SIZE && ny >= 0 && ny < MNIST_IMAGE_SIZE) {
                                float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy));
                                float intensity = (distance < 1.0f) ? 1.0f :
                                    std::max(0.0f, 1.0f - (distance - 1.0f));
                                canvas[ny][nx] = std::min(1.0f, canvas[ny][nx] + intensity * DRAWING_BRUSH_INTENSITY);
                            }
                        }
                    }
                }

                last_mouse_pos = mouse_pos;
                has_last_pos = true;
            }
        }

        animation_time += 0.016f;
        if (animation_time > 6.28f) animation_time = 0.0f;

        window.clear(sf::Color(15, 15, 20));

        if (mode == VisualizationMode::Configuration) {
            draw_config_mode(window);
        }
        else if (mode == VisualizationMode::Training) {
            draw_training_mode(window);
        }
        else {
            draw_interactive_mode(window);
        }

        window.display();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Visualizer::draw_config_mode(sf::RenderWindow& win) {
    TrainingConfig cfg;
    {
        std::lock_guard<std::mutex> lk(mutex_config);
        cfg = config;
    }

    if (font_loaded) {
        sf::Text title("Configure Your Neural Network", font, 36);
        title.setPosition(sf::Vector2f(50, 30));
        title.setFillColor(sf::Color(220, 220, 255));
        win.draw(title);

        struct SectionLabel {
            const char* text;
            float x;
            float y;
        };

        const SectionLabel section_labels[] = {
            {"Architecture", 50.0f, 85.0f},
            {"Activation", 650.0f, 85.0f},
            {"Optimizer", 1000.0f, 85.0f},
            {"Weight Init", 1000.0f, 215.0f},
            {"Learning Rate", 50.0f, 380.0f},
            {"Batch Size", 650.0f, 380.0f},
            {"Epochs", 50.0f, 500.0f}
        };

        for (const auto& section : section_labels) {
            sf::Text text(section.text, font, 22);
            text.setPosition(sf::Vector2f(section.x, section.y));
            text.setFillColor(sf::Color(200, 200, 220));
            win.draw(text);
        }
    }

    for (size_t i = 0; i < LAYER_PRESETS.size(); ++i) {
        bool selected = (cfg.preset_index == static_cast<int>(i));
        draw_button(win, get_preset_button_layout(i), LAYER_PRESETS[i].name, selected);
    }

    const char* activation_names[] = { "ReLU", "Tanh", "Sigmoid", "Leaky ReLU" };
    for (int i = 0; i < 4; ++i) {
        bool selected = (cfg.activation == static_cast<ActivationFunction>(i));
        draw_button(win, get_activation_button_layout(i), activation_names[i], selected);
    }

    const char* optimizer_names[] = { "SGD", "Momentum", "Adam" };
    for (int i = 0; i < 3; ++i) {
        bool selected = (cfg.optimizer == static_cast<Optimizer>(i));
        draw_button(win, get_optimizer_button_layout(i), optimizer_names[i], selected);
    }

    const char* weight_init_names[] = { "Xavier", "He", "LeCun" };
    for (int i = 0; i < 3; ++i) {
        bool selected = (cfg.weight_init == static_cast<WeightInit>(i));
        draw_button(win, get_weight_init_button_layout(i), weight_init_names[i], selected);
    }

    for (size_t i = 0; i < LEARNING_RATE_OPTIONS.size(); ++i) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << LEARNING_RATE_OPTIONS[i];
        bool selected = (lr_index == static_cast<int>(i));
        draw_button(win, get_learning_rate_button_layout(i), oss.str(), selected);
    }

    for (size_t i = 0; i < BATCH_SIZE_OPTIONS.size(); ++i) {
        bool selected = (batch_index == static_cast<int>(i));
        draw_button(win, get_batch_size_button_layout(i), std::to_string(BATCH_SIZE_OPTIONS[i]), selected);
    }

    for (size_t i = 0; i < EPOCH_OPTIONS.size(); ++i) {
        bool selected = (epoch_index == static_cast<int>(i));
        draw_button(win, get_epoch_button_layout(i), std::to_string(EPOCH_OPTIONS[i]), selected);
    }

    ButtonLayout start_layout = get_start_button_layout();
    sf::RectangleShape start_btn(sf::Vector2f(start_layout.width, start_layout.height));
    start_btn.setPosition(sf::Vector2f(start_layout.x, start_layout.y));
    start_btn.setFillColor(sf::Color(50, 200, 100));
    start_btn.setOutlineThickness(3);
    start_btn.setOutlineColor(sf::Color(70, 255, 130));
    win.draw(start_btn);

    if (font_loaded) {
        sf::Text start_text("START TRAINING", font, 32);
        sf::FloatRect text_bounds = start_text.getLocalBounds();
        start_text.setPosition(sf::Vector2f(
            start_layout.x + (start_layout.width - text_bounds.width) / 2,
            start_layout.y + 30
        ));
        start_text.setFillColor(sf::Color::White);
        win.draw(start_text);
    }

    ButtonLayout load_layout = get_load_default_button_layout();
    sf::RectangleShape load_btn(sf::Vector2f(load_layout.width, load_layout.height));
    load_btn.setPosition(sf::Vector2f(load_layout.x, load_layout.y));
    load_btn.setFillColor(sf::Color(100, 150, 255));
    load_btn.setOutlineThickness(3);
    load_btn.setOutlineColor(sf::Color(130, 180, 255));
    win.draw(load_btn);

    if (font_loaded) {
        sf::Text load_text("LOAD DEFAULT", font, 32);
        sf::FloatRect text_bounds = load_text.getLocalBounds();
        load_text.setPosition(sf::Vector2f(
            load_layout.x + (load_layout.width - text_bounds.width) / 2,
            load_layout.y + 30
        ));
        load_text.setFillColor(sf::Color::White);
        win.draw(load_text);
    }
}

void Visualizer::draw_training_mode(sf::RenderWindow& win) {
    VizSnapshot snap;
    bool have_snap = false;
    {
        std::lock_guard<std::mutex> lk(mutex_snap);
        if (has_snapshot) {
            snap = latest;
            have_snap = true;
        }
    }

    TrainingProgress prog;
    {
        std::lock_guard<std::mutex> lk(mutex_progress);
        prog = progress;
    }

    if (font_loaded) {
        sf::Text title("Training in Progress...", font, 32);
        title.setPosition(sf::Vector2f(50, 20));
        title.setFillColor(sf::Color(220, 220, 255));
        win.draw(title);
    }

    constexpr float progress_bar_x = 50.0f;
    constexpr float progress_bar_y = 70.0f;
    constexpr float progress_bar_height = 30.0f;
    float progress_bar_width = win_w - 100.0f;

    sf::RectangleShape progress_bg(sf::Vector2f(progress_bar_width, progress_bar_height));
    progress_bg.setPosition(sf::Vector2f(progress_bar_x, progress_bar_y));
    progress_bg.setFillColor(sf::Color(30, 30, 50));
    win.draw(progress_bg);

    if (prog.total_batches > 0) {
        float total_progress = (prog.current_epoch - 1) * prog.total_batches + prog.current_batch;
        float total_max = prog.total_epochs * prog.total_batches;
        float progress_pct = total_progress / total_max;

        sf::RectangleShape progress_fill(sf::Vector2f(progress_bar_width * progress_pct, progress_bar_height));
        progress_fill.setPosition(sf::Vector2f(progress_bar_x, progress_bar_y));
        progress_fill.setFillColor(sf::Color(80, 150, 255));
        win.draw(progress_fill);

        if (font_loaded) {
            std::ostringstream oss;
            oss << "Epoch " << prog.current_epoch << "/" << prog.total_epochs
                << " - Batch " << prog.current_batch << "/" << prog.total_batches
                << " (" << std::fixed << std::setprecision(1) << (progress_pct * 100) << "%)";
            sf::Text prog_text(oss.str(), font, 18);
            prog_text.setPosition(sf::Vector2f(progress_bar_x + 10, progress_bar_y + 5));
            prog_text.setFillColor(sf::Color::White);
            win.draw(prog_text);
        }
    }

    if (have_snap) {
        draw_neural_network(win, snap, 50, 120, win_w - 100, win_h - 170);
    }
}

void Visualizer::draw_neural_network(sf::RenderWindow& win, const VizSnapshot& snap,
    int x_offset, int y_offset, int width, int height) {
    const int num_layers = static_cast<int>(snap.layer_sizes.size());
    if (num_layers == 0) return;

    float layer_spacing = static_cast<float>(width) / std::max(1, num_layers - 1);
    std::vector<std::vector<sf::Vector2f>> node_positions(num_layers);

    for (int layer = 0; layer < num_layers; ++layer) {
        int num_nodes = snap.layer_sizes[layer];
        float layer_x = static_cast<float>(x_offset) + layer * layer_spacing;
        float available_height = static_cast<float>(height - 200);
        float start_y = static_cast<float>(y_offset + 100);
        float node_spacing = num_nodes > 1 ? available_height / (num_nodes - 1) : 0;

        node_positions[layer].resize(num_nodes);
        for (int node = 0; node < num_nodes; ++node) {
            float node_y = num_nodes == 1 ? start_y + available_height / 2 : start_y + node * node_spacing;
            node_positions[layer][node] = sf::Vector2f(layer_x, node_y);
        }
    }

    if (!snap.input_image.empty()) {
        constexpr float input_cell_size = 3.0f;
        float img_x = static_cast<float>(x_offset);
        float img_y = static_cast<float>(y_offset);

        for (int row = 0; row < MNIST_IMAGE_SIZE; ++row) {
            for (int col = 0; col < MNIST_IMAGE_SIZE; ++col) {
                int idx = row * MNIST_IMAGE_SIZE + col;
                if (idx < static_cast<int>(snap.input_image.size())) {
                    float val = snap.input_image[idx];
                    std::uint8_t intensity = static_cast<std::uint8_t>(255 * val);
                    sf::RectangleShape pixel(sf::Vector2f(input_cell_size, input_cell_size));
                    pixel.setPosition(sf::Vector2f(img_x + col * input_cell_size, img_y + row * input_cell_size));
                    pixel.setFillColor(sf::Color(intensity, intensity, intensity));
                    win.draw(pixel);
                }
            }
        }
    }

    constexpr int max_connections_to_draw = 50;
    for (int layer = 1; layer < num_layers; ++layer) {
        int num_output_nodes = snap.layer_sizes[layer];
        int num_input_nodes = snap.layer_sizes[layer - 1];

        for (int k = 0; k < max_connections_to_draw; ++k) {
            int output_idx = rand() % num_output_nodes;
            int input_idx = rand() % num_input_nodes;

            sf::Color connection_color = sf::Color(120, 120, 180, 30);
            sf::Vertex line[] = {
                sf::Vertex(node_positions[layer - 1][input_idx], connection_color),
                sf::Vertex(node_positions[layer][output_idx], connection_color)
            };
            win.draw(line, 2, sf::PrimitiveType::Lines);
        }
    }

    for (int layer = 0; layer < num_layers; ++layer) {
        int num_nodes = snap.layer_sizes[layer];
        int sample_step = std::max(1, num_nodes / 30);

        for (int node = 0; node < num_nodes; node += sample_step) {
            float activation = 0.0f;
            if (layer < static_cast<int>(snap.activations.size()) &&
                node < static_cast<int>(snap.activations[layer].size())) {
                activation = snap.activations[layer][node];
            }

            float node_radius = (layer == 0 || layer == num_layers - 1) ? 7.0f : 5.0f;
            sf::CircleShape node_circle(node_radius);
            node_circle.setOrigin(sf::Vector2f(node_radius, node_radius));
            node_circle.setPosition(node_positions[layer][node]);

            activation = std::max(0.0f, std::min(1.0f, activation));
            std::uint8_t intensity = static_cast<std::uint8_t>(60 + 190 * activation);

            node_circle.setFillColor(sf::Color(20, intensity, intensity));
            node_circle.setOutlineThickness(1.5f);
            node_circle.setOutlineColor(sf::Color(100, 100, 120));
            win.draw(node_circle);
        }
    }

    if (num_layers > 0 && !snap.activations.empty() && !snap.activations.back().empty()) {
        const auto& output_probs = snap.activations.back();
        int num_outputs = std::min(10, static_cast<int>(output_probs.size()));
        const auto& output_positions = node_positions.back();

        constexpr float prediction_bar_start_x_offset = 220.0f;
        float bar_start_x = static_cast<float>(x_offset + width) - prediction_bar_start_x_offset;
        constexpr float bar_width = 200.0f;
        constexpr float bar_height = 18.0f;

        if (font_loaded) {
            sf::Text title("Predictions", font, 20);
            title.setPosition(sf::Vector2f(bar_start_x, static_cast<float>(y_offset)));
            title.setFillColor(sf::Color(220, 220, 255));
            win.draw(title);
        }

        for (int i = 0; i < num_outputs && i < static_cast<int>(output_positions.size()); ++i) {
            float prob = output_probs[i];
            float node_y = output_positions[i].y;
            float bar_y = node_y - bar_height / 2.0f;

            sf::RectangleShape bg(sf::Vector2f(bar_width, bar_height));
            bg.setPosition(sf::Vector2f(bar_start_x, bar_y));
            bg.setFillColor(sf::Color(30, 30, 45));
            bg.setOutlineThickness(1.0f);
            bg.setOutlineColor(sf::Color(60, 60, 80));
            win.draw(bg);

            float filled_width = bar_width * prob;
            sf::RectangleShape bar(sf::Vector2f(filled_width, bar_height));
            bar.setPosition(sf::Vector2f(bar_start_x, bar_y));

            if (i == snap.true_label) {
                bar.setFillColor(sf::Color(100, 255, 100));
            }
            else {
                std::uint8_t g = static_cast<std::uint8_t>(100 + 155 * prob);
                bar.setFillColor(sf::Color(100, g, 140));
            }
            win.draw(bar);

            sf::Color line_color = (i == snap.true_label) ?
                sf::Color(100, 255, 100, 80) : sf::Color(120, 120, 180, 60);
            sf::Vertex connector_line[] = {
                sf::Vertex(output_positions[i], line_color),
                sf::Vertex(sf::Vector2f(bar_start_x - 5, node_y), line_color)
            };
            win.draw(connector_line, 2, sf::PrimitiveType::Lines);

            if (font_loaded) {
                std::ostringstream oss;
                oss << i << ": " << std::fixed << std::setprecision(1) << (prob * 100) << "%";
                sf::Text label(oss.str(), font, 13);
                label.setPosition(sf::Vector2f(bar_start_x + 5, bar_y + 1));
                label.setFillColor(sf::Color::White);
                win.draw(label);
            }
        }
    }
}

void Visualizer::draw_interactive_mode(sf::RenderWindow& win) {
    if (font_loaded) {
        sf::Text title("Draw a Digit (0-9)", font, 32);
        title.setPosition(sf::Vector2f(50, 20));
        title.setFillColor(sf::Color(220, 220, 255));
        win.draw(title);

        sf::Text instructions("Press 'C' to clear", font, 20);
        instructions.setPosition(sf::Vector2f(50, 65));
        instructions.setFillColor(sf::Color(180, 180, 200));
        win.draw(instructions);
    }

    constexpr int canvas_cell_size = 20;
    constexpr int canvas_offset_x = 50;
    constexpr int canvas_offset_y = 100;

    {
        std::lock_guard<std::mutex> lk(mutex_drawing);
        for (int row = 0; row < MNIST_IMAGE_SIZE; ++row) {
            for (int col = 0; col < MNIST_IMAGE_SIZE; ++col) {
                float val = canvas[row][col];
                std::uint8_t intensity = static_cast<std::uint8_t>(255 * val);

                sf::RectangleShape cell(sf::Vector2f(
                    static_cast<float>(canvas_cell_size - 1),
                    static_cast<float>(canvas_cell_size - 1)
                ));
                cell.setPosition(sf::Vector2f(
                    static_cast<float>(canvas_offset_x + col * canvas_cell_size),
                    static_cast<float>(canvas_offset_y + row * canvas_cell_size)
                ));
                cell.setFillColor(sf::Color(intensity, intensity, intensity));
                win.draw(cell);
            }
        }
    }

    int prediction_panel_x = canvas_offset_x + MNIST_IMAGE_SIZE * canvas_cell_size + 60;
    int prediction_panel_y = canvas_offset_y;

    if (font_loaded) {
        sf::Text pred_title("Predictions", font, 24);
        pred_title.setPosition(sf::Vector2f(
            static_cast<float>(prediction_panel_x),
            static_cast<float>(prediction_panel_y - 40)
        ));
        pred_title.setFillColor(sf::Color::White);
        win.draw(pred_title);
    }

    {
        std::lock_guard<std::mutex> lk(mutex_drawing);
        if (has_prediction && current_prediction.size() == 10) {
            constexpr float prediction_bar_width = 300.0f;
            constexpr float prediction_bar_height = 30.0f;
            constexpr float prediction_bar_spacing = 40.0f;

            for (int digit = 0; digit < 10; ++digit) {
                float prob = current_prediction[digit];
                float bar_y = static_cast<float>(prediction_panel_y + digit * prediction_bar_spacing);

                sf::RectangleShape bg(sf::Vector2f(prediction_bar_width, prediction_bar_height));
                bg.setPosition(sf::Vector2f(static_cast<float>(prediction_panel_x), bar_y));
                bg.setFillColor(sf::Color(40, 40, 60));
                win.draw(bg);

                float filled_bar_width = prediction_bar_width * prob;
                sf::RectangleShape bar(sf::Vector2f(filled_bar_width, prediction_bar_height));
                bar.setPosition(sf::Vector2f(static_cast<float>(prediction_panel_x), bar_y));
                std::uint8_t g = static_cast<std::uint8_t>(100 + 155 * prob);
                bar.setFillColor(sf::Color(100, g, 140));
                win.draw(bar);

                if (font_loaded) {
                    std::ostringstream oss;
                    oss << digit << ": " << std::fixed << std::setprecision(1) << (prob * 100) << "%";
                    sf::Text label(oss.str(), font, 18);
                    label.setPosition(sf::Vector2f(
                        static_cast<float>(prediction_panel_x + 10),
                        bar_y + 5
                    ));
                    label.setFillColor(sf::Color::White);
                    win.draw(label);
                }
            }
        }
    }
}