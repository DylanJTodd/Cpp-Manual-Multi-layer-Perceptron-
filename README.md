# Manual C++ Neural Network Training Simulation

A complete neural network trainer for MNIST digit classification with real-time visualization and pre-trained model support. Built from scratch in C++ with no external ML libraries (SFML for visualization). Features interactive configuration for architecture and hyperparameters, real-time training visualization with live neuron activations, interactive drawing interface for testing predictions, model persistence in JSON format, and support for loading pre-trained models.

## Quick Start

### Option 1: Download Pre-Built Executable (Windows Only)

1. Go to [Releases](https://github.com/DylanJTodd/Cpp-Manual-Multi-layer-Perceptron-/releases)
2. Download the latest release
3. Extract the ZIP file
4. Run `NeuralNetworkTrainer.exe`

The ZIP includes everything (exe, DLLs, MNIST dataset, pre-trained model). Just extract and run.

### Option 2: Build from Source (Windows - Visual Studio)

1. **Install prerequisites:**
   - Visual Studio 2019+ with C++ Desktop Development
   - SFML 2.5+ via vcpkg: `vcpkg install sfml:x64-windows` (NOTE: NOT SFML 3.0 COMPATIBLE)

2. **Clone and build:**
   ```bash
   git clone https://github.com/DylanJTodd/Cpp-Manual-Multi-layer-Perceptron-
   # Open CPP_MANUAL_MLP.sln in Visual Studio
   # Press F5 to build and run
   ```

3. **Download MNIST dataset** (see dataset setup below)

### Option 3: Build with Makefile (Linux/Mac/Windows with MinGW)

1. **Install dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential libsfml-dev
   
   # macOS
   brew install sfml
   
   # Windows with MinGW
   # Install SFML manually from https://www.sfml-dev.org/
   ```

2. **Clone and build:**
   ```bash
   git clone https://github.com/DylanJTodd/Cpp-Manual-Multi-layer-Perceptron-.git
   cd 
   make              # Builds main application
   make trainer      # Builds optimal model trainer
   ```

3. **Download MNIST dataset** (see dataset setup below)

4. **Run:**
   ```bash
   ./neural_net
   ```

### Option 4: Manual Compilation (Any Platform)

```bash
# Main application
g++ -o neural_net main.cpp mlp.cpp utils.cpp visualization.cpp \
    -lsfml-graphics -lsfml-window -lsfml-system -std=c++11 -O3 -pthread

# Optimal model trainer
g++ -o train_optimal train_optimal_model.cpp mlp.cpp utils.cpp \
    -std=c++11 -O3

# Run
./neural_net
```

## MNIST Dataset Setup

If you downloaded the pre-built executable (Option 1), skip this section as the dataset is included.

If building from source (Options 2-4):

1. Create a `mnist_data` folder in the project directory
2. Download these files from http://yann.lecun.com/exdb/mnist/:
   - `train-images-idx3-ubyte.gz`
   - `train-labels-idx1-ubyte.gz`
   - `t10k-images-idx3-ubyte.gz`
   - `t10k-labels-idx1-ubyte.gz`
3. Extract all files (remove `.gz` extension)
4. Place the 4 extracted files in the `mnist_data/` folder

**Linux quick download:**
```bash
mkdir mnist_data && cd mnist_data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

Alternatively, use the Makefile command:
```bash
make download-mnist
```

## Usage

### Training a Model

1. Launch the application
2. Select your network architecture (4 presets available)
3. Configure learning rate, batch size, and epochs
4. Click "START TRAINING"
5. Watch real-time training progress
6. Test your model by drawing digits

### Loading Pre-Trained Model

If using pre-built executable: Click "LOAD DEFAULT" to load the included model.

If building from source:
1. First, generate the model:
   ```bash
   ./train_optimal    # Creates optimal_model.json (takes 10-15 min)
   ```
2. Then in main application, click "LOAD DEFAULT"
3. Skip directly to drawing and testing

### Interactive Drawing

- Draw digits 0-9 with your mouse
- See confidence scores for all 10 digits
- Press 'C' to clear the canvas
- Test edge cases and unusual writing styles

## Optimal Model Architecture

The pre-trained model uses the following architecture:

- **Input Layer**: 784 neurons (28x28 pixels)
- **Hidden Layer 1**: 512 neurons (ReLU activation)
- **Hidden Layer 2**: 256 neurons (ReLU activation)
- **Hidden Layer 3**: 128 neurons (ReLU activation)
- **Output Layer**: 10 neurons (Softmax activation)
- **Optimizer**: Adam with momentum
- **Loss**: Cross-entropy

## Performance

| Configuration | Epochs | Accuracy | Training Time |
|--------------|--------|----------|---------------|
| Small (1x128) | 5 | ~95% | 2-3 min |
| Medium (2x256) | 10 | ~96-97% | 5-7 min |
| Large (3x512) | 10 | ~97% | 8-12 min |
| Optimal (512-256-128) | 15 | ~97-98% | 10-15 min |

*Tested on Intel i7-6700, times may vary*

## Makefile Commands

```bash
make                # Build main application
make trainer        # Build optimal model trainer
make full           # Build both applications
make clean          # Remove build artifacts
make install-deps   # Install SFML (Ubuntu/Debian only)
make download-mnist # Download MNIST dataset automatically
make help           # Show all available commands
```

## Acknowledgments

- **MNIST Database**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **SFML Library**: Laurent Gomila and contributors
