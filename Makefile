# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system -pthread

# Source files
MAIN_SOURCES = main.cpp mlp.cpp utils.cpp visualization.cpp
TRAINER_SOURCES = train_optimal_model.cpp mlp.cpp utils.cpp

# Object files
MAIN_OBJECTS = $(MAIN_SOURCES:.cpp=.o)
TRAINER_OBJECTS = $(TRAINER_SOURCES:.cpp=.o)

# Executables
MAIN_TARGET = neural_net
TRAINER_TARGET = train_optimal

# Default target
all: $(MAIN_TARGET)

# Main application
$(MAIN_TARGET): $(MAIN_OBJECTS)
	@echo "Linking main application..."
	$(CXX) $(MAIN_OBJECTS) -o $(MAIN_TARGET) $(LDFLAGS)
	@echo "Build complete! Run with: ./$(MAIN_TARGET)"

# Optimal model trainer
trainer: $(TRAINER_TARGET)

$(TRAINER_TARGET): $(TRAINER_OBJECTS)
	@echo "Linking trainer..."
	$(CXX) $(TRAINER_OBJECTS) -o $(TRAINER_TARGET)
	@echo "Trainer built! Run with: ./$(TRAINER_TARGET)"

# Compile source files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build everything
full: all trainer

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(MAIN_OBJECTS) $(TRAINER_OBJECTS) $(MAIN_TARGET) $(TRAINER_TARGET)
	@echo "Clean complete!"

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing SFML dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential libsfml-dev
	@echo "Dependencies installed!"

# Download MNIST dataset
download-mnist:
	@echo "Downloading MNIST dataset..."
	@mkdir -p mnist_data
	@cd mnist_data && \
	wget -q http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz && \
	wget -q http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz && \
	wget -q http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz && \
	wget -q http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz && \
	gunzip -f *.gz
	@echo "MNIST dataset downloaded and extracted to mnist_data/"

# Help target
help:
	@echo "Neural Network Training System - Build Commands"
	@echo ""
	@echo "Usage:"
	@echo "  make              Build main application"
	@echo "  make trainer      Build optimal model trainer"
	@echo "  make full         Build both applications"
	@echo "  make clean        Remove build artifacts"
	@echo "  make install-deps Install SFML (Ubuntu/Debian only)"
	@echo "  make download-mnist Download MNIST dataset"
	@echo "  make help         Show this help message"
	@echo ""
	@echo "After building, run:"
	@echo "  ./neural_net      Launch main application"
	@echo "  ./train_optimal   Generate pre-trained model"

# Phony targets (not actual files)
.PHONY: all trainer full clean install-deps download-mnist help