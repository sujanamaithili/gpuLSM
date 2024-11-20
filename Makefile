# Compiler and flags
NVCC := nvcc
CXXFLAGS := -w -Ikernels -Isrc --expt-relaxed-constexpr

# Source files
SRCS := src/main.cu

# Object files
OBJS := $(SRCS:.cu=.o)

# Executable
TARGET := GPULSM

# Example source and target
EXAMPLE_SRCS := src/example.cu
EXAMPLE_OBJS := $(EXAMPLE_SRCS:.cu=.o)
EXAMPLE_TARGET := GPULSM_example

# Default target
all: $(TARGET)

# Compile object files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Link object files into executable
$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# Build the example
example: $(EXAMPLE_TARGET)

$(EXAMPLE_TARGET): $(EXAMPLE_OBJS)
	$(NVCC) $(CXXFLAGS) $(EXAMPLE_OBJS) -o $(EXAMPLE_TARGET)

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

# Clean up example files
example_clean:
	rm -f $(EXAMPLE_OBJS) $(EXAMPLE_TARGET)

# Run the executable
run: $(TARGET)
	./$(TARGET)

# Run the example
run_example: example
	./$(EXAMPLE_TARGET)

