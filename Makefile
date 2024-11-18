# Compiler and flags
NVCC := nvcc
CXXFLAGS := -w -Ikernels -Isrc --expt-relaxed-constexpr

# Source files
SRCS := src/lsm.cu src/main.cu

# Object files
OBJS := $(SRCS:.cu=.o)

# Executable
TARGET := GPULSM

# Default target
all: $(TARGET)

# Compile object files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Link object files into executable
$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the executable
run: $(TARGET)
	./$(TARGET)
