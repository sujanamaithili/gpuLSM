# Compiler and flags
NVCC := nvcc
CXXFLAGS := -std=c++11 -I./kernels -I./src

# Source files
SRCS := src/lsm.cu kernels/merge.cu kernels/query.cu

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
