# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17

# Project files
SRC = cpuTest.cpp
TEMPLATE_IMPL = cpuLsm.tpp
HEADERS = cpuLsm.h
OBJECTS = cpuTest.o
EXEC = cpuTest

# Default target
all: $(EXEC)

# Build the test executable
$(EXEC): $(SRC) $(TEMPLATE_IMPL) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXEC)

# Run the tests
test: $(EXEC)
	./$(EXEC)

# Clean up build artifacts
clean:
	rm -f $(EXEC) $(OBJECTS)

# Phony targets (not actual files)
.PHONY: all test clean
