CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
INCLUDES = -I../include
LIBS = -lgtest -lgtest_main -lpthread

# Directories
SRC_DIR = ../src/common
TEST_DIR = .
BUILD_DIR = build

# Source files
SRCS = test_timingModel.cpp \
	   $(SRC_DIR)/timingModel.cpp

# Object files
OBJS = $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

# Test executable
TARGET = $(BUILD_DIR)/test_timing_model

.PHONY: all clean directories

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(SRC_DIR)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LIBS)

$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)