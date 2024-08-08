CXX = g++

CXXFLAGS =-I/usr/local/include/eigen3 -std=c++14

SRCS = src/main.cpp src/PLS.cpp
HEADERS = include/PLS.hpp
OBJS = $(patsubst src/%.cpp, build/%.o, $(SRCS))

TARGET = PLS
LIBRARY = libPLS.a

all: $(TARGET) $(LIBRARY)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

$(LIBRARY): $(OBJS)
	ar rcs $@ $^

build/%.o: src/%.cpp $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f build/*.o $(TARGET) $(LIBRARY)

# Phony targets
.PHONY: all clean
