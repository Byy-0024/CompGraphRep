CXX = g++
CPPFLAGS = -std=c++20 -O3 -march=native
RM = rm -f
BIN_DIR  = ../bin
OBJS = Graph.o CompressGraph.o Interval.o
OBJ_FILES = $(addprefix $(BIN_DIR)/, $(OBJS))
# all: $(OBJS)
# 	$(CXX) $(CPPFLAGS) test.cpp Graph.o CompressGraph.o Interval.o -o $(BIN_DIR)/test
all: $(OBJ_FILES)
	$(CXX) $(CPPFLAGS) test.cpp $(OBJ_FILES) -o $(BIN_DIR)/test
	$(CXX) $(CPPFLAGS) -fopenmp reorder.cpp $(OBJ_FILES) -o $(BIN_DIR)/reorder

$(BIN_DIR)/%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ_FILES) test

.PHONY: all clean