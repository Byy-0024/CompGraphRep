NVCC ?= nvcc
ARCH ?= -arch=compute_60
INCL := -I. 

NVCC_FLAGS  := -std=c++20 -O3

ROOT_DIR := ~/yby/HVI
BIN_DIR := $(ROOT_DIR)/bin

FILES := bfs.cu ../graph.cu $(BIN_DIR)/Graph.o $(BIN_DIR)/Interval.o $(BIN_DIR)/CompressGraph.o
EXEC  := $(BIN_DIR)/bfs_gpu

all: $(EXEC)

$(EXEC): $(FILES)
	$(NVCC) $(NVCC_FLAGS) $(INCL) main.cu $^ -o $@
	
clean:
	rm -f $(EXEC)