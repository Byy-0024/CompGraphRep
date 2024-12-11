NVCC ?= nvcc
# ARCH ?= -arch=compute_60
INCL := -I. 
TASK := tc

# NVCC_FLAGS  := $(ARCH) -std=c++17
NVCC_FLAGS  := -std=c++20 -rdc=true -O3

ROOT_DIR := ~/yby/HVI
BIN_DIR := $(ROOT_DIR)/bin

FILES := ${TASK}.cu ../graph.cu $(BIN_DIR)/Graph.o $(BIN_DIR)/Interval.o
EXEC  := $(BIN_DIR)/${TASK}_gpu

all: $(EXEC)

$(EXEC): $(FILES)
	$(NVCC) $(NVCC_FLAGS) $(INCL) main.cu $^ -o $@
	
clean:
	rm -f $(EXEC)