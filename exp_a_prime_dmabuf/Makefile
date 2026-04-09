# Exp A' — DMA-BUF benchmark build
#
# Build:  make
# Clean:  make clean

NVCC     ?= nvcc
CUDA_HOME ?= /usr/local/cuda

NVCC_FLAGS := -O2 -std=c++17 -arch=sm_80 -Xcompiler "-Wall -Wextra"
LD_FLAGS   := -lcuda -lcudart

TARGET  := dmabuf_bench
SRC_DIR := src
SRCS_CU := $(SRC_DIR)/dmabuf_bench.cu $(SRC_DIR)/cumem_ipc.cu \
           $(SRC_DIR)/cudaipc.cu $(SRC_DIR)/bench.cu
SRCS_C  := $(SRC_DIR)/uds.c
HEADERS := $(wildcard $(SRC_DIR)/*.h)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS_CU) $(SRCS_C) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS_CU) $(SRCS_C) -o $@ $(LD_FLAGS)
	@echo "Built $(TARGET)"

clean:
	rm -f $(TARGET)
