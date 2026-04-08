# XTrans Shim — LD_PRELOAD library for NCCL/RCCL gate interception
#
# Build:  make
# Clean:  make clean
# Test:   make test   (prints intercepted hostname)

CC      ?= gcc
CFLAGS  := -Wall -Wextra -O2 -fPIC -std=c99
LDFLAGS := -shared -ldl

TARGET  := libxtrans_shim.so
SRC     := xtrans_shim.c
HDR     := xtrans_shim.h

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC)
	@echo "Built $(TARGET)"

clean:
	rm -f $(TARGET)

# Quick smoke test: verify gethostname interception works
test: $(TARGET)
	@echo "--- Testing gethostname interception ---"
	XTRANS_VERBOSE=1 XTRANS_HOSTNAME=test-node \
	  LD_PRELOAD=./$(TARGET) hostname
	@echo ""
	@echo "--- Testing stat interception ---"
	XTRANS_VERBOSE=1 XTRANS_SHMDEV=0x42 \
	  LD_PRELOAD=./$(TARGET) stat /dev/shm 2>&1 | head -5
