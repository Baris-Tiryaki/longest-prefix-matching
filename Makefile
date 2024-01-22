

CFLAGS= -Wall
DFLAGS=-g
RFLAGS=-O3

BUILD_TYPE=RELEASE

# Choose flags based on build type
ifeq ($(BUILD_TYPE), DEBUG)
	CFLAGS += $(DFLAGS)
else
	CFLAGS += $(RFLAGS)
endif


all: bin/main

bin/main: src/main.cpp src/*.hpp bin/LpmGpuSailL.o
	g++ src/main.cpp bin/LpmGpuSailL.o $(CFLAGS) -o bin/main -L/usr/local/cuda/lib64 -I/usr/local/cuda/include/ -lcudart

bin/LpmGpuSailL.o: src/LpmGpuSailL.cu src/LpmGpuSailL.hpp
	/usr/local/cuda-11.4/bin/nvcc -O3 -arch=compute_86 -code=sm_86,compute_86 --use_fast_math -Xcompiler "-O3" -Xptxas -O3,-v,-dlcm=ca -c src/LpmGpuSailL.cu -o bin/LpmGpuSailL.o


run: bin/main
	./bin/main

clean:
	rm -rf ./bin/*