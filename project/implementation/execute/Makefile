# Makefile 

TARGET_CPU_NO_REDUCTION  		= cpu_no_reduction		# Not computing the error for convergence check
TARGET_CPU_REDUCTION 			= cpu_reduction			# Computing the error for convergence check
TARGET_GPU_NO_REDUCTION  		= gpu_no_reduction		# Not computing the error for convergence check
TARGET_GPU_REDUCTION 			= gpu_reduction			# Computing the error for convergence check
TARGET_GPU_REDUCTION_ATOMIC 	= gpu_reduction_atomic	# Computing the error for convergence check
TARGET_MGPU_NO_REDUCTION 		= mgpu_no_reduction		# Not computing the error for convergence check
TARGET_MGPU_REDUCTION 			= mgpu_reduction		# Computing the error for convergence check
# TARGET_MGPU_NO_REDUCTION_ASYN 	= mgpu_no_reduction_asyn

UTILS = ../src/utils/

SOURCES	= main.cpp $(UTILS)print.cu $(UTILS)malloc3d.cu $(UTILS)dumpOutput.cu $(UTILS)cudaMalloc3d.cu $(UTILS)alloc.cpp $(UTILS)init.cpp $(UTILS)poisson.cpp $(UTILS)setupMultipleGPU.cpp $(UTILS)sendToDevice.cpp $(UTILS)sendToHost.cpp $(UTILS)swapArrays.cpp $(UTILS)finalize.cpp $(UTILS)setup_f_matrix.cpp
OBJECTS	= 		   $(UTILS)print.o  $(UTILS)malloc3d.o  $(UTILS)dumpOutput.o  $(UTILS)cudaMalloc3d.o  $(UTILS)alloc.o 	$(UTILS)init.o 	 $(UTILS)poisson.o   $(UTILS)setupMultipleGPU.o   $(UTILS)sendToDevice.o   $(UTILS)sendToHost.o   $(UTILS)swapArrays.o   $(UTILS)finalize.o	  $(UTILS)setup_f_matrix.o

CPUFOLDER = ../src/cpu/
GPUFOLDER = ../src/gpu/
MGPUFOLDER = ../src/mgpu/

OBJS_CPU_NO_REDUCTION		= main.o 	$(CPUFOLDER)no_reduction.o 				$(CPUFOLDER)call.o
OBJS_CPU_REDUCTION			= main.o 	$(CPUFOLDER)reduction.o 				$(CPUFOLDER)call.o
OBJS_GPU_NO_REDUCTION		= main.o 	$(GPUFOLDER)no_reduction/jacobi.o 		$(GPUFOLDER)call.o  $(GPUFOLDER)no_reduction/iteration.o
OBJS_GPU_REDUCTION			= main.o 	$(GPUFOLDER)reduction/jacobi.o 			$(GPUFOLDER)call.o  $(GPUFOLDER)reduction/iteration.o
OBJS_GPU_REDUCTION_ATOMIC	= main.o 	$(GPUFOLDER)reduction_atomic/jacobi.o 	$(GPUFOLDER)call.o  $(GPUFOLDER)reduction_atomic/iteration.o
OBJS_MGPU_NO_REDUCTION		= main2.o 	$(MGPUFOLDER)no_reduction/jacobi.o 		$(MGPUFOLDER)call.o $(MGPUFOLDER)no_reduction/iteration.o
OBJS_MGPU_REDUCTION			= main2.o 	$(MGPUFOLDER)reduction/jacobi.o 		$(MGPUFOLDER)call.o $(MGPUFOLDER)reduction/iteration.o
# OBJS_MGPU_NO_REDUCTION_ASYN	= main2.o 	$(MGPUFOLDER)no_reduction_asyn/jacobi.o $(MGPUFOLDER)call.o $(MGPUFOLDER)no_reduction_asyn/iteration.o

# options and settings for the GCC compilers

CC		= gcc
CCC		= g++
CXX		= nvcc
MPICC	= mpicc
DEFS	= 
OPT		= -g -O3
IPO		= 
ISA		= 
CHIP	= 
ARCH	= 
PARA	= -fopenmp
XOPTS 	= -Xptxas=-v -arch=sm_90 -lineinfo -diag-suppress 2464 -diag-suppress 68  -diag-suppress 177 # Suppressing error regarding string conversion
CFLAGS	= $(DEFS) $(ARCH) $(OPT) $(ISA) $(CHIP) $(IPO) $(PARA) -lstdc++
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)
LDFLAGS = -lm $(INCLUDES) $(SOFLAGS) $(XLIBS)
LIBS	= 
CUDA_PATH ?= /appl/cuda/12.2.0
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/Common 
SOFLAGS = -L$(CUDA_PATH)/lib64  -L/appl/nccl/2.17.1-1-cuda-12.1/lib
XLIBS =  -lcudart -lnccl

all: $(TARGET_CPU_NO_REDUCTION) $(TARGET_CPU_REDUCTION) $(TARGET_GPU_REDUCTION) $(TARGET_GPU_NO_REDUCTION) $(TARGET_GPU_REDUCTION_ATOMIC) $(TARGET_MGPU_NO_REDUCTION) $(TARGET_MGPU_REDUCTION) # $(TARGET_MGPU_NO_REDUCTION_ASYN)

main.o: main.cpp
	$(MPICC) -o $@ -D_CPU_GPU $(CFLAGS) -c $<

main2.o: main.cpp
	$(MPICC) -o $@ -D_MGPU $(CFLAGS) -c $<

## CPU

$(TARGET_CPU_REDUCTION): $(OBJECTS) $(OBJS_CPU_REDUCTION)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_CPU_NO_REDUCTION): $(OBJECTS) $(OBJS_CPU_NO_REDUCTION)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

## GPU

$(TARGET_GPU_REDUCTION): $(OBJECTS) $(OBJS_GPU_REDUCTION)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_GPU_NO_REDUCTION): $(OBJECTS) $(OBJS_GPU_NO_REDUCTION)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_GPU_REDUCTION_ATOMIC): $(OBJECTS) $(OBJS_GPU_REDUCTION_ATOMIC)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# MGPU

$(TARGET_MGPU_NO_REDUCTION): $(OBJECTS) $(OBJS_MGPU_NO_REDUCTION)
	$(MPICC) $(CFLAGS) -o $@ -D_NORMAL_MAKE $^ $(LDFLAGS)

$(TARGET_MGPU_REDUCTION): $(OBJECTS) $(OBJS_MGPU_REDUCTION)
	$(MPICC) $(CFLAGS) -o $@ -D_NORMAL_MAKE $^ $(LDFLAGS)

# $(TARGET_MGPU_NO_REDUCTION_ASYN): $(OBJECTS) $(OBJS_MGPU_NO_REDUCTION_ASYN)
# 	$(MPICC) $(CFLAGS) -o $@ -D_NORMAL_MAKE $^ $(LDFLAGS)

# Utils

%.o: %.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -D_NORMAL_MAKE -c $<

%.o: %.cpp
	$(MPICC) -o $@ -D_NORMAL_MAKE $(CFLAGS) -c $<


clean:
	@/bin/rm -f core *.o *~
	@/bin/rm -f core $(CPUFOLDER)*.o
	@/bin/rm -f core $(GPUFOLDER)*.o
	@/bin/rm -f core $(GPUFOLDER)/no_reduction/*.o
	@/bin/rm -f core $(GPUFOLDER)/reduction/*.o
	@/bin/rm -f core $(GPUFOLDER)/reduction_atomic/*.o
	@/bin/rm -f core $(MGPUFOLDER)*.o
	@/bin/rm -f core $(MGPUFOLDER)/no_reduction/*.o
	@/bin/rm -f core $(MGPUFOLDER)/reduction/*.o
	@/bin/rm -f core ../src/utils/*.o

realclean: clean
	@/bin/rm -f $(TARGET_CPU_NO_REDUCTION) $(TARGET_CPU_REDUCTION) $(TARGET_GPU_REDUCTION) $(TARGET_GPU_NO_REDUCTION) $(TARGET_GPU_REDUCTION_ATOMIC) $(TARGET_MGPU_NO_REDUCTION) $(TARGET_MGPU_REDUCTION)

# Dependencies

# main.o: main.c
# ../src/utils/malloc3d.o: 	../lib/malloc3d.h
# ../src/utils/init.o: 	../lib/init.h
# ../src/utils/print.o: 		../lib/print.h
# ../src/utils/dumpOutput.o: 	../lib/print.h ../lib/dumpOutput.h
# ../src/cpu/call.o: 			../src/cpu/call.cu ../lib/call.h ../lib/jacobi.h ../lib/init.h  ../lib/dumpOutput.h
