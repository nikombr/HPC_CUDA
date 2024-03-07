TARGET	= reduction
OBJS	= main.o reduction.o reduction_baseline.o reduction_smem.o

OPT	= -g -O3
ISA	= 
PARA	= -fopenmp

XOPTS = -Xptxas=-v -arch=sm_90 -lineinfo

CC	= gcc

CCC	= g++
CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPT)

F90C  	= gfortran
LIBS	= 

CUDA_PATH ?= /appl/cuda/12.2.0
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/Common

all: $(PNGWRITERLIB) $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)


.SUFFIXES: .cu .c
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

.c.o:
	$(CCC) -o $*.o -c $*.c $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@cd pngwriter/src && $(MAKE) -f Makefile.gnu clean
	@rm -f $(TARGET)

# dependencies
#
main.o  : main.cu reduction.h reduction_baseline.h reduction_smem.h
reduction.o: reduction.cu 
reduction_baseline.o: reduction_baseline.cu
reduction_smem.o: reduction_smem.cu
