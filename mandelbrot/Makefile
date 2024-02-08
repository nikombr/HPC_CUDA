TARGET	= mandelbrot
OBJS	= main.o mandel.o writepng.o

PNGWRITERPATH = pngwriter
ARCH	      = $(shell uname -p)
PNGWRTLPATH   = $(PNGWRITERPATH)/lib/$(ARCH)
PNGWRTIPATH   = $(PNGWRITERPATH)/include
PNGWRITERLIB  = $(PNGWRTLPATH)/libpngwriter.a

OPT	= -g -O3
ISA	= 
PARA	= -fopenmp

XOPTS = -Xptxas=-v -arch=sm_90 -lineinfo

CC	= gcc

CCC	= g++
CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS) -I $(PNGWRTIPATH)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPT)

F90C  	= gfortran
LIBS	= -L $(PNGWRTLPATH) -lpngwriter -lpng 

CUDA_PATH ?= /appl/cuda/12.2.0
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/Common

all: $(PNGWRITERLIB) $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

$(PNGWRITERLIB):
	@cd pngwriter/src && $(MAKE) -f Makefile

.SUFFIXES: .cu .c
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

.c.o:
	$(CCC) -o $*.o -c $*.c $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@cd pngwriter/src && $(MAKE) -f Makefile.gnu clean
	@rm -f $(PNGWRITERLIB)
	@rm -f $(TARGET)
	@rm -f mandelbrot.png

# dependencies
#
main.o  : main.cu mandel.h
mandel.o: mandel.cu
writepng.o: writepng.h writepng.cc
