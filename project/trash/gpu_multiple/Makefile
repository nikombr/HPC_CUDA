# Makefile 

TARGET_NO_REDUCTION  	= jacobi_no_reduction		# Not computing the error for convergence check
TARGET_REDUCTION 		= jacobi_reduction			# Computing the error for convergence check

SOURCES	= main.cu utils/print.cu utils/alloc3d.cu utils/init.cu 
OBJECTS	= utils/print.o utils/alloc3d.o utils/init.o 

MAIN_NO_REDUCTION		= main_no_reduction.o
MAIN_REDUCTION 			= main_reduction.o

OBJS_NO_REDUCTION		= $(MAIN_NO_REDUCTION) 		no_reduction/jacobi.o
OBJS_REDUCTION			= $(MAIN_REDUCTION) 		reduction/jacobi.o


# options and settings for the GCC compilers

OPT	= -g -O3
ISA	= 
PARA	= -fopenmp

XOPTS = -Xptxas=-v -arch=sm_90 -lineinfo

CC	= gcc

CCC	= g++
CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPTS)

F90C  	= gfortran
LIBS	= 

CUDA_PATH ?= /appl/cuda/12.2.0
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/Common
LDFLAGS = -lm 

all: $(TARGET_NO_REDUCTION) $(TARGET_REDUCTION) $(TARGET_REDUCTION_ATOMIC) 

$(TARGET_REDUCTION): $(OBJECTS) $(OBJS_REDUCTION)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_REDUCTION) $(OBJECTS) $(LDFLAGS)

$(MAIN_REDUCTION): main.cu 
	$(CXX) -o $@ -D_REDUCTION $(CXXFLAGS) $(INCLUDES) -c $<

$(TARGET_NO_REDUCTION): $(OBJECTS) $(OBJS_NO_REDUCTION)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_NO_REDUCTION) $(OBJECTS) $(LDFLAGS)

$(MAIN_NO_REDUCTION): main.cu
	$(CXX) -o $@ -D_NO_REDUCTION $(CXXFLAGS) $(INCLUDES) -c $<


#no_reduction/jacobi.o: no_reduction/jacobi.cu
#	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

#reduction/jacobi.o: reduction/jacobi.cu
#	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

%.o: %.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<


clean:
	@/bin/rm -f core *.o *~
	@/bin/rm -f core reduction/*.o
	@/bin/rm -f core no_reduction/*.o
	@/bin/rm -f core utils/*.o

realclean: clean
	@/bin/rm -f $(TARGET_NO_REDUCTION)  $(TARGET_REDUCTION) 

# DO NOT DELETE

main_no_reduction.o: 		main.cu utils/print.h no_reduction/jacobi.h utils/init.h 
main_reduction.o: 			main.cu utils/print.h reduction/jacobi.h utils/init.h 
utils/print.o: 				utils/print.h 
utils/alloc3d.o: 			utils/alloc3d.h 
utils/init.o: 				utils/init.h 
