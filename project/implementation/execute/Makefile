# Makefile 

TARGET_CPU_NO_REDUCTION  	= cpu_no_reduction		# Not computing the error for convergence check
TARGET_CPU_REDUCTION 		= cpu_reduction			# Computing the error for convergence check

SOURCES	= main.c ../src/utils/print.c ../src/utils/malloc3d.c ../src/utils/init.c ../src/utils/dumpOutput.c
OBJECTS	= ../src/utils/print.o ../src/utils/malloc3d.o ../src/utils/init.o ../src/utils/dumpOutput.o

CPUFOLDER = ../src/cpu/

MAIN_CPU_NO_REDUCTION		= cpu_no_reduction.o
MAIN_CPU_REDUCTION 			= cpu_reduction.o

OBJS_CPU_NO_REDUCTION		= $(MAIN_CPU_NO_REDUCTION) 		$(CPUFOLDER)no_reduction.o $(CPUFOLDER)call.o
OBJS_CPU_REDUCTION			= $(MAIN_CPU_REDUCTION) 		$(CPUFOLDER)reduction.o $(CPUFOLDER)call.o

# options and settings for the GCC compilers

CC		= gcc
DEFS	= 
OPT		= -g -O3 -fopenmp
IPO		= 
ISA		= 
CHIP	= 
ARCH	= 
PARA	= 
CFLAGS	= $(DEFS) $(ARCH) $(OPT) $(ISA) $(CHIP) $(IPO) $(PARA) $(XOPTS)
LDFLAGS = -lm 

all: $(TARGET_CPU_NO_REDUCTION) $(TARGET_CPU_REDUCTION) 

$(TARGET_CPU_REDUCTION): $(OBJECTS) $(OBJS_CPU_REDUCTION)
	$(CC) -o $@ $(CFLAGS) $(OBJS_CPU_REDUCTION) $(OBJECTS) $(LDFLAGS)

$(MAIN_CPU_REDUCTION):
	$(CC) -o $@ -D_REDUCTION $(CFLAGS) -c main.c

$(TARGET_CPU_NO_REDUCTION): $(OBJECTS) $(OBJS_CPU_NO_REDUCTION)
	$(CC) -o $@ $(CFLAGS) $(OBJS_CPU_NO_REDUCTION) $(OBJECTS) $(LDFLAGS)

$(MAIN_CPU_NO_REDUCTION):
	$(CC) -o $@ -D_NO_REDUCTION $(CFLAGS) -c main.c 

clean:
	@/bin/rm -f core *.o *~
	@/bin/rm -f core $(CPUFOLDER)*.o
	@/bin/rm -f core ../src/utils/*.o

realclean: clean
	@/bin/rm -f $(TARGET_CPU_NO_REDUCTION)  $(TARGET_CPU_REDUCTION)

# DO NOT DELETE

cpu_no_reduction.o: 		main.c 
cpu_reduction.o: 			main.c
../src/utils/print.o: 		../lib/print.h
../src/utils/dumpOutput.o: 	../lib/print.h ../lib/dumpOutput.h
../src/cpu/call.o: 			../src/cpu/call.c ../lib/call.h ../lib/jacobi.h ../lib/init.h  ../lib/dumpOutput.h
