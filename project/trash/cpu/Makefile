# Makefile 

TARGET_NO_REDUCTION  	= jacobi_no_reduction		# Not computing the error for convergence check
TARGET_REDUCTION 		= jacobi_reduction			# Computing the error for convergence check

SOURCES	= main.c utils/print.c utils/alloc3d.c utils/init.c 
OBJECTS	= utils/print.o utils/alloc3d.o utils/init.o 

MAIN_NO_REDUCTION		= main_no_reduction.o
MAIN_REDUCTION 			= main_reduction.o

OBJS_NO_REDUCTION		= $(MAIN_NO_REDUCTION) 		no_reduction/jacobi.o
OBJS_REDUCTION			= $(MAIN_REDUCTION) 		reduction/jacobi.o

# options and settings for the GCC compilers

CC	= gcc
DEFS	= 
OPT	= -g -O3 -fopenmp
IPO	= 
ISA	= 
CHIP	= 
ARCH	= 
PARA	= 
CFLAGS	= $(DEFS) $(ARCH) $(OPT) $(ISA) $(CHIP) $(IPO) $(PARA) $(XOPTS)
LDFLAGS = -lm 

all: $(TARGET_NO_REDUCTION) $(TARGET_REDUCTION) 

$(TARGET_REDUCTION): $(OBJECTS) $(OBJS_REDUCTION)
	$(CC) -o $@ $(CFLAGS) $(OBJS_REDUCTION) $(OBJECTS) $(LDFLAGS)

$(MAIN_REDUCTION):
	$(CC) -o $@ -D_REDUCTION $(CFLAGS) -c main.c

$(TARGET_NO_REDUCTION): $(OBJECTS) $(OBJS_NO_REDUCTION)
	$(CC) -o $@ $(CFLAGS) $(OBJS_NO_REDUCTION) $(OBJECTS) $(LDFLAGS)

$(MAIN_NO_REDUCTION):
	$(CC) -o $@ -D_NO_REDUCTION $(CFLAGS) -c main.c 

clean:
	@/bin/rm -f core *.o *~
	@/bin/rm -f core reduction/*.o
	@/bin/rm -f core no_reduction/*.o
	@/bin/rm -f core utils/*.o

realclean: clean
	@/bin/rm -f $(TARGET_NO_REDUCTION)  $(TARGET_REDUCTION)

# DO NOT DELETE

main_no_reduction.o: 		main.c utils/print.h no_reduction/jacobi.h utils/init.h 
main_reduction.o: 			main.c utils/print.h reduction/jacobi.h utils/init.h 
utils/print.o: 				utils/print.h
