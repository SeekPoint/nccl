LD_FLAGS := -lnccl -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include

MPI_FLAGS := -I /usr/lib/x86_64-linux-gnu/openmpi/include -L /usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lmpi_cxx

EXE := ex02

all: $(EXE)

ex02: ex02.cpp
	g++ -g $< -o $@ $(LD_FLAGS)

.PHONY: clean
clean:
	-rm $(EXE)