#make file - this is a comment section

all:    #target name
	mkdir -p bin
	nvcc main.c sparse_matrix.cu helper.c -o bin/main