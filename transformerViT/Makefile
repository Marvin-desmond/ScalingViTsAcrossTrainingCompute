
hello: build/hello_parallel
	mpirun -np 4 ./build/hello_parallel
	
build/hello_parallel: hello_parallel.c
	mpicc -o ./build/hello_parallel hello_parallel.c
