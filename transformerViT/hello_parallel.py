from mpi4py import MPI
import numpy as np 

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    data = [39, 72, 129, 42]
else:
    data = None 

# simulate sending batches to nodes
data = comm.scatter(data, root=0)
print(f"Processor {rank} received {data}")
comm.Barrier() #here only for printing convenience
# simulate forward and back pass 
data = data * 0.5
print(f"Processor {rank} updated {data:.2f}")
# reduce 
resulting_sum = np.array(0.0,dtype=np.float64)
comm.Reduce(
        np.array(data, dtype=np.float64),
        resulting_sum, 
        op=MPI.SUM, root=0)

if rank == 0:
    print(f"Processor {rank} sum {resulting_sum:.2f}")
# broadcast the average
data = comm.bcast(resulting_sum/size, root=0)
print(f"Processor {rank} final {data:.2f}")

