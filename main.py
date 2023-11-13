from time import sleep

from func.dataFunctions import *
from mpi4py import MPI

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    path_apple = "data/AAPL.csv"

    all_data = []

    chunk_size = 100000
    start_row = rank * chunk_size + 1
    counter = 1

    if rank == 0:
        data = get_data(path_apple, chunk_size=chunk_size, start_row=1)
        all_data.append(data)
    else:
        all_data.append(get_data_parallel(path_apple, chunk_size, rank, size, start_row, counter))

    all_data = comm.gather(all_data, root=0)

    if rank == 0:
        print(len(all_data))
        # Sort the data by the first element in each row which is the date
        #all_data = sorted(all_data, key=lambda x: x[0][0])
        #print(all_data)
        pass

