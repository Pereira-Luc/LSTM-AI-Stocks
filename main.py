from time import sleep

from func.dataFunctions import *
from mpi4py import MPI

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # path_apple = "data/AAPL_index.csv"
    # path_TSLA = "data/TSLA_index.csv"

    chunk_size = 200000
    file_path = "data/AAPL_index.csv"
    get_data_parallel_sorted(file_path, chunk_size)

    # calculate the time
    start_time = MPI.Wtime()
    data = get_data(file_path)
    end_time = MPI.Wtime()

    #print(data[2:])

    #print("Time taken: ", end_time - start_time)