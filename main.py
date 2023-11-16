from time import sleep

from func.dataFunctions import *
from mpi4py import MPI

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    path_apple = "data/AAPL_index.csv"
    path_TSLA = "data/TSLA_index.csv"

    file_paths = [path_apple, path_TSLA,path_apple, path_TSLA,path_apple, path_TSLA]
    read_multiple_files_parallel(file_paths)

