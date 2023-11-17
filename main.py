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
    data_list_array = read_multiple_files_parallel(file_paths)


    if rank == 0:
        apple_data = data_list_array[0][0]

        print("Data: ", len(apple_data))
        train, test = get_train_test_data(apple_data, 0.8)

        print("Train: ", len(train))
        print("Test: ", len(test))

