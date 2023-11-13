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
        while True:
            if counter == 200000:
                break
            data = get_data(path_apple, chunk_size=chunk_size, start_row=start_row)
            all_data.append(data)
            # print("Rank: ", rank, "Start Row: ", start_row, "End Row: ", chunk_size * rank + chunk_size,
            #        "Next Start Row: ", chunk_size * counter * rank + size * chunk_size + 1, "var", start_row , "Counter: ", counter)
            start_row = chunk_size * counter * rank + size * chunk_size + 1
            counter += 1

            if len(data) < chunk_size:
                print('Stopping Rank: ', rank, 'Counter: ', counter)
                break

    #all_data = comm.gather(all_data, root=0)
    # print(all_data)
