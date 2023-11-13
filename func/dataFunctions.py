import csv
import numba as nb


# All Data follows the format of [Date, open, high, low, close, tick_volume, spread, real_volume]
# ** Chunk Size of -1 means all data will be returned
def get_data(file_name, start_row=1, chunk_size=2):
    data = []

    with open(file_name, "r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i > start_row:
                data.append(row)

            if i >= start_row + chunk_size:
                break
    return data


def get_data_parallel(dataFile, chunk_size, rank, size, start_row=1, counter=1):
    data = []
    while True:
        if counter == 200000:
            break
        data = get_data(dataFile, chunk_size=chunk_size, start_row=start_row)
        data.append(data)
        # print("Rank: ", rank, "Start Row: ", start_row, "End Row: ", chunk_size * rank + chunk_size,
        #        "Next Start Row: ", chunk_size * counter * rank + size * chunk_size + 1, "var", start_row , "Counter: ", counter)
        start_row = chunk_size * counter * rank + size * chunk_size + 1
        counter += 1

        if len(data) < chunk_size:
            print('Stopping Rank: ', rank, 'Counter: ', counter)
            break
    
    return data


