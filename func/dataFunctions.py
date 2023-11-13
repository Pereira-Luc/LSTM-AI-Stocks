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


