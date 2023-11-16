import numba as nb
from mpi4py import MPI
import pandas as pd

"""
    This function is used to get the data from the CSV file
    It uses pandas to read the data from the file

    Args:
        file_name (str): Path to the input CSV file.
        start_row (int): Row number from where the data reading should start.
        chunk_size (int): Number of rows to be read at a time.
        custom_header (list): Custom header for the CSV file.

    Returns:
        data (DataFrame): Data read from the CSV file.
"""
def get_data(file_name, start_row=0, chunk_size=-1, custom_header=None):
    try:
        if chunk_size == -1:
            return pd.read_csv(file_name)


        if custom_header is not None:
            data = pd.read_csv(file_name, skiprows=start_row, nrows=chunk_size,header=None, names=custom_header)
        else:
            data = pd.read_csv(file_name, skiprows=start_row, nrows=chunk_size)
    except:
        print("Error reading file")
        return pd.DataFrame()

    return data

"""
    This function is used to get the data in parallel
    The data is split into chunks and each process gets a chunk of the data
    This function runs as long as there is data to be read

    Args:
        dataFile (str): Path to the input CSV file.
        chunk_size (int): Number of rows to be read at a time.
        rank (int): Rank of the process.
        size (int): Total number of processes.
        start_row (int): Row number from where the data reading should start.
        counter (int): Counter for the number of iterations.

    Returns:
        data (DataFrame): Data read from the CSV file.
"""
def get_data_parallel(dataFile, chunk_size, rank, size, start_row=1, counter=1):
    custom_header = ['Index','Date','open','high','low','close','tick_volume','spread','real_volume']
    data = pd.DataFrame()

    chunk_chache = []

    while True:
        chunk_of_data = get_data(dataFile, start_row, chunk_size,custom_header)
        if (not chunk_of_data.empty):
            # Whatever Modifications you want to do to the data
            chunk_of_data = data_manipulations_during_parallel_exec(chunk_of_data)
            #data = pd.concat([data, chunk_of_data])
            chunk_chache.append(chunk_of_data)

        # Update the start_row for the next iteration
        start_row += chunk_size * size

        counter += 1

        if len(chunk_of_data) < chunk_size:
            #print('Stopping Rank: ', rank, 'Iterations: ', counter)
            break
    if (chunk_chache != []):
        data = pd.concat(chunk_chache, ignore_index=True)

    return data


"""
Adds a row index to each row of a CSV file. The first row (header) gets 'Index' as its index.

Args:
file_path (str): Path to the input CSV file.
output_path (str): Path where the modified CSV file will be saved.
"""
def enumerate_data(file_name, output_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Adding 'Index' to the header
    lines[0] = 'Index,' + lines[0]

    # Adding row index to each line
    for i in range(1, len(lines)):
        lines[i] = f'{i - 1},' + lines[i]

    # Writing the modified content to a new file
    with open(output_name, 'w') as file:
        file.writelines(lines)


"""
    This function is used to get the data in parallel
    The data is split into chunks and each process gets a chunk of the data
    This function is used to change the data in parallel like normalizing the data
    or adding new columns or removing columns

    Don't remove Index column as it is used to sort the data

    Args:
        dataFile (str): Path to the input CSV file.
        chunk_size (int): Number of rows to be read at a time.
        rank (int): Rank of the process.
        size (int): Total number of processes.
        start_row (int): Row number from where the data reading should start.
        counter (int): Counter for the number of iterations.

    Returns:
        data (DataFrame): Newly modified data
"""
def data_manipulations_during_parallel_exec(data_chunk):
    #Whatever Modifications you want to do to the data
    return data_chunk

"""
    This function is used to get the data in parallel
    The data is split into chunks and each process gets a chunk of the data
    This function runs as long as there is data to be read

    Args:
        dataFile (str): Path to the input CSV file.
        chunk_size (int): Number of rows to be read at a time.

    Returns:
        data (DataFrame): Data read from the CSV file.
"""
def get_data_parallel_sorted(dataFile, chunk_size):
        all_data_chunks = pd.DataFrame()

        #start_time = MPI.Wtime()

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        start_row = rank * chunk_size + 1

        #print("Rank: ", rank, "Start Row: ", start_row, "End Row: ", chunk_size * rank + chunk_size)
        data_chunk = get_data_parallel(dataFile, chunk_size, rank, size, start_row)
        # Gather all chunks at the root process
        all_data_chunks = comm.gather(data_chunk, root=0)

        if rank == 0:
            print("Data all gathered at root process: ", rank)

            # combine all the data frames inside the
            all_data = pd.concat(all_data_chunks)

            # Sort the data according to the index
            all_data = all_data.sort_values(by=['Index'])

            #stop_time = MPI.Wtime()

            print("Data: ", len(all_data))
            #print("Data: ", all_data[:2])

            #print("Time: ", stop_time - start_time)


"""
    Function to read multiple files in parallel

    Also no need to use enumerate_data function as the data is sorted inside the files itself

    Args:
        file_paths (list): List of paths to the input files.

    Returns:
        data (DataFrame): Data read from the CSV file.
"""
def read_multiple_files_parallel(file_paths):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rank_counter = 0
    file_counter = 0
    data_list = []

    while file_counter  < len(file_paths):
        # print("Rank: ", rank, "File: ", file_paths[file_counter] + " Rank Counter: ", rank_counter , " File Counter: ", file_counter)
        # Giving each process a file to read
        if rank == rank_counter:
            print("Rank: ", rank, "File: ", file_paths[file_counter])
            data = get_data(file_paths[file_counter])
            # Whatever Modifications you want to do to the data
            data_list.append(data_manipulations_during_parallel_exec(data))
        
        file_counter += 1
        # Changing the rank of the process and file
        rank_counter += 1
        # In the case there are more files than processes
        # We need to reset the counter
        rank_counter = rank_counter % size 

        if file_counter < size:
            for i in range(file_counter, size):
                if rank == i: data = pd.DataFrame()

    # Gather all chunks at the root process
    all_data_chunks = comm.gather(data_list, root=0)

    return all_data_chunks


"""
This function is used to split data into train and test data

Args:
    data (DataFrame): Data read from the CSV file.
    train_percentage (float): Percentage of data to be used for training.

    Returns:
        train_data (DataFrame): Training data.
        test_data (DataFrame): Testing data.
"""
def get_train_test_data(data, train_percentage=0.8):
    train_data = data[:int(len(data) * train_percentage)]
    test_data = data[int(len(data) * train_percentage):]

    return train_data, test_data
