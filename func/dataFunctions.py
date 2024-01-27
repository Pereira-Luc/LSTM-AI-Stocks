from datetime import datetime

import numba as nb
from mpi4py import MPI
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from classes.StockDataset import StockDataset
from mpi4py.futures import MPIPoolExecutor as Executor

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
            return pd.read_csv(file_name, header=0)

        if custom_header is not None:
            data = pd.read_csv(file_name, skiprows=start_row, nrows=chunk_size,header=None, names=custom_header)
        else:
            data = pd.read_csv(file_name, skiprows=start_row, nrows=chunk_size)
    except:
        print("Error reading file")
        return pd.DataFrame()

    return data


"""
    This function is to normalize the data using any scaler given

    Don't remove Index column as it is used to sort the data

    Args:
        data (DataFrame): Data read from the CSV file.
        scaler (MinMaxScaler): Scaler to be used for normalizing the data.
    Returns:
        data (DataFrame): Newly modified data
"""
def data_manipulations_during_parallel_exec(data_chunk, scaler=MinMaxScaler()):
    # Normalizing the data using min-max normalization MinMaxScaler

    data_chunk['close'] = scaler.fit_transform(data_chunk[['close']])

    # Convert the date column to datetime format
    data_chunk['Date'] = pd.to_datetime(data_chunk['Date'])

    # only return the close and date columns
    return data_chunk[['Date', 'close']], scaler
    # Uncomment this for news
    #return data_chunk[['Date', 'close', 'sentiment_score','intensity_score','type_token_ratio']], scaler

"""
    This functions reverses the normalization done by the scaler

    Args:
        data (DataFrame): Data read from the CSV file.
        scaler (MinMaxScaler): Scaler to be used for normalizing the data.    

    Returns:
        data (DataFrame): Newly modified data

"""
def reverse_normalization(data, historical_information, scaler=MinMaxScaler()):
    d = np.zeros((data.shape[0], historical_information + 1))
    x = data.flatten()
    d[:, 0] = x
    d = scaler.inverse_transform(d)
    return dc(d[:, 0])
        


"""
    Function to read multiple files in parallel

    Also no need to use enumerate_data function as the data is sorted inside the files itself

    Args:
        file_paths (list): List of paths to the input files.
    Returns:
        data_list_array (DataFrame): List of DataFrames
        list_of_scalers: List of scalers
"""
def read_data_in_parallel_future(file_paths):
    # read multiple files using executor specify 4 cores
    with Executor(max_workers=5) as executor:
        data_list_array_executor = executor.map(get_data, file_paths)
        data_list_array_executor_list = list(data_list_array_executor)
        results = executor.map(data_manipulations_during_parallel_exec, data_list_array_executor_list)

        data_list_array = []
        list_of_scalers = []

        # Unpack each tuple from the results
        for data_chunk, scaler in results:
            data_list_array.append(data_chunk)
            list_of_scalers.append(scaler)
            print('data length', len(data_list_array))


    return data_list_array, list_of_scalers


"""
This function is used to split data into train and test data

Args:
    data (DataFrame): Data read from the CSV file.
    train_percentage (float): Percentage of data to be used for training.
    validation_percentage (float): Percentage of data to be used for validation.
    example: 10% validation is actually 10% of the remaining 80% of the training data

    Returns:
        train_data (DataFrame): Training data.
        test_data (DataFrame): Testing data.
        validation_data (DataFrame): Validation data.
"""
def get_split_data(data, train_percentage=0.8, validation_percentage=0.1, batch_size=64):
    # Splitting the data into train and test
    train_data = data.sample(frac=train_percentage, random_state=0)
    test_data = data.drop(train_data.index)

    # Splitting the train data into train and validation
    validation_data = train_data.sample(frac=validation_percentage, random_state=0)
    train_data = train_data.drop(validation_data.index)

    # Creating instances of StockDataset
    train_dataset = StockDataset(train_data)
    validation_dataset = StockDataset(validation_data)
    test_dataset = StockDataset(test_data)

    # Also converting the data into pytorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


"""
    This function is used to create a sequence of data for the LSTM model to use for prediction

    Args:
        data (DataFrame): Data read from the CSV file.
        seq_length (int): Length of the sequence meaning how many days of data to be used for prediction.

    Returns:
        seq_data (numpy): Data with sequence.
""" 
def create_sequence(data, seq_length):
    # data is not devided into days but into minutes
    # so we need to check if the data is for the same day or not
    # if it is for the same day then we can use it for prediction
    df = dc(data)
    df['Date'] = pd.to_datetime(df['Date'])

    df.set_index('Date', inplace=True)
    print('Data first 5 rows: ', df[:5])

    for i in range(1, seq_length + 1):
        df['close - ' + str(i)] = df['close'].shift(i)

        # Uncomment this for news
        #df['sentiment_score - ' + str(i)] = df['sentiment_score'].shift(i)
        #df['intensity_score - ' + str(i)] = df['intensity_score'].shift(i)
        #df['type_token_ratio - ' + str(i)] = df['type_token_ratio'].shift(i)

    df.dropna(inplace=True)

    print('Data first 5 rows after adding sequence: ', df[:5])

    return df.to_numpy()
    

"""
    This Function return data to be used for prediction by the LSTM model

    Args:
        data (DataFrame): Data read from the CSV file.
        go_back_days (int): Number of days to go back for prediction.

    Returns:
        train_data (StockDataset): Training data.
        test_data (StockDataset): Testing data.
        X_train (Tensor): Training data.
        y_train (Tensor): Training data.
        X_test (Tensor): Testing data.
        y_test (Tensor): Testing data.
"""
def get_data_for_prediction(data, historical_information = 7, train_percentage=0.8):

    seq_data = create_sequence(data, historical_information)

    # this is required for the LSTM model since we have 4 features Uncomment this for news
    # historical_information = historical_information * 4 + 3

    X = seq_data[:, 1:]
    X = dc(np.flip(X, axis=1))
    y = seq_data[:, 0]

    # Split the data into training and testing sets
    train_size = int(len(X) * train_percentage)

    X_train = X[:train_size]
    X_test = X[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    # convert data to tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()

    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    # add an additional dimension
    X_train = X_train.reshape(-1, historical_information, 1)
    X_test = X_test.reshape(-1, historical_information,1)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # convert to StockDataset
    train_data = StockDataset(X_train, y_train)
    test_data = StockDataset(X_test, y_test)

    return train_data, test_data, X_train, y_train, X_test, y_test


"""
    This function gives me the name of the stock depending on the first date in the file
"""
def get_stock_name(date):
    # AMZN = 1997-05-15
    # AAPL = 1980-12-12
    # MSFT = 1986-03-13
    # NFLX = 2002-05-23
    # TSLA = 2010-06-29
    if not isinstance(date, str):
        date = date.strftime('%Y-%m-%d')

    # Convert the input date string to a datetime object
    input_date = datetime.strptime(date, '%Y-%m-%d')
    
    # Define IPO dates for the stocks
    ipo_dates = {
        'AMZN': datetime(1997, 5, 15),
        'AAPL': datetime(1980, 12, 12),
        'MSFT': datetime(1986, 3, 13),
        'NFLX': datetime(2002, 5, 23),
        'TSLA': datetime(2010, 6, 29)
    }

    # Check each stock's IPO date
    for name, ipo_date in ipo_dates.items():
        if input_date == ipo_date:
            # Return the stock name if the input date matches exactly
            return name

    # Return a message if no exact match is found
    return "No stock found with the exact IPO date"


#------------------------------------------------------
# Stuff Down here do work but not without modificat of the main
# so don't use

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
@DeprecationWarning
def get_data_parallel(dataFile, chunk_size, rank, size, start_row=1, counter=1):
    custom_header = ['Index','Date','open','high','low','close','tick_volume','spread','real_volume']
    data = pd.DataFrame()

    chunk_chache = []

    while True:
        chunk_of_data = get_data(dataFile, start_row, chunk_size,custom_header)
        if (not chunk_of_data.empty):
            # Whatever Modifications you want to do to the data
            chunk_of_data = data_manipulations_during_parallel_exec(chunk_of_data)
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
    This function runs as long as there is data to be read

    Args:
        dataFile (str): Path to the input CSV file.
        chunk_size (int): Number of rows to be read at a time.

    Returns:
        data (DataFrame): Data read from the CSV file.
"""
@DeprecationWarning
def get_data_parallel_sorted(dataFile, chunk_size):
        all_data_chunks = pd.DataFrame()

        #start_time = MPI.Wtime()

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        start_row = rank * chunk_size + 1

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
        scaler (MinMaxScaler): Scaler to be used for normalizing the data.

    Returns:
        data (DataFrame): Data read from the CSV file.
    Format of the data returned:
        [
            [dataframe1, dataframe2, dataframe3],
            [dataframe1, dataframe2, dataframe3],
            [dataframe1, dataframe2, dataframe3],
            [dataframe1, dataframe2, dataframe3]
        ]
"""
@DeprecationWarning
def read_multiple_files_parallel(file_paths, scaler=MinMaxScaler()):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rank_counter = 0
    file_counter = 0
    data_list = []

    while file_counter  < len(file_paths):
        # Giving each process a file to read
        if rank == rank_counter:
            print("Rank: ", rank, "File: ", file_paths[file_counter])
            data = get_data(file_paths[file_counter])
            #data_list.append(data_manipulations_during_parallel_exec(data,scaler))
            data_list.append(data)
        
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