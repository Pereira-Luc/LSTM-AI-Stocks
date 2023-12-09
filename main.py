from time import sleep
from func.dataFunctions import *
from matplotlib import pyplot as plt, scale
from func.LSTM_pytorch import *
from mpi4py import MPI
import torch
import torch.nn as nn

if __name__ == "__main__":
    print('Program started')
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Check for GPU availability
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        print("CUDA (GPU support) is available on this machine.")
        print("Number of GPUs available:", torch.cuda.device_count())
        print("Name of the GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA (GPU support) is not available on this machine.")
        #print("It requires CUDA 12.1 to run and torch 2.1.1+cu121 to run.")


    path_apple = "data/TSLA_index.csv"
    path_last_7days = "data/Last_7_days_AMZN.csv"

    testDataOctober = "data/ForTesting/AMZN_Oktober2023.csv"
    testDataNovember = "data/ForTesting/AMZN_November2023.csv"

    file_paths = [path_apple,path_last_7days, testDataOctober, testDataNovember]

    scaler = MinMaxScaler(feature_range=(0, 1))
    # This creates additional columns (features) based on the historical information like t-1, t-2, t-3, t-4, t-5, t-6, t-7...
    historical_information = 30

    # Get the data from the files already normalized
    data_list_array = read_multiple_files_parallel(file_paths, scaler)


    apple_data = data_list_array[0][0]

    train_data, test_data, X_train, y_train, X_test, y_test = get_data_for_prediction(apple_data, historical_information, 0.8)

    # convert to loaders
    batch_size = 32

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    learning_rate = 0.001
    num_epochs = 30
    loss_function = nn.MSELoss()

    # create the model
    model = start_lstm(train_loader, test_loader, learning_rate,num_epochs, loss_function, batch_size)

    #Save the moedel
    torch.save(model, 'model2.pth')

    #Test the model on seen data should be close quite close to the actual data
    #predictions = predict_in_batches(model, X_train, batch_size=batch_size)


    # reverse the normalization
    #train_predictions =  reverse_normalization(predictions, historical_information, scaler)
    #y_train_reverse = reverse_normalization(y_train, historical_information, scaler)


    # plot the predictions
    #plt.plot(y_train_reverse, label='Actual Close')
    #plt.plot(train_predictions, label='Predicted Close')
    #plt.legend()
    #plt.xlabel('Days')
    #plt.ylabel('Close')
    #plt.title('Predictions')
    #plt.show()

    # test the model on not seen data
    predictions = predict_in_batches(model, X_test, batch_size=batch_size)

    # reverse the normalization
    #test_predictions = reverse_normalization(predictions, historical_information, scaler)
    #y_test_reverse = reverse_normalization(y_test, historical_information, scaler)

    # plot the predictions
    #plt.plot(y_test_reverse, label='Actual Close')
    #plt.plot(test_predictions, label='Predicted Close')
   #plt.legend()
   # plt.xlabel('Days')
    #plt.ylabel('Close')
   # plt.title('Predictions')
   # plt.show()

    # Create predictions for the next 20 days using the last 7 days of data
   # last_October = data_list_array[0][2]
   # last_November = data_list_array[0][3]

    # prpare the data for prediction
  #  last_October = prepare_data_for_prediction(last_October, historical_information)

   # print('shape of last_October', last_October.shape)

    # predict the next 30 days
   # predictions = iterative_prediction(model, last_October, num_predictions=20)

    # reverse the normalization
    #predictions = reverse_normalization(predictions, historical_information, scaler)

    # actual data data of November
   # actual_data = last_November['close'].values

   # print('shape of predictions', predictions.shape)
   # print('shape of actual_data', actual_data.shape)

   # actual_data= reverse_normalization(actual_data, historical_information, scaler)

    # plot the predictions
   # plt.plot(predictions, label='Predicted Close')
   # plt.plot(actual_data, label='Actual Close')
   # plt.legend()
   # plt.xlabel('Days')
   # plt.ylabel('Close')
  #  plt.title('Predictions')
  #  plt.show()




