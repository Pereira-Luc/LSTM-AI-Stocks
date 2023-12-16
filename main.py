from time import sleep
import time
from func.dataFunctions import *
from matplotlib import pyplot as plt, scale
from func.LSTM_pytorch import *
from func.ploting import *
#from mpi4py import MPI
import torch
import torch.nn as nn

# For testing 
from mpi4py.futures import MPIPoolExecutor as Executor

if __name__ == "__main__":
    print('Program started')

    # Check for GPU availability
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        print("CUDA (GPU support) is available on this machine.")
        print("Number of GPUs available:", torch.cuda.device_count())
        print("Name of the GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA (GPU support) is not available on this machine.")
        #print("It requires CUDA 12.1 to run and torch 2.1.1+cu121 to run.")


    path_amzn = "data/AMZN.csv"
    #path_apple = "data/APPL.csv"
    path_msfx = "data/MSFT.csv"
    path_tsla = "data/TSLA.csv"
    path_nflx = "data/NFLX.csv"

    file_paths = [path_amzn,path_msfx,path_tsla,path_nflx]

    data_list_array, list_of_scalers = read_data_in_parallel_future(file_paths)
    train_and_save_model_on_data(data_list_array, list_of_scalers, num_epochs=30)




    # print("Done saving models")

    # print("list of scalers", len(list_of_scalers))

    # print("Plotting models")

    # plot_model(model_rdy_list[0], list_of_scalers[0], prepared_data[0]['X_test'], prepared_data[0]["y_test"] , batch_size, device, historical_information)

    # test the model on not seen data
    # predictions = predict_in_batches(model_rdy_list[0], prepared_data[0]["X_test"], batch_size, device)

    # # reverse the normalization
    # test_predictions = reverse_normalization(predictions, historical_information, list_of_scalers[0])
    # y_test_reverse = reverse_normalization(prepared_data[0]["y_test"], historical_information, list_of_scalers[0])

    # plot the predictions
    #plt.plot(y_test_reverse, label='Actual Close')
    #plt.plot(test_predictions, label='Predicted Close')
    #plt.legend()
    #plt.xlabel('Days')
    #plt.ylabel('Close')
    #plt.title('Predictions')
    #plt.show()


