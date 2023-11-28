from time import sleep
from func.dataFunctions import *
from matplotlib import pyplot as plt, scale
from func.LSTM_pytorch import *
from mpi4py import MPI
import torch
import torch.nn as nn

if __name__ == "__main__":
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
        print("It requires CUDA 12.1 to run and torch 2.1.1+cu121 to run.")


    path_apple = "data/AMZN.csv"
    file_paths = [path_apple]

    data_list_array = read_multiple_files_parallel(file_paths)

    apple_data = data_list_array[0][0]

    train_data, test_data, X_train, y_train, X_test, y_test = get_data_for_prediction(apple_data, 7, 0.8)

    # convert to loaders
    batch_size = 16

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    learning_rate = 0.001
    num_epochs = 10
    loss_function = nn.MSELoss()
    batch_size = 16

    # create the model
    model = start_lstm(train_loader, test_loader, learning_rate,num_epochs, loss_function, 16)

    predictions = predict_in_batches(model, X_train, batch_size=batch_size)



    # train_predictions = predictions.flatten()

    # dummy = np.zeros((X_train.shape[0], 8))
    # dummy[:, 0] = train_predictions
    # dummy = scaler.inverse_transform(dummy)

    # train_predictions = dc(dummy[:, 0])


    # plot the predictions
    plt.plot(y_train, label='Actual Close')
    plt.plot(predictions, label='Predicted Close')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Close')
    plt.title('Predictions')
    plt.show()

