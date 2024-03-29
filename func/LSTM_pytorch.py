from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from func.dataFunctions import *
import numpy as np
from classes.StockDataset import StockDataset
from classes.LSTM import LSTM
from mpi4py.futures import MPIPoolExecutor as Executor

"""
    This function is used to train the model for one epoch

    Args:
        epoch: the epoch number
        optimizer: the optimizer to be used
        loss_function: the loss function to be used
        model: the model to be used for training
        train_loader: the train loader to be used for training
    
    Returns:
        None
"""

def train_one_epoch(epoch, optimizer, loss_function, model, train_loader, device):
    model.train(True)
    print('Training...')
    running_loss = 0.0
    # Iterate over data to train the model for one epoch
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        outputs = model(x_batch)
        loss = loss_function(outputs, y_batch)
        running_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print statistics
        if batch_index % 100 == 99:
            avg_loss = running_loss / 100
            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_index + 1, avg_loss))
            running_loss = 0.0
    print()

"""
    This function is used to validate the model for one epoch

    Args:
        loss_function: the loss function to be used
        model: the model to be used for validation
        test_loader: the test loader to be used for validation

    Returns:
        None
"""

def validate_one_epoch(loss_function, model, test_loader, device):
    model.train(False)
    print('Validating...')
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            outputs = model(x_batch)
            loss = loss_function(outputs, y_batch)
            running_loss += loss
        
    avg_loss = running_loss / len(test_loader)

    print('Avg. Validation Loss: {}'.format(avg_loss))
    print("--------------------------------------------------------------")
    print()


"""
    This function is used to predict the stock price for the dataset in batches
    Args:
        model: the model to be used for prediction
        dataset: the dataset to be used for prediction
        batch_size: the batch size to be used for prediction

    Returns:
        predictions: the predictions for the input dataset
"""
def predict_in_batches(model, dataset, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size].to(device)
            batch_predictions = model(batch).cpu()
            predictions.append(batch_predictions)
    return torch.cat(predictions, dim=0).numpy()


"""
    This function is used to start the lstm model training

    Args:
        train_loader[DataLoader]: the train loader to be used for training 
        test_loader[DataLoader]: the test loader to be used for validation
        learning_rate: the learning rate to be used for training
        num_epochs: the number of epochs to be used for training
        loss_function: the loss function to be used for training
        batch_size: the batch size to be used for training
    
    Returns:
        model: the trained model
"""
def start_lstm(train_loader, test_loader, learning_rate = 0.001, num_epochs = 10, loss_function = nn.MSELoss(),batch_size = 16, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), identifyer = 0):
    # create the model
    model = LSTM(input_size=1, hidden_size=4, num_layers=2, output_size=1, device = device).to(device)
    model

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(epoch, optimizer, loss_function,model, train_loader,device)
        validate_one_epoch(loss_function,model,test_loader,device)

    return model, identifyer


def test_model(model, test_loader, loss_function, scaler, historical_information,device):
    predictions = predict_in_batches(model, test_loader, device,batch_size=16)
    return predictions
"""
    This function was a experiance how well the predictions are for multiple days basesd on predicted values
"""
@DeprecationWarning
def iterative_prediction(model, starting_sequance, device, num_predictions = 30):
    model.eval()
    current_sequence = starting_sequance.clone().detach()
    predictions = []

    with torch.no_grad():
        for _ in range(num_predictions):
            # Predict the next step
            prediction = predict_in_batches(model, current_sequence, batch_size=16)

            # print the prediction
            print(prediction)

            prediction_tensor = torch.from_numpy(prediction[0]).type_as(current_sequence).to(device)

            # Update the sequence with the new prediction
            current_sequence = torch.roll(current_sequence, -1, 1)
            current_sequence[:, -1, :] = prediction_tensor.squeeze(0)

            # Store the prediction
            predictions.append(prediction_tensor.cpu())

    return torch.cat(predictions, dim=0).numpy()


"""
    This function is the main function used in this project in here is the training of the models in parallel but also the ploting
"""
def train_and_save_model_on_data(data_list_array, list_of_scalers, historical_information = 30, num_epochs = 10, learning_rate = 0.001, batch_size = 32, loss_function = nn.MSELoss(), amount_of_training_data = 0.8):
        prepared_data = []

        for i,data in enumerate(data_list_array):

            train_data, test_data, X_train, y_train, X_test, y_test = get_data_for_prediction(data, historical_information, amount_of_training_data)

            prepared_data.append({
                'train_data': train_data,
                'test_data': test_data,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'scaler': list_of_scalers[i],
                'id': get_stock_name(data["Date"][0])
            })

        # Get available GPUs
        num_gpus = torch.cuda.device_count()
        num_gpus = num_gpus if num_gpus > 0 else 1

        list_of_models = []
        model_rdy_list = []

        with Executor(max_workers=5) as executor:
            for i, data in enumerate(prepared_data):
                device = torch.device(f"cuda:{i % num_gpus}" if num_gpus > 0 else "cpu")
                train_loader = torch.utils.data.DataLoader(dataset=data['train_data'], batch_size=batch_size, shuffle=False)
                test_loader = torch.utils.data.DataLoader(dataset=data['test_data'], batch_size=batch_size, shuffle=False)

                list_of_models.append(executor.submit(start_lstm, train_loader, test_loader, learning_rate,num_epochs, loss_function, batch_size, device, data['id']))

            for future in list_of_models:
                model, identifier = future.result()
                torch.save(model.state_dict(), f"models/model_{identifier}.pt")

                # Create plots
                plot_data = next(item for item in prepared_data if item['id'] == identifier)
                X = plot_data['X_test']
                y = plot_data['y_test']
                scaler = plot_data['scaler'] # Assuming the scalers are indexed similarly

                model_device = next(model.parameters()).device

                # Create and save plots
                plot_model(model, scaler, X, y, batch_size, model_device, historical_information, f"plots/plot_{identifier}.png")


"""
    This function is used to save plots for a given moddel
"""
def plot_model(model, scaler, X, y, batch_size, device, historical_information,save_path):
    print("Plotting model batch size:", batch_size, "device:", device, "historical_information:", historical_information)
    predictions = predict_in_batches(model, X, batch_size, device)

    predictions = reverse_normalization(predictions, historical_information,scaler)
    y_reversed = reverse_normalization(y, historical_information, scaler)

    # Create a DataFrame comparing the actual and predicted values
    comparison_df = pd.DataFrame({
        'Actual Close': y_reversed.flatten(),
        'Predicted Close': predictions.flatten()
    })


    # plot the predictions
    plt.plot(y_reversed, label='Actual Close')
    plt.plot(predictions, label='Predicted Close')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Close')
    plt.title('Predictions')
    
    plt.savefig(save_path)
    plt.clf()
    # Return the comparison DataFrame
    return comparison_df


