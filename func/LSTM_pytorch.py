from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from func.dataFunctions import *
import numpy as np
from classes.StockDataset import StockDataset
from classes.LSTM import LSTM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(epoch, optimizer, loss_function, model, train_loader):
    model.train(True)
    print('Training...')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        outputs = model(x_batch)
        loss = loss_function(outputs, y_batch)
        running_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:
            avg_loss = running_loss / 100
            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_index + 1, avg_loss))
            running_loss = 0.0
    print()

def validate_one_epoch(loss_function, model, test_loader):
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



def predict_in_batches(model, dataset, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size].to(device)
            batch_predictions = model(batch).cpu()
            predictions.append(batch_predictions)
    return torch.cat(predictions, dim=0).numpy()


def start_lstm(train_loader, test_loader, learning_rate = 0.001, num_epochs = 10, loss_function = nn.MSELoss(),batch_size = 16):
    # create the model
    model = LSTM(input_size=1, hidden_size=4, num_layers=2, output_size=1).to(device)
    model

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(epoch, optimizer, loss_function,model, train_loader)
        validate_one_epoch(loss_function,model,test_loader)

    return model


def test_model(model, test_loader, loss_function, scaler, historical_information):
    predictions = predict_in_batches(model, test_loader, batch_size=16)
    return predictions


def iterative_prediction(model, starting_sequance, num_predictions = 30):
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
