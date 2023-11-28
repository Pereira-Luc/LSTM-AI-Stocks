from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from func.dataFunctions import *
import numpy as np
from classes.StockDataset import StockDataset
from classes.LSTM import LSTM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# data_list_array = read_multiple_files_parallel(file_paths)

# apple_data = data_list_array[0][0]

# go_back_days = 7

# seq_apple_data = create_sequence(apple_data, go_back_days)

# X = seq_apple_data[:, 1:]
# X = dc(np.flip(X, axis=1))
# y = seq_apple_data[:, 0]


# # Split the data into training and testing sets
# train_size = int(len(X) * 0.8)
# test_size = len(X) - train_size

# X_train = X[:train_size]
# X_test = X[train_size:]

# y_train = y[:train_size]
# y_test = y[train_size:]

# # convert data to tensors
# X_train = torch.tensor(X_train).float()
# y_train = torch.tensor(y_train).float()

# X_test = torch.tensor(X_test).float()
# y_test = torch.tensor(y_test).float()

# # add an additional dimension
# X_train = X_train.reshape(-1, go_back_days, 1)
# X_test = X_test.reshape(-1, go_back_days,1)

# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)


# # convert to StockDataset
# train_data = StockDataset(X_train, y_train)
# test_data = StockDataset(X_test, y_test)

# # create the dataloaders
# batch_size = 16

# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
# test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# print('Create the dataloaders')

# for batch_index, batch in enumerate(train_loader):
#     print(batch[0].size())
#     print(batch[1].size())
#     break


# model = LSTM(input_size=1, hidden_size=4, num_layers=1, output_size=1).to(device)
# model

# def train_one_epoch():
#     model.train(True)
#     print('Training...')
#     running_loss = 0.0

#     for batch_index, batch in enumerate(train_loader):
#         x_batch, y_batch = batch[0].to(device), batch[1].to(device)

#         outputs = model(x_batch)
#         loss = loss_function(outputs, y_batch)
#         running_loss += loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch_index % 100 == 99:
#             avg_loss = running_loss / 100
#             print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_index + 1, avg_loss))
#             running_loss = 0.0



#     print()

# def validate_one_epoch():
#     model.train(False)
#     print('Validating...')
#     running_loss = 0.0

#     for batch_index, batch in enumerate(test_loader):
#         x_batch, y_batch = batch[0].to(device), batch[1].to(device)

#         with torch.no_grad():
#             outputs = model(x_batch)
#             loss = loss_function(outputs, y_batch)
#             running_loss += loss
        
#     avg_loss = running_loss / len(test_loader)

#     print('Avg. Validation Loss: {}'.format(avg_loss))
#     print("--------------------------------------------------------------")
#     print()

# # Training loop
# learning_rate = 0.001
# num_epochs = 10
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     train_one_epoch()
#     validate_one_epoch()


# def predict_in_batches(model, dataset, batch_size=batch_size):
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for i in range(0, len(dataset), batch_size):
#             batch = dataset[i:i+batch_size].to(device)
#             batch_predictions = model(batch).cpu()
#             predictions.append(batch_predictions)
#     return torch.cat(predictions, dim=0).numpy()




# predictions = predict_in_batches(model, X_train, batch_size=batch_size)

# # for i in range(0, len(predictions), 100):
# #     print('Prediction: {}, Actual: {}'.format(predictions[i], y_train[i]))



# print('Predictions: ', predictions.shape)
# plt.plot(y_train, label='Actual Close')
# plt.plot(predictions, label='Predicted Close')
# plt.legend()
# plt.xlabel('Days')
# plt.ylabel('Close')
# plt.title('Predictions')
# plt.show()


# path_apple = "data/AMZN.csv"
# file_paths = [path_apple]

# data_list_array = read_multiple_files_parallel(file_paths)

# apple_data = data_list_array[0][0]

# train_data, test_data, x_train, y_train, x_test, x_train = get_data_for_prediction(apple_data, 7, 0.8)

# # convert to loaders
# batch_size = 16

# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
# test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# # create the model
# model = LSTM(input_size=1, hidden_size=4, num_layers=1, output_size=1).to(device)
# model



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
    model = LSTM(input_size=1, hidden_size=4, num_layers=1, output_size=1).to(device)
    model

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(epoch, optimizer, loss_function,model, train_loader)
        validate_one_epoch(loss_function,model,test_loader)

    return model




