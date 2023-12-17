import matplotlib.pyplot as plt
from func.LSTM_pytorch import *

def plot_model(model, scaler, X, y, batch_size, device, historical_information):
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
    plt.show()

    # Return the comparison DataFrame
    return comparison_df


