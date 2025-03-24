import numpy as np
import matplotlib.pyplot as plt

def create_sequences(data, target_column, sequence_length=5):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, target_column])
    
    return np.array(x), np.array(y)

def calc_corr_of_feature(df, features, target):
    target_shifted = df[target].shift(-1) # Shift target to have next-hour correlation
    for feature in features:
        correlations = df[feature].corr(target_shifted)
        print(feature, "corr:", correlations)

def plot_training_loss(training_loss, validation_loss):
    epochs = range(1, len(training_loss) + 1)

    plt.plot(epochs, training_loss, 'bo', label='Training loss')
    plt.plot(epochs, validation_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    #plt.plot(epochs, mean_absolute_error, '--', label='Mean absolute error')

    plt.show()

def plot_power_time(y_test, y_pred=np.array([]), y_baseline_pred=np.array([])):
    plt.figure(figsize=(10,5))
    plt.plot(y_test, label="Actual Power Consumption", color="blue")
    if y_pred.size != 0:
        plt.plot(y_pred, label="Predicted Power Consumption", color="red", linestyle="dashed")
    if y_baseline_pred.size != 0:
        plt.plot(y_baseline_pred, label="Baseline Predicted Power Consumption", color="green", linestyle="dashed")
    plt.xlabel("Time")
    plt.ylabel("Power Consumption")
    plt.legend()
    plt.show()