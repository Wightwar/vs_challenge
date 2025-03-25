import numpy as np
import matplotlib.pyplot as plt

def create_sequences(data, label_feature_column, sequence_length=5):
    """This function creates a sequence of lengt n from given data.

    Args:
        data (ndarray): The raw data where the sequence should be created from.
        label_feature_column (int): Index of the label feature column.
        sequence_length (int): Length of the sequence that should be created. Defaults to 5.

    Returns:
        ndarray: An array containing the input data called x and an array containing the label feature data y.
    """
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length]) # Last 5 hours as input
        y.append(data[i+sequence_length, label_feature_column]) # Next hour as label feature
    
    return np.array(x), np.array(y)

def calc_corr_of_feature(df, features, label_feature):
    """This function calculates the linear correlation between a list of features and the label feature.

    Args:
        df (DataFrame): A data frame containing the values.
        features (list[string]): A list of strings with features.
        label_feature (list[string]): A list of strings with the label feature.
    """
    label_feature_shifted = df[label_feature].shift(-1) # Shift label feature to have next-hour correlation
    for feature in features:
        correlations = df[feature].corr(label_feature_shifted)
        print(feature, "corr:", correlations)

def plot_training_loss(training_loss, validation_loss):
    """This function plots the training and the validation loss over the epochs.

    Args:
        training_loss (ndarray): An array with the training losses.
        validation_loss (ndarray): An array with the validation losses.
    """
    epochs = range(1, len(training_loss) + 1)

    plt.plot(epochs, training_loss, color='blue', linestyle='-', label='Training loss')
    plt.plot(epochs, validation_loss, color='green', linestyle='--', label='Validation loss')
    plt.xlabel('epochs')
    plt.title('Training and validation loss')
    plt.legend()

    #plt.plot(epochs, mean_absolute_error, '--', label='Mean absolute error')

    plt.show()

def plot_power_time(y_test, y_pred=np.array([]), y_baseline_pred=np.array([])):
    """This function plots the power consumption and the predicted power consumption over the time.

    Args:
        y_test (ndarray): An array with the real power consumption values.
        y_pred (ndarray): An array with the predicted power consumption values. Defaults to np.array([]).
        y_baseline_pred (ndarray): An array with the predicted power consumption values by the baseline. Defaults to np.array([]).
    """
    plt.figure(figsize=(10,5))
    plt.plot(y_test, label="Actual Power Consumption", color="blue")
    if y_pred.size != 0:
        plt.plot(y_pred, label="Predicted Power Consumption", color="red", linestyle="dashed")
    if y_baseline_pred.size != 0:
        plt.plot(y_baseline_pred, label="Baseline Predicted Power Consumption", color="green", linestyle="dashed")
    plt.xlabel("Time [hours]")
    plt.ylabel("Power Consumption")
    plt.legend()
    plt.show()