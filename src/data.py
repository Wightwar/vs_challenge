import numpy as np

def create_sequences(data, target_column, sequence_length=5):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, target_column])
    
    return np.array(x), np.array(y)