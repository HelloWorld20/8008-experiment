import numpy as np
import torch

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

data = np.random.rand(1900, 1)
x, y = sliding_windows(data, 28)
print(f"Type of x: {type(x)}, dtype: {x.dtype}, shape: {x.shape}")
print("Converting to tensor...")
dataX = torch.tensor(x, dtype=torch.float32)
print("Done")
