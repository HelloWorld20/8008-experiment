import numpy as np
import torch
x_list = []
for i in range(1884):
    x_list.append(np.random.rand(28, 1))

print("Step 1")
x_np = np.array(x_list)
print("Step 2")
try:
    dataX = torch.tensor(x_np, dtype=torch.float32)
    print("Step 3")
except Exception as e:
    print(f"Error: {e}")
