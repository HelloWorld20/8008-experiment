import pandas as pd
import numpy as np

# Create mock data
n = 1000
df = pd.DataFrame({
    'ID': range(1, n + 1),
    'Difficulty Level': np.random.uniform(0.1, 0.9, n),
    'Chapter_ID': np.random.randint(1, 11, n),
    'Group': np.random.choice([1, 2, 3, 4, np.nan], n, p=[0.1, 0.1, 0.1, 0.1, 0.6])
})
df.to_csv('/Users/leon.w/workspace/cityu/8008/CNN+LSTM/Main_1000_2.csv', index=False)
print("Mock data generated at Main_1000_2.csv")
