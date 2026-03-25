import re
import uuid

py_path = '/Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorch-lstm-deep-learning-with-m5-data.py'

with open(py_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换 plt.show()
show_replacement = """
import os
os.makedirs('plots', exist_ok=True)
import uuid
plt.savefig(f"plots/plot_{uuid.uuid4().hex[:8]}.png")
plt.close()
"""
content = re.sub(r'plt\.show\(\)', show_replacement.strip(), content)

# 增加函数级注释
comments = {
    "def reduce_mem_usage": "def reduce_mem_usage(df, verbose=True):\n    '''\n    减少 pandas DataFrame 的内存占用。\n    遍历所有列，将数值类型向下转换为更小的内存类型（如 int64 转 int8 等），从而优化内存使用。\n    '''",
    "def read_data": "def read_data():\n    '''\n    读取 M5 数据集中的各种 CSV 文件?mport re
import uuid

py_path = '/Users/leon.w/workspace/cityu/8008/CNN+LSTM/learning-pytorcgeimport u??
py_path =???with open(py_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换 plt.show()
show_???   content = f.read()

# 替换 plt.show()
show_replacement =th
# 替换 plt.show()
orcshow_replacement =??import os
os.makedirs?s.maked"import uuid