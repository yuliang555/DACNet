import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("/home/yl/datasets/ETTh1.csv").iloc[:, 1:4].values


plt.plot(data[1048:1078, 0] + 5, linewidth=3, color='#383838')
plt.plot(data[1048:1078, 1] - 5, linewidth=3, color='#686868')
plt.plot(data[1048:1078, 2], linewidth=3, color='#A0A0A0')

# plt.xticks([])
# plt.yticks([])

plt.savefig('3.png')  


