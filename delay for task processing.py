import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})

# 数据
epochs = [4,5,6,7,8]
marl = [0.08325, 0.08593, 0.07319, 0.07845, 0.06951]
sac = [0.18878, 0.1823, 0.16437, 0.15481, 0.14859]
marl_df = [0.08150, 0.085857, 0.0905321, 0.092977, 0.097629]
sac_df = [0.18452, 0.18461, 0.17583, 0.17591, 0.16743]

# 绘图
plt.figure(figsize=(10, 6))

# 绘制折线
plt.plot(epochs, marl, marker='o', linestyle='-', color='r', label='MARL \u0394f=0.2')
plt.plot(epochs, sac, marker='s', linestyle='-', color='b', label='SAC \u0394f=0.2')
plt.plot(epochs, marl_df, marker='o', linestyle='-.', color='r', label='MARL \u0394f=-0.2')
plt.plot(epochs, sac_df, marker='s', linestyle='-.', color='b', label='SAC \u0394f=-0.2')

# 添加标题和标签
plt.title('Comparison of delay for task processing')
plt.xlabel('The number of vehicles')
plt.ylabel('delay for task processing')
plt.xticks(epochs)

# 添加图例
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
