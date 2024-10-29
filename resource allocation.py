import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})

# 数据
marl = [19.4815, 21.385, 57.7939, 73.0239, 93.9938]
sac = [52.696, 61.384, 72.541, 82.625, 97.685]
labels = ['4 Ves', '5 Ves', '6 Ves', '7 Ves', '8 Ves']  # 柱状图的标签

x = np.arange(len(labels))  # x轴位置
width = 0.35  # 柱状图宽度

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, marl, width, label='Marl')
rects2 = ax.bar(x + width/2, sac, width, label='Sac')

# 添加标签、标题和图例
ax.set_xlabel('The number of vehicles')
ax.set_ylabel('Resource used for computing tasks')
ax.set_title('Resource usage')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 添加数值标签
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
