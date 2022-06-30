# 将读取某个传感器中某一段范围交通流数据，以时间为x轴作论文科研风格的二维图表
import pandas as pd # 用于处理表格数据
import numpy as np # 用于科学计算
import matplotlib.pyplot as plt  # 绘图的核心库
from matplotlib.font_manager import FontProperties # 字体属性管理器，知道就好
plt.style.use(['science','no-latex'])

def model(x, p):
   return x ** (2 * p + 1) / (1 + x ** (2 * p))

x = np.linspace(0.75, 1.25, 201)

def paint():
    with plt.style.context(['science','no-latex']):
        fig, ax = plt.subplots()
        for p in [10, 15, 20, 30, 50, 100]:
            ax.plot(x, model(x, p), label=p)
        ax.legend(title='Order')
        ax.set(xlabel='Voltage (mV)')
        ax.set(ylabel='Current ($\mu$A)')
        ax.autoscale(tight=True)
        # fig.savefig('figures/fig1.pdf')
       
        fig.savefig('figures/fig1.jpg', dpi=300)

paint()