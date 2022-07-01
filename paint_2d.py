# 将读取某个传感器中某一段范围交通流数据，以时间为x轴作论文科研风格的二维图表
import pandas as pd # 用于处理表格数据
import numpy as np # 用于科学计算
import matplotlib.pyplot as plt  # 绘图的核心库
from pylab import mpl
# 设置显示中文字体
plt.rcParams['legend.title_fontsize'] = 'small'
import matplotlib.font_manager as fm
# print(fm)
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号
from matplotlib.font_manager import FontProperties # 字体属性管理器，知道就好
# plt.style.use(['science','no-latex','cjk-sc-font'])




def paint(x,y,name,title=''):
    # 绘制单个图
    with plt.style.context(['science','no-latex']):
        fig, ax = plt.subplots()
        ax.plot(x, y, label='origin',)
        ax.legend(title=title,loc=0 ,fontsize=6)
        ax.set(xlabel='time/(5min)')
        ax.set(ylabel='flow/veh')
        ax.autoscale(tight=False)
        # fig.savefig('figures/fig1.pdf')

        fig.savefig('figures/'+name  , dpi=600)

def m_paint():
    # 绘制多个图
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots()