# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:35:47 2017

@author: 1
"""
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pylab
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.style.use('ggplot')
from scipy import stats

# if len(sys.argv == 1):
#     print('Please check your input.')
# elif sys.argv[1].endswith('.xlsx'):
#     df = pd.read_excel(sys.argv[1])
# elif sys.argv[1].endswith('.csv'):
#     df = pd.read_csv(sys.argv[1])
warnings.filterwarnings('ignore')
###辅助函数
def get_separate_point(xy1, xy2):
    #好客户
    df1 = pd.DataFrame(xy1)
    #坏客户
    df2 = pd.DataFrame(xy2)
    # print("sepation good: {}".format(df1.ix[separation, 0]))
    # print("sepation bad: {}".format(df2.ix[separation, 0]))
    # print(df1)
    # print(df2)
    if df1[0].min() >= df2[0].max():
        return (df1[0].min(), '>:good')
    if df1[0].max() < df2[0].min():
        return (df1[0].max(), '<:good')
    separation = round(df1.ix[(df1[1] - df2[1]).apply(lambda x: abs(x)).argmax()][0], 0)
    if df1.ix[separation, 0] > df2.ix[separation, 0]:
        return (separation, '>:good')
    if df1.ix[separation, 0] <= df2.ix[separation, 0]:
        return (separation, '<:good')
def get_rates(df, flag, predict):
    #返回(命中率， 误伤率)
    return (round(sum((df[predict] == 0) & (df[flag] == 0)) * 1.0 / sum(df[flag] == 0) * 100, 2),
            round(sum((df[predict] == 0) & (df[flag] == 1)) * 1.0 / sum(df[flag] == 1) * 100, 2))
def separate_data(df, flag, metric):
    #返回(好样本，坏样本)
    return (df[df[flag] == 1], df[df[flag] == 0])
def compute_ks_value(lst_1, lst_2):
    return round(stats.ks_2samp(lst_1, lst_2)[0], 2)
###

###概率分布曲线
def plot_PDF(df):
    flag = df.columns[2]
    metric = df.columns[1]
    good_part, bad_part = separate_data(df, flag, metric)
    fig, ax = plt.subplots(figsize = (8, 4))
    sns.distplot(good_part[metric], hist = False, kde_kws = {'shade': True}, color = 'red', ax = ax, label = '好样本分布')
    sns.distplot(bad_part[metric], hist = False, kde_kws = {'shade': True}, color = 'blue', ax = ax, label = '坏样本分布')
    ax.set_xlim(0)
    ax.set_xlabel(metric)
    ax.set_ylabel('比例')
    ax.set_title('分布曲线')
    pylab.savefig('分布曲线.png')
    

###累积分布曲线、KS值、分界点
def plot_CDF(df, threshold = 0.0005):
    flag = df.columns[2]
    metric = df.columns[1]
    fig, ax = plt.subplots(figsize = (8, 4))
    good_part, bad_part = separate_data(df, flag, metric)
    ax.hist(good_part[metric], 1000, normed=1, histtype='step',
                           cumulative=True, label='好样本分布', color = 'blue')
    ax.hist(bad_part[metric], 1000, normed=1, histtype='step',
                           cumulative=True, label='坏样本分布', color = 'red')
    ax.legend(loc = 'upper left')
    ax.set_xlim(0, max([bad_part[metric].max(), good_part[metric].max()])- 1)
    #删除最后一个点
    ax.get_children()[0].xy = ax.get_children()[0].xy[:-1]
    ax.get_children()[1].xy = ax.get_children()[1].xy[:-1]
    ax.set_xlabel(metric)
    ax.set_ylabel('累计百分比')
    ax.set_title('累计分布曲线', position = [0.5, 1.05])
    xy1 = ax.get_children()[0].xy
    xy2 = ax.get_children()[1].xy
    separation, condition = get_separate_point(xy1, xy2)
    min_injure_num = int(round(good_part.shape[0] * threshold, 0))
    if condition == '>:good':
        new_separation = good_part.sort_values(metric)[metric].iloc[min_injure_num]
    else:
        new_separation = good_part.sort_values(metric)[metric].iloc[good_part.shape[0] - min_injure_num]
    ax.axvline(x = separation + 0.01, color = 'red')
    x_bounds = ax.get_xlim()
    ax.annotate(s='ks最大分界点(x = {})'.format(separation), xy =(((separation-x_bounds[0])/(x_bounds[1]-x_bounds[0])),1.01), 
            xycoords='axes fraction', verticalalignment='right', horizontalalignment='right bottom')# , rotation = 270)
    ax.axvline(x = new_separation + 0.01, color = 'blue')
    ax.annotate(s='最小误伤分界点(x = {})'.format(new_separation), xy =(((new_separation-x_bounds[0])/(x_bounds[1]-x_bounds[0])),1.01), 
            xycoords='axes fraction', verticalalignment='right', horizontalalignment='right bottom')# , rotation = 270)
    pylab.savefig('累计概率分布.png')
    return (separation, condition)

def compute_metrics(df, separation, condition):
    #如果指标>separation，则定义为好客户
    flag = df.columns[2]
    metric = df.columns[1]
    if condition == '>:good':
        df['predict'] = df[metric] > separation
    else:
        df['predict'] = df[metric] < separation
    good_part, bad_part = separate_data(df, flag, metric)
    values = []
    values.append(compute_ks_value(good_part[metric], bad_part[metric]))
    values.append(separation)
    for v in get_rates(df, flag, 'predict'):
        values.append(v)
    cols = ['数据量', '好客户数量', '坏客户数量', 'KS值', '分界点', '坏命中率(%)', '好误伤率(%)']
    output = pd.DataFrame({'数据量':df.shape[0], '好客户数量':good_part.shape[0], '坏客户数量':bad_part.shape[0], 'KS值': values[0],
                         '分界点':values[1], '坏命中率(%)':values[2], '好误伤率(%)':values[3]}, index = ['-'])
    return output[cols]

def compute_metrics_threshold(df, condition, threshold = 0.0005):
    flag = df.columns[2]
    metric = df.columns[1]
    good_part, bad_part = separate_data(df, flag, metric)
    min_injure_num = int(round(good_part.shape[0] * threshold, 0))
    new_separation = 0
    if condition == '>:good':
        new_separation = good_part.sort_values(metric)[metric].iloc[min_injure_num]
    else:
        new_separation = good_part.sort_values(metric)[metric].iloc[good_part.shape[0] - min_injure_num]
    new_metrics = compute_metrics(df, new_separation, condition)
    new_metrics['KS值'] = np.nan
    return new_metrics

#separation = plot_CDF(df, 'flag', 'c.dated')
df = sys.argv[1]
if df.endswith('.xlsx'):
    df = pd.read_excel(df)
else:
    df = pd.read_csv(df)

metric = df.columns[1]
flag = df.columns[2]
if len(df[metric].unique()) > 2:
    plot_PDF(df)
    separation, condition = plot_CDF(df)
    out1 = compute_metrics(df, separation, condition)
    out2 = None
    if len(sys.argv) == 3:
        out2 = compute_metrics_threshold(df, condition, float(sys.argv[2]))
    else:
        out2 = compute_metrics_threshold(df, condition)
    output = pd.concat([out1, out2])
    output.to_csv('output.csv')
else:
    hit_rate, injure_rate = get_rates(df, 'label', 'score')
    data_v = df.shape[0]
    cols = ['数据量', '好客户数量', '坏客户数量', '坏命中率(%)', '好误伤率(%)']
    output = pd.DataFrame({'数据量': [data_v], '好客户数量':df[df[flag] == 1].shape[0], '坏客户数量':df[df[flag] == 0].shape[0],
                           '坏命中率(%)':[hit_rate], '好误伤率(%)':[injure_rate]})
    output[cols].to_csv('output.csv') 

