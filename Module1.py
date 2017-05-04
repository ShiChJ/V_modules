# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:39:18 2017

@author: 1
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pylab
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from scipy import stats
plt.style.use('ggplot')

#制表/绘图
def get_plot(output, title):
    fig, ax = plt.subplots(figsize = (8, output.shape[0] * 0.8))
    sns.barplot(y = '数据名称', x = '覆盖率(%)', data = output, color = 'blue', alpha = 0.5, ax = ax)
    ax.set_xlabel('覆盖率(%)')
    ax.set_title(title)
    fig.tight_layout()
    pylab.savefig('覆盖率.png')

def get_coverage_plot(data_file, source, field_list_zh = None):
    #检查文件格式是否为csv或xlsx
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.xlsx'):
        df = pd.read_excel(data_file)
    else:
        print('File format is not correct.')
        return None
    #字段列表（除用户ID和标签）
    field_list = df.columns[1:-1]
    #用户ID字段
    id_field = df.columns[0]
    coverage_rates = []
    #以用户ID做GROUP_BY并计数
    grouped_df = df.groupby(id_field).count()
    #对每个字段
    for field in field_list:
        #计算覆盖率（group_df中计数不为0的数据行数量/总的数据行数量
        coverage_rates.append(round(grouped_df.ix[grouped_df[field] != 0, field].count() * 1.0 / grouped_df.shape[0] * 100, 2))
    cols = []
    if field_list_zh == None:
        cols = ['产品名称', '数据名称', '覆盖率(%)']
        output_df = pd.DataFrame({'产品名称':source, '数据名称':field_list, '覆盖率(%)':coverage_rates})
        output_df[cols].to_csv('覆盖率.csv')
        get_plot(output_df[cols], source)
        return output_df[cols]
    else:
        cols = ['产品名称', '数据名称', '含义', '覆盖率(%)']
        output_df = pd.DataFrame({'产品名称':source, '数据名称':field_list, '含义':field_list_zh, '覆盖率(%)':coverage_rates})
        output_df[cols].to_csv('覆盖率.csv')
        get_plot(output_df[cols], source)
        return output_df[cols]

###辅助函数
#根据flag分割数据
def seperate_data(df, flag, metric):
    #返回(好样本，坏样本)
    return (df[df[flag] == 1], df[df[flag] == 0])
#计算KS值
def compute_ks_value(lst_1, lst_2):
    return round(stats.ks_2samp(lst_1, lst_2)[0], 2)

#计算连续字段的好坏划分阈值
def get_seperate_point(xy1, xy2):
    #好客户
    df1 = pd.DataFrame(xy1)
    #坏客户
    df2 = pd.DataFrame(xy2)


    # print("sepation good: {}".format(df1.ix[seperation, 0]))
    # print("sepation bad: {}".format(df2.ix[seperation, 0]))
    print(df1)
    print(df2)
    if df1[0].min() >= df2[0].max():
        return (df1[0].min(), '>:good')
    if df1[0].max() < df2[0].min():
        return (df1[0].max(), '<:good')
    seperation = round(df1.ix[(df1[1] - df2[1]).apply(lambda x: abs(x)).argmax()][0], 0)
    if df1.ix[seperation, 0] > df2.ix[seperation, 0]:
        return (seperation, '>:good')
    if df1.ix[seperation, 0] <= df2.ix[seperation, 0]:
        return (seperation, '<:good')
###

def get_rates(data_file, source, field_list_zh = None):
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.xlsx'):
        df = pd.read_excel(data_file)
    else:
        print('File format is not correct.')
        return None
    field_list = df.columns[1:-1]
    label = df.columns[-1]
    recog_rates = []
    injure_rates = []
    ks_values = []
    for field in field_list:
        #删除该变量存在缺失值的行
        df_c = df.dropna(subset = [field])
        #如果该字段只有两个唯一值，即为0/1变量
        if len(df_c[field].unique()) <= 2:
            #计算识别率
            recog_rates.append(round(sum((df_c[field] == 0) & (df_c[label] == 0)) * 1.0 / sum(df_c[label] == 0) * 100, 2))
            #计算误伤率
            injure_rates.append(round(sum((df_c[field] == 0) & (df_c[label] == 1)) * 1.0 / sum(df_c[label] == 1) * 100, 2))
            #不计算KS值，添加空值
            ks_values.append(np.nan)
        else:
            fig, ax = plt.subplots(figsize = (8, 4))
            #将数据根据标签分成2部分，便于计算KS值
            good_part, bad_part = seperate_data(df_c, label, field)
            ks_values.append(compute_ks_value(good_part[field], bad_part[field]))
            ax.hist(bad_part[field], 1000, normed=1, histtype='step',
                                   cumulative=True, label='坏样本分布', color = 'blue')
            ax.hist(good_part[field], 1000, normed=1, histtype='step',
                                   cumulative=True, label='好样本分布', color = 'red')
            ax.legend(loc = 'upper left')
            ax.set_xlim(0, min([bad_part[field].max(), good_part[field].max()])- 50)
            ax.set_xlabel(field)
            ax.set_ylabel('累计百分比')
            ax.set_title('累计分布曲线')
            #得到好客户曲线的X,Y值
            xy1 = ax.get_children()[1].xy
            #得到坏客户曲线的X,Y值
            xy2 = ax.get_children()[0].xy
            #计算分割点
            seperation, condition = get_seperate_point(xy1, xy2)
            print(seperation)
            df_new = df_c
            #计算预测标签
            if condition == '>:good':
                df_new['predict'] = df_new[field] > seperation
            else:
                df_new['predict'] = df_new[field] < seperation
            #计算识别率和误伤率
            recog_rates.append(round(sum((df_new['predict'] == 0) & (df_new[label] == 0)) * 1.0 / sum(df_new[label] == 0) * 100, 2))
            injure_rates.append(round(sum((df_new['predict'] == 0) & (df_new[label] == 1)) * 1.0 / sum(df_new[label] == 1) * 100, 2))
    cols = []
    if field_list_zh == None:
        cols = ['产品名称', '数据名称', 'KS值', '坏客户识别率(%)', '好客户误伤率(%)']
        output_df = pd.DataFrame({'数据名称':field_list, '产品名称':source, 'KS值':ks_values,
                                  '坏客户识别率(%)':recog_rates, '好客户误伤率(%)':injure_rates})
    else:
        cols = ['产品名称', '数据名称', '含义', 'KS值', '坏客户识别率(%)', '好客户误伤率(%)']
        output_df = pd.DataFrame({'数据名称':field_list, '产品名称':source, '含义':field_list_zh, 'KS值':ks_values,
                                  '坏客户识别率(%)':recog_rates, '好客户误伤率(%)':injure_rates})
    output_df[cols].to_csv('识别_误伤.csv')
    return output_df[cols]


if len(sys.argv) == 3:
    get_coverage_plot(sys.argv[1], sys.argv[2])
    get_rates(sys.argv[1], sys.argv[2])
else:
    get_coverage_plot(sys.argv[1], sys.argv[2], sys.argv[3:])
    get_rates(sys.argv[1], sys.argv[2], sys.argv[3:])