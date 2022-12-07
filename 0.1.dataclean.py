# -*- coding: utf-8 -*-
import getopt
import os
import sys

import pandas as pd
from pandas import Series, DataFrame


def clean(path,targetfile):
    (filename, extension) = os.path.splitext(targetfile)
    df = DataFrame(pd.read_table(path+targetfile))
    # 对列进行重命名
    # df.rename(columns={0: '目标检测', 1: '性别', 2: '年龄', 3: '体重', 4: '身高'}, inplace=True)
    # 对整行为空值的数据进行删除

    df.dropna(how='all', inplace=True)
    # 使用平均值来填充体重缺失的值
    # df[u'体重'].fillna(int(df[u'体重'].mean()), inplace=True)

    # 对身高列的度量做统一，我们使用df.apply方法来统一身高的度量，使用df.columns.str.upper方法将首字母统一为大写
    # # def format_height(df):
    # #     if (df['身高'] < 3):
    # #         return df['身高'] * 100
    # #     else:
    #         return df['身高']

    # df['身高'] = df.apply(format_height, axis=1)
    # 2 姓名首字母大小写不统一，统一成首字母大写
    df.columns = df.columns.str.lower()#   .upper()
    # 对姓名列的非法字符做过滤，我们可以使用df.replace方法，删除字母前面的空格，我们可以使用df.map方法
    # 1、英文字母出现中文->删除非ASCLL码的字符
    # df['姓名'].replace({r'[^\x00-\x7f]+': ''}, regex=True, inplace=True)
    # # 2、英文名字出现了问号->删除问号
    # df['姓名'].replace({r'\?+': ''}, regex=True, inplace=True)
    # # 3、名字前出现空格->删除空格
    # df['姓名'] = df['姓名'].map(str.lstrip)

    # 将年龄列为负值的年龄处理为正数，我们可以使用df.apply方法:
    # def format_sex(df):
    #     return abs(df['年龄'])

    # df['年龄'] = df.apply(format_sex, axis=1)
    # 删除行记录重复的数据，我们可以使用df.drop_duplicates方法:
    df.drop_duplicates()
    # df.drop_duplicates([16], inplace=True) # 时间戳
    # 我们讲清洗好的数据保存至新的excel中，我们可以使用df.to_excel方法:
    df.to_excel(filename+'_clean.xlsx', index=False)


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "wdir=", ])
    for opt, arg in opts:
        if opt == '-h':
            print
            '8.riskhotmap.py -i <inputfile> -w <working directory>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            targetfile = arg
        elif opt in ("-w", "--wdir"):
            path = arg
            if not path.endswith('/'):
                path = path +'/'
    path = 'F:/DataContest/data/0805 (有图数据)/'
    targetfile = '1659670062.6_1659670103.73.csv'
    clean(path,targetfile)