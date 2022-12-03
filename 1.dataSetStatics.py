# -*- coding: utf-8 -*-
import os

path = "D:/DataContest/data/0805 (有图数据)/"

# # file_image = 'F:\\car\\object\\image4\\0805 (1)\\0805\\image'
# datanames = os.listdir(path)
# list = []
# for i in datanames:
#     list.append(i)
# already_draw_one = 0
# row_total=0
# for b in list:
#     (filename, extension) = os.path.splitext(b)
#     file = path + b
#     if not os.access(file, os.X_OK):
#         continue;
#     row_total = row_total+len(open(file).readlines())
#
# import os

row_total=0
def fn(path, tail2):
    # key = dict()
    for i in os.listdir(path):
        sub_path = os.path.join(path, i)
        if os.path.isdir(sub_path):  # 递归遍历子目录下文件及目录
            # key.update({i: dict()})  # 父级标签
            # key[i].update(fn(sub_path, tail2))
            fn(sub_path, tail2)
        elif os.path.isfile(sub_path):  # 读取目录下文件
            tail1 = i.split('.')[-1]  # 取出后缀
            # key.update({i: list()})  # 子级标签
            # 读取后缀为txt的目标文件内容
            if tail1 == tail2:
                with open(sub_path, "r", encoding="utf-8") as f:
                    text = f.readlines()
                    global row_total
                    row_total = row_total + len(open(sub_path).readlines())
                # for j in text:
                #     key[i].append(j.strip())
    return #key,row_total


if __name__ == '__main__':
    path = "F:/DataContest/data"
    fn(path, "csv")
    # print(key)
    print(row_total)