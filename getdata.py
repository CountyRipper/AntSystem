from typing import get_type_hints
import numpy as np
# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    np.set_printoptions(suppress=True)
    #position = np.array(data)[:,1:]
    #print(position)
    return data
data = read_tsp("./berlin52.tsp")
# for i in data:
#     i[0] = i[0]-1
# num = len(data)
# print(type(data[0][1]))
# #dist = [[0 for i in range(num)] for j in range(num)]
# dist = np.zeros((num,num)) # 距离矩阵
# for i in range(num):
#     for j in range(num):
#         dist[i][j] = np.sqrt(pow((data[i][1]-data[j][1]),2))
# print(dist)


    