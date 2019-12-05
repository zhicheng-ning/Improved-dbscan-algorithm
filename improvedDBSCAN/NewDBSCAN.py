import matplotlib.pyplot as plt
import numpy as np
import math


# 导入数据集
def loadDataSet(fileName, splitChar=','):
    dataSet = []
    with open(fileName) as fr:  # 读取完文件数据后自动关闭文件
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))  # 强转为float型列表
            dataSet.append(fltline)
    return dataSet


# 计算距离
def dist(x, y):
    return math.sqrt(math.pow(x[0] - y[0], 2) + math.pow(x[1] - y[1], 2))


# 获得距离矩阵
def getDistanceMat(dataSet):
    num = len(dataSet)
    Dn = np.ones((num, num))
    for i in range(num):
        for j in range(num):
            Dn[i, j] = dist(dataSet[i], dataSet[j])
    print('原始距离矩阵:')
    print(Dn)
    return Dn


# 对距离矩阵的每一列进行升序排序
def doDSORT(Dn):
    num = len(Dn)
    doSort = np.zeros((num, num))
    for i in range(num):
        doSort[:, i] = np.sort(Dn[:, i])
    print('按照每列升序排序后的距离矩阵:')
    print(doSort)
    return doSort


# 专家专家凭经验给出一批适合这批数据点分布特征的MinPtsProfessor参数值
def inputMINPTSPROFESSOR():
    MINPTSPROFESSOR = []
    MINPTSPROFESSOR.extend(list(map(int, input().strip().split(' '))))
    # print('Professor Input 5 number as MinPts:')
    # for i in range(5):
    #     MINPTSPROFESSOR.append(eval(input()))
    print('专家给出的MinPts参考值:')
    print('MINPTSPROFESSOR:', end=' ')
    print(MINPTSPROFESSOR)
    return MINPTSPROFESSOR


# 从MINPTSPROFESSOR中循环取出数据，，从而获得EPSMIN列表
def getEPSMINI(Dsort, MINPTSPROFESSOR):
    EPSMIN = []
    for ni in MINPTSPROFESSOR:
        EpsMini = np.sort(Dsort[ni - 1, :])[0]
        EPSMIN.append(EpsMini)
    print('依据每个MinPts参考值而找到每一行中的最小的EPS值:')
    print('EPSMIN:', end='')
    print(EPSMIN)
    return EPSMIN


def getLESSEPSMINI(Dsort, EPSMIN, MINPTSPROFESSOR):
    num = len(Dsort)
    chooseMinpts = []
    for Eps in EPSMIN:
        sum = 0
        for i in range(num):
            count = 0
            for j in range(num):
                if (Dsort[j, i] <= Eps):
                    count = count + 1
                else:
                    break
            sum = sum + math.fabs(count - MINPTSPROFESSOR[EPSMIN.index(Eps)])
        chooseMinpts.append(sum)
    print('对每个EPS，以每个中心点按照该EPS所选取的MinPts与对应MINPTSPROFESSOR的预测值之差的求和:')
    print('偏差值列表:', end='')
    print(chooseMinpts)

    print(MINPTSPROFESSOR[chooseMinpts.index(np.sort(chooseMinpts)[1])])
    print(EPSMIN[chooseMinpts.index(np.sort(chooseMinpts)[1])])
    return chooseMinpts


dataset = loadDataSet('788points.txt', splitChar=',')
dn = getDistanceMat(dataset)
print('After sorting')
DSORT = doDSORT(dn)
MINPTSPROFESSOR = inputMINPTSPROFESSOR()
EPSMIN = getEPSMINI(DSORT, MINPTSPROFESSOR)
getLESSEPSMINI(DSORT, EPSMIN, MINPTSPROFESSOR)

plt.figure(figsize=(8,6),dpi=80)
plt.scatter(MINPTSPROFESSOR,getLESSEPSMINI(DSORT, EPSMIN, MINPTSPROFESSOR))
plt.show()

# x = []
# y = []
# for data in dn:
#     x.append(data[0])
#     y.append(data[1])
# plt.figure(figsize=(8, 6), dpi=80)
# plt.scatter(x, y, marker='o')
# plt.show()
