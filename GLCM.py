
import cv2
from numpy import *
from sklearn.externals import joblib
import glob
from sklearn import  svm
import os
import numpy as np
import re

#定义最大灰度级数
gray_level = 16
glcm_train_data = []


train_label = np.zeros((6))
def path(INPUT_DATA):
    train = []


    # sub_dirs用于存储INPUT_DATA下的全部子文件夹目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
    is_root_dir = True

    count = 1
    x = 0
    h = -1
    # 循环计数器
    # 对每个在sub_dirs中的子文件夹进行操作
    for sub_dir in sub_dirs:


        # 直观上感觉这个条件结构是多此一举，暂时不分析为什么要加上这个语句
        if is_root_dir:
            is_root_dir = False
            continue    # 继续下一轮循环，下一轮就无法进入条件分支而是直接执行下列语句


        print("开始读取第%d类图片：" % count)
        count += 1
        h+=1
        print("---"*30)
        #mkpath=("../save/"+str(count))
        # 调用函数
        #mkdir(mkpath)

        # 获取一个子目录中所有的图片文件
        #extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']     # 列出所有扩展名
        file_list = []
        # os.path.basename()返回path最后的文件名。若path以/或\结尾，那么就会返回空值
        dir_name = os.path.basename(sub_dir)    # 返回子文件夹的名称（sub_dir是包含文件夹地址的串，去掉其地址，只保留文件夹名称）

        # 针对不同的扩展名，将其文件名加入文件列表
        # for extension in extensions:
        #     # INPUT_DATA是数据集的根文件夹，其下有五个子文件夹，每个文件夹下是一种花的照片；
        #     # dir_name是这次循环中存放所要处理的某种花的图片的文件夹的名称
        #     # file_glob形如"INPUT_DATA/dir_name/*.extension"
        file_glob = os.path.join(INPUT_DATA,dir_name, '*.' + 'JPG')


            # extend()的作用是将glob.glob(file_glob)加入file_list
            # glob.glob()返回所有匹配的文件路径列表,此处返回的是所有在INPUT_DATA/dir_name文件夹中，且扩展名是extension的文件
        file_list.extend(glob.glob(file_glob))

        for i in file_list:

            str = re.compile(r'\\')
            a = str.sub('/',i)
            b = a[10:]

            print(a)

            d=teste(a)
            glcm_train_data.append(d)
            train_label[x]=count-2
            x +=1

    return  train
def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
   # print(height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)

    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm


def teste(image_name):
    img = cv2.imread(image_name)
    try:
        img_shape = img.shape
    except:
        print('imread error')
        return

    img = cv2.resize(img, (img_shape[1] // 2, img_shape[0] // 2), interpolation=cv2.INTER_CUBIC)

    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0 = getGlcm(src_gray, 1, 0)
    glcm_1=getGlcm(src_gray, 0,1)
    glcm_2=getGlcm(src_gray, 1,1)
    #glcm_3 = getGlcm(src_gray, -1, 1)

    asm0, con0, eng0, idm0 = feature_computer(glcm_0)
    asm1, con1, eng1, idm1 = feature_computer(glcm_1)
    asm2, con2, eng2, idm2 = feature_computer(glcm_2)
    #asm3, con3, eng3, idm3 = feature_computer(glcm_3)

    return [ asm0, con0, eng0, idm0 , asm1, con1, eng1, idm1, asm2, con2, eng2, idm2]


if __name__ == '__main__':
    print("----------")
    img = path('./train2700')
    #result = test(img)
    print(glcm_train_data)
    print("-"*50)
    print(train_label)
    classifier = svm.SVC(C=5000, kernel='rbf', gamma=12,decision_function_shape='ovr',probability=True)  # C是惩罚系数，即对误差的宽容度;C越小，容易欠拟合。C过大或过小，泛化能力变差;
    # gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。gamma值越大映射的维度越高，训练的结果越好，但是越容易引起过拟合，即泛化能力低。
    classifier.fit(glcm_train_data, train_label.ravel())
    joblib.dump(classifier, "glcm_model.ckpt")

#测试
# gray_level = 16
# test_data = []
#
#
# test_label = np.zeros((300))
# def path(INPUT_DATA):
#     test = []
#
#
#     # sub_dirs用于存储INPUT_DATA下的全部子文件夹目录
#     sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
#     is_root_dir = True
#
#     count = 1
#     x = 0
#     h = -1
#     # 循环计数器
#     # 对每个在sub_dirs中的子文件夹进行操作
#     for sub_dir in sub_dirs:
#
#
#         # 直观上感觉这个条件结构是多此一举，暂时不分析为什么要加上这个语句
#         if is_root_dir:
#             is_root_dir = False
#             continue    # 继续下一轮循环，下一轮就无法进入条件分支而是直接执行下列语句
#
#
#         print("开始读取第%d类图片：" % count)
#         count += 1
#         h+=1
#         print("---"*30)
#         #mkpath=("../save/"+str(count))
#         # 调用函数
#         #mkdir(mkpath)
#
#         # 获取一个子目录中所有的图片文件
#         #extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']     # 列出所有扩展名
#         file_list = []
#         # os.path.basename()返回path最后的文件名。若path以/或\结尾，那么就会返回空值
#         dir_name = os.path.basename(sub_dir)    # 返回子文件夹的名称（sub_dir是包含文件夹地址的串，去掉其地址，只保留文件夹名称）
#
#         # 针对不同的扩展名，将其文件名加入文件列表
#         # for extension in extensions:
#         #     # INPUT_DATA是数据集的根文件夹，其下有五个子文件夹，每个文件夹下是一种花的照片；
#         #     # dir_name是这次循环中存放所要处理的某种花的图片的文件夹的名称
#         #     # file_glob形如"INPUT_DATA/dir_name/*.extension"
#         file_glob = os.path.join(INPUT_DATA,dir_name, '*.' + 'JPG')
#
#
#             # extend()的作用是将glob.glob(file_glob)加入file_list
#             # glob.glob()返回所有匹配的文件路径列表,此处返回的是所有在INPUT_DATA/dir_name文件夹中，且扩展名是extension的文件
#         file_list.extend(glob.glob(file_glob))
#
#         for i in file_list:
#
#             str = re.compile(r'\\')
#             a = str.sub('/',i)
#             b = a[10:]
#
#             print(a)
#
#             d=teste(a)
#             test_data.append(d)
#             test_label[x]=count-2
#             x +=1
#
#
#     return  test
# def maxGrayLevel(img):
#     max_gray_level = 0
#     (height, width) = img.shape
#    # print(height, width)
#     for y in range(height):
#         for x in range(width):
#             if img[y][x] > max_gray_level:
#                 max_gray_level = img[y][x]
#     return max_gray_level + 1
#
#
# def getGlcm(input, d_x, d_y):
#     srcdata = input.copy()
#     ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
#     (height, width) = input.shape
#
#     max_gray_level = maxGrayLevel(input)
#
#     # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
#     if max_gray_level > gray_level:
#         for j in range(height):
#             for i in range(width):
#                 srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level
#
#     for j in range(height - d_y):
#         for i in range(width - d_x):
#             rows = srcdata[j][i]
#             cols = srcdata[j + d_y][i + d_x]
#             ret[rows][cols] += 1.0
#
#     for i in range(gray_level):
#         for j in range(gray_level):
#             ret[i][j] /= float(height * width)
#
#     return ret
#
#
# def feature_computer(p):
#     Con = 0.0
#     Eng = 0.0
#     Asm = 0.0
#     Idm = 0.0
#     for i in range(gray_level):
#         for j in range(gray_level):
#             Con += (i - j) * (i - j) * p[i][j]
#             Asm += p[i][j] * p[i][j]
#             Idm += p[i][j] / (1 + (i - j) * (i - j))
#             if p[i][j] > 0.0:
#                 Eng += p[i][j] * math.log(p[i][j])
#     return Asm, Con, -Eng, Idm
#
#
# def teste(image_name):
#     img = cv2.imread(image_name)
#     try:
#         img_shape = img.shape
#     except:
#         print('imread error')
#         return
#
#     img = cv2.resize(img, (img_shape[1] // 2, img_shape[0] // 2), interpolation=cv2.INTER_CUBIC)
#
#     src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     glcm_0 = getGlcm(src_gray, 1, 0)
#     glcm_1=getGlcm(src_gray, 0,1)
#     glcm_2=getGlcm(src_gray, 1,1)
#     #glcm_3=getGlcm(src_gray, -1,1)
#
#     asm0, con0, eng0, idm0 = feature_computer(glcm_0)
#     asm1, con1, eng1, idm1 = feature_computer(glcm_1)
#     asm2, con2, eng2, idm2 = feature_computer(glcm_2)
#
#     return [asm0, con0, eng0, idm0,asm1, con1, eng1, idm1,asm2, con2, eng2, idm2]
#
#
# def acc(acc1,acc2):
#     i = 0
#     j = 0
#     h = 0
#     while i<len(acc1) and j<len(acc2):
#
#         if acc1[i] == acc2[j]:
#             h += 1
#             i += 1
#             j += 1
#         else:
#             i += 1
#             j += 1
#     l = len(acc1)
#     acc = h / l
#
#     return acc
#
# if __name__ == '__main__':
#     print("----------")
#     img = path('./test300')
#     #result = test(img)
#     print(test_data)
#     print("-"*50)
#     print('test_label')
#     print(test_label)
#
#
#
# clf = joblib.load("glcm_model.ckpt")
#
# predict_gailv = clf.predict_proba(test_data)
# predict_gailv =predict_gailv.tolist()
# list=[]
# for index in range(len(predict_gailv)):
#         predict = max(predict_gailv[index])
#         label = predict_gailv[index].index(max(predict_gailv[index]))
#         list.append(label)
# shibiejieguo= array(list)
# test_label=np.array(test_label,dtype=np.int)
# print('识别标签')
# print(shibiejieguo)
#
# glcmgailv=acc(shibiejieguo,test_label)
# print('*'*80)
# print('测试集准确率')
# print(glcmgailv)