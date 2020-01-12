from sklearn.externals import joblib
from skimage import feature as skft
from sklearn import  svm
import re
import os
from tqdm import tqdm
import numpy as np
import cv2
from numpy import *
import json
import copy
import tensorflow as tf

train_label=np.zeros( (2700) )
train_data=np.zeros((2700,512,512))

array_img = []
#读取训练集数据
def read_train_image(path):
     print("-" * 50)
     print("训练集读取")
     '''读取路径下所有子文件夹中的图片并存入list'''
     train = []
     dir_counter = 0
     x=0
     i=0
     h=-1

     for child_dir in os.listdir(path):
         child_path = os.path.join(path, child_dir)
         h += 1
         for dir_image in tqdm(os.listdir(child_path)):
             img = cv2.imread(child_path + "\\" + dir_image, cv2.IMREAD_COLOR)
             img=cv2.resize(img,(512,512))
             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             data=np.zeros((512,512))
             data[0:img.shape[0],0:img.shape[1]] = img
             train_data[i, :, :] = data[0:512, 0:512]
             train_label[x] = h

             i += 1
             x += 1

         dir_counter += 1
         train.append(train_label)
         train.append(train_data)

     return train

#读取测试集数据
def read_test_img(path):
     '''读取路径下所有子文件夹中的图片并存入list'''

     dir_counter = 0
     i=0
     x=0
     h = -1
     test=[]
     test_label = np.zeros((300))
     test_data=np.zeros((300,512,512))
     for child_dir in os.listdir(path):
         child_path = os.path.join(path, child_dir)
         h += 1
         for dir_image in tqdm(os.listdir(child_path)):
             array_img.append(dir_image)
             img = cv2.imread(child_path + "\\" + dir_image, cv2.IMREAD_COLOR)
             img=cv2.resize(img,(512,512))
             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             data=np.zeros((512,512))
             data[0:img.shape[0],0:img.shape[1]] = img
             test_data[i, :, :] = data[0:512, 0:512]
             test_label[x] = h
             x +=1
             i += 1
         dir_counter += 1
         test.append(test_label)
         test.append(test_data)
         print(test[1])

     return test

def texture_detect():
    radius = 1
    n_point = radius * 8
    train_hist = np.zeros( (2700,256) )
    test_hist = np.zeros( (300,256) )
    for i in np.arange(2700):
        #使用LBP方法提取图像的纹理特征.
        lbp=skft.local_binary_pattern(train_data[i],n_point,radius,'default')

        #统计图像的直方图
        max_bins = int(lbp.max() + 1)
        #hist size:256
        train_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    for i in np.arange(300):
        lbp = skft.local_binary_pattern(test_data[i],n_point,radius,'default')
        # print("图像识别")
        # print(图像识别)
        # print(图像识别.shape)
        #统计图像的直方图
        max_bins = int(lbp.max() + 1)
        #hist size:256
        test_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    return train_hist,test_hist

list=[]


def liy():
    train_hist, test_hist = texture_detect()

    classifier=svm.SVC(C=5000,kernel='rbf',gamma=20,decision_function_shape='ovr', probability=True)#C是惩罚系数，即对误差的宽容度;C越小，容易欠拟合。C过大或过小，泛化能力变差; gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。gamma值越大映射的维度越高，训练的结果越好，但是越容易引起过拟合，即泛化能力低。
    # print("train_hist")
    # print(train_hist)
    # print("train_hist.shape")
    # print(train_hist.shape)
    classifier.fit(train_hist,train_label.ravel())

    joblib.dump(classifier, "lbp_model.ckpt")
    clf = joblib.load("lbp_model.ckpt")
    predict_gailv = clf.predict_proba(test_hist)
    predict_gailv =predict_gailv.tolist()
    print(predict_gailv)

    for index in range(len(predict_gailv)):
        predict = max(predict_gailv[index])
        label = predict_gailv[index].index(max(predict_gailv[index]))
        list.append(label)


#保存结果
    tlist =[]
    #list1 = []
    #dict = {}
  #  i=0
#     while i< 300 :
#         key=array_img[i]
#         #ret = re.match("[a-zA-Z0-9_]*",key)
#         value =int(test_label[i])
# #[ { "image_id": "prcv2019test05213.jpg", "disease_class":1 }, ...]
#         try:
#
#             dict = {}
#             dict["image_id"] = key
#             dict["disease_class"] =  value
#             dict["test_hist"]=test_hist
#             print("字典输入%d",dict)
#             tlist.append(dict)
#             #print(list)
#             # if i%1 == 0:
#             #     list.append('\n')
#         except Exception as f:
#             print("没有添加")
#         i += 1
#     print("~"*80)
#     with open('图像识别.txt', 'w') as f:
#         for i in list:
#             json_str = json.dumps(i)
#             f.writelines(json_str+"\n")
#     print("~"*80)
#     print(tlist)
#     print("~" * 80)
   # print('训练集：',classifier.score(train_hist,train_label))
def acc(acc1,acc2):
    i = 0
    j = 0
    h = 0
    while i<len(acc1) and j<len(acc2):

        if acc1[i] == acc2[j]:
            h += 1
            i += 1
            j += 1
        else:
            i += 1
            j += 1
    l = len(acc1)
    acc = h / l

    return acc

if __name__ == '__main__':

    image_path='train2700'#训练集路径
    train =read_train_image(image_path)
   # print(train)
    train_label = train[0]
   # print(train_label)
    train_data =  train[1]
    print(train_data)
    print("-" * 50)
    print("测试集读取")
    image_path='test300'#测试集路径
    test=read_test_img(image_path)
    test_label=test[0]
    test_data=test[1]
    liy()
    shibiejieguo= array(list)
    test_label=np.array(test_label,dtype=np.int)
    test_acc = acc(shibiejieguo, test_label)
    print('test_acc:', test_acc)
    print('test_label:',test_label)
   # print('test_label  type:',type(test_label))
    print('list:',shibiejieguo)
    #print('list  type',type(shibiejieguo))









