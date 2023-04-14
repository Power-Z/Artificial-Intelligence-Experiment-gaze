#用于将眼球区域和瞳孔区域的mask生成多边形的代码
#2023-1-31
#张力
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
from matplotlib import image # mpimg 用于读取图片
import numpy as np
import json
from tqdm import *

def mask(name):
    path_json = './dataset/UnityEyes/'+name+'.json'  # json文件路径


    # 读取json文件
    with open(path_json, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        eye_landmark = json_data['interior_margin_2d']#可视眼球区域关键点
        iris_landmark = json_data['iris_2d']#瞳孔区域关键点
        #caruncle_landmark = json_data['caruncle_2d']

        #可视眼球区域
        eye_landmark = [s[1:-1].split(',') for s in eye_landmark]
        eye_landmark_x = [[float(i) for i in s][0] for s in eye_landmark]
        eye_landmark_y = [768-[float(i) for i in s][1] for s in eye_landmark] #matplot和UnityEyes中的坐标原点不一样，需要调整
        eye_landmark_mask = [list(t) for t in zip(eye_landmark_x,eye_landmark_y)]

        #虹膜区域
        iris_landmark = [s[1:-1].split(',') for s in iris_landmark]
        iris_landmark_x = [[float(i) for i in s][0] for s in iris_landmark]
        iris_landmark_y = [768 - [float(i) for i in s][1] for s in iris_landmark]
        iris_landmark_mask = [list(t) for t in zip(iris_landmark_x, iris_landmark_y)]

    mask_eyes = np.zeros([768,1280], dtype="uint8")
    mask_iris = np.zeros([768, 1280], dtype="uint8")

    # path_img = './dataset/demo/'+name+'.jpg'#图片路径
    # mask_eyes = image.imread(path_img) #读取图片


    #通过landmark画出可视眼球区域图像
    """
    这里一定要将区域填充为255，否则后续的处理会有问题
    """
    cv2.polylines(mask_eyes, np.int32([eye_landmark_mask]),True, 255) #画多边形
    cv2.fillPoly(mask_eyes, np.int32([eye_landmark_mask]), 255) #进行多边形填充

    #通过landmark画出虹膜区域图像
    cv2.polylines(mask_iris, np.int32([iris_landmark_mask]),True, 255) #画多边形
    cv2.fillPoly(mask_iris, np.int32([iris_landmark_mask]), 255) #进行多边形填充

    #保存
    #可视眼球区域
    saveFile_eys = './dataset/UnityEyes/'+name+'_eyeshape.jpg'
    cv2.imwrite(saveFile_eys, mask_eyes)
    #虹膜区域
    saveFile_iris = './dataset/UnityEyes/'+name+'_iris.jpg'
    cv2.imwrite(saveFile_iris, mask_iris)



    #显示
    # plt.figure(figsize=(30, 20))  # 设置显示分辨率
    #
    # plt.imshow(mask_iris)  # 显示图片
    # #plt.imshow(mask_iris)  # 显示图片
    # plt.axis('on')  # 不显示坐标轴



for i in tqdm(range(60000)):
    mask(str(i+1))
#mask(str(3))
#plt.show()