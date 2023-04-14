#!/usr/bin/env python
#用于MSGazeNet的数据处理，注意针对MPIIGaze的数据

import argparse
import pathlib

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io
import tqdm
import os


def convert_pose(vector: np.ndarray) -> np.ndarray:
    rot = cv2.Rodrigues(np.array(vector).astype(np.float32))[0]
    vec = rot[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    return np.array([pitch, yaw]).astype(np.float32)


def convert_gaze(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw]).astype(np.float32)


def get_eval_info(person_id: str, eval_dir: pathlib.Path) -> pd.DataFrame:
    #按所给列表得到验证集各个数据的路径
    eval_path = eval_dir / f'{person_id}.txt'
    df = pd.read_csv(eval_path,
                     delimiter=' ',
                     header=None,
                     names=['path', 'side'])
    df['day'] = df.path.apply(lambda path: path.split('/')[0])
    df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    df = df.drop(['path'], axis=1)#删除path列
    return df


def save_one_person(person_id: str, data_dir: pathlib.Path,
                    eval_dir: pathlib.Path, output_path: pathlib.Path) -> None:
    left_images = dict()
    left_poses = dict()
    left_gazes = dict()
    right_images = dict()
    right_poses = dict()
    right_gazes = dict()
    filenames = dict()
    person_dir = data_dir / person_id #具体的数据路径


    #读取每一个人的数据（,mat文件）
    for path in sorted(person_dir.glob('*')): #
        #读取mat数据
        mat_data = scipy.io.loadmat(path.as_posix(),
                                    struct_as_record=False,
                                    squeeze_me=True)
        data = mat_data['data']

        day = path.stem
        #修改进行左右眼数据区分
        left_images[day] = data.left.image
        left_poses[day] = data.left.pose
        left_gazes[day] = data.left.gaze

        right_images[day] = data.right.image
        right_poses[day] = data.right.pose
        right_gazes[day] = data.right.gaze

        filenames[day] = mat_data['filenames']

        #判断类型并进行类型转换
        if not isinstance(filenames[day], np.ndarray):
            left_images[day] = np.array([left_images[day]])
            left_poses[day] = np.array([left_poses[day]])
            left_gazes[day] = np.array([left_gazes[day]])
            right_images[day] = np.array([right_images[day]])
            right_poses[day] = np.array([right_poses[day]])
            right_gazes[day] = np.array([right_gazes[day]])
            filenames[day] = np.array([filenames[day]])

    df = get_eval_info(person_id, eval_dir)#得到所需数据集路径
    images = []
    poses = []
    gazes = []
    #读取数据集数据，忽略左右眼
    for _, row in df.iterrows():#iterrows返回每一行索引以及这一行本身
        day = row.day
        index = np.where(filenames[day] == row.filename)[0][0]
        if row.side == 'left':
            image = left_images[day][index]
            pose = convert_pose(left_poses[day][index]) #转化为pitch和yaw
            gaze = convert_gaze(left_gazes[day][index]) #转化为pitch和yaw
        else:
            image = right_images[day][index][:, ::-1] #右眼进行翻转
            pose = convert_pose(right_poses[day][index]) * np.array([1, -1]) #右眼的gaze也要随着图片进行翻转
            gaze = convert_gaze(right_gazes[day][index]) * np.array([1, -1]) #翻转
        images.append(image)
        poses.append(pose)
        gazes.append(gaze)

    #存储图片
    image_path = os.path.join(output_path, 'Image/'+str(person_id)+'/')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    for i in range(len(images)):
        saveFile_eys = image_path + str(i+1)+'.jpg'
        cv2.imwrite(saveFile_eys, images[i])

    #存储视线向量和头部姿态向量（txt文件）
    path_txt = os.path.join(output_path, 'Label/'+str(person_id)+'.txt')

    with open(path_txt, 'w', encoding="utf-8") as f:
        f.write(person_id)

    for num in range(len(gazes)):
        data_txt = person_id+'/'+str(num+1)+'.jpg'+' '+str(poses[num][0])+','+str(poses[num][1])+' '+str(gazes[num][0])+','+str(gazes[num][1])
        with open(path_txt, 'a', encoding="utf-8") as f:
            f.write('\n' + data_txt)
            f.close()

    #images = np.asarray(images).astype(np.uint8)
    #poses = np.asarray(poses).astype(np.float32)
    #poses = np.asarray(poses).astype(np.float32)
    #output_path_person = os.path.join(output_path, str(person_id)+'.npz')

    #np.savez(output_path_person, image=images, pose=poses, gaze=poses)

def main():
    #读取命令行参数
    parser = argparse.ArgumentParser()#创建对象
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    args = parser.parse_args()#解析参数

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir
    #if output_path.exists():
        #raise ValueError(f'{output_path} already exists.')

    dataset_dir = pathlib.Path(args.dataset) #原始数据来源路径

    #原始数据处理
    for person_id in tqdm.tqdm(range(15)):
        #tqdm用于产生进度条
        person_id = f'p{person_id:02}'
        data_dir = dataset_dir / 'Data' / 'Normalized' #指定原始数据路径
        eval_dir = dataset_dir / 'Evaluation Subset' / 'sample list for eye image'  #代码所用数据目录存储路径
        save_one_person(person_id, data_dir, eval_dir, output_path) #获取并存储一个人的数据


if __name__ == '__main__':
    main()
