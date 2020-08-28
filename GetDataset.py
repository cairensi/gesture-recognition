# Author by CRS-club and wizard

import os
import numpy as np
import cv2
import sys
import pickle

def GetJpg():
    video_dataset_path = 'dataset'
    label_name = os.listdir(video_dataset_path)
    label_dir = {}
    index = 0
    for i in label_name:
        if i.startswith('.'):
            continue
        label_dir[i] = index
        index += 1
        video_src_path = os.path.join(video_dataset_path, i)
        video_save_path = os.path.join(video_dataset_path, i) + 'Jpg'
        if not os.path.exists(video_save_path):
            os.mkdir(video_save_path)

        videos = os.listdir(video_src_path)
        # 过滤出mp4文件
        videos = filter(lambda x: x.endswith('mp4'), videos)

        for each_video in videos:

            each_video_name, _ = each_video.split('.')
            if not os.path.exists(video_save_path + '/' + each_video_name):
                os.mkdir(video_save_path + '/' + each_video_name)

            each_video_save_full_path = os.path.join(video_save_path, each_video_name) + '/'

            each_video_full_path = os.path.join(video_src_path, each_video)

            cap = cv2.VideoCapture(each_video_full_path)
            frame_count = 1
            success = True
            while success:
                success, frame = cap.read()
                params = []
                params.append(1)
                if success:
                    cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame, params)

                frame_count += 1
            cap.release()
    np.save('label.npy', label_dir)
    print(label_dir)

def GetPkl():
    label_dic = np.load('label.npy', allow_pickle=True).item()
    print(label_dic)

    source_dir = 'dataset'
    target_train_dir = 'dataset/train'
    target_test_dir = 'dataset/test'
    target_val_dir = 'dataset/val'
    if not os.path.exists(target_train_dir):
        os.mkdir(target_train_dir)
    if not os.path.exists(target_test_dir):
        os.mkdir(target_test_dir)
    if not os.path.exists(target_val_dir):
        os.mkdir(target_val_dir)

    for key in label_dic:
        each_mulu = key + 'Jpg'
        print(each_mulu, key)

        label_dir = os.path.join(source_dir, each_mulu)
        label_mulu = os.listdir(label_dir)
        tag = 1
        for each_label_mulu in label_mulu:
            image_file = os.listdir(os.path.join(label_dir, each_label_mulu))
            image_file.sort()
            image_name = image_file[0][:-6]
            image_num = len(image_file)
            frame = []
            vid = image_name
            for i in range(image_num):
                image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_name + '_' + str(i+1) + '.jpg')
                frame.append(image_path)

            output_pkl = vid + '.pkl'

            if tag == 9:
                output_pkl = os.path.join(target_test_dir, output_pkl)
            elif tag == 10:
                output_pkl = os.path.join(target_val_dir, output_pkl)
            else:
                output_pkl = os.path.join(target_train_dir, output_pkl)
            tag += 1
            f = open(output_pkl, 'wb')
            pickle.dump((vid, label_dic[key], frame), f, -1)
            f.close()

def GetTxt():
    data_dir = 'dataset/'

    train_data = os.listdir(data_dir + 'train')
    train_data = [x for x in train_data if not x.startswith('.')]
    print(len(train_data))

    test_data = os.listdir(data_dir + 'test')
    test_data = [x for x in test_data if not x.startswith('.')]
    print(len(test_data))

    val_data = os.listdir(data_dir + 'val')
    val_data = [x for x in val_data if not x.startswith('.')]
    print(len(val_data))

    f = open('dataset/train.txt', 'w')
    for line in train_data:
        f.write(data_dir + 'train/' + line + '\n')
    f.close()
    f = open('dataset/test.txt', 'w')
    for line in test_data:
        f.write(data_dir + 'test/' + line + '\n')
    f.close()
    f = open('dataset/val.txt', 'w')
    for line in val_data:
        f.write(data_dir + 'val/' + line + '\n')
    f.close()

if __name__=='__main__':
    GetJpg()
    GetPkl()
    GetTxt()
