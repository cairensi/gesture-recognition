# Author by CRS-club and wizard

import os
import numpy as np
import cv2

video_src_src_path = 'test/'
label_name = os.listdir(video_src_src_path)
print(label_name)
label_dir = {}
index = 0
for i in label_name:
    if i.startswith('.') or 'Jpg' in i:
        continue
    label_dir[i] = index
    index += 1
    video_src_path = os.path.join(video_src_src_path, i)
    video_save_path = os.path.join(video_src_src_path, i) + 'Jpg'
    if not os.path.exists(video_save_path):
        os.mkdir(video_save_path)

    videos = os.listdir(video_src_path)
    # 过滤出avi文件
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
            # print('read a new frame:', success)

            params = []
            params.append(1)
            if success:
                cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame, params)

            frame_count += 1
        cap.release()
np.save('test/label.npy', label_dir)
print(label_dir)

import os
import numpy as np
import pickle


label_dic = np.load('test/label.npy', allow_pickle=True).item()
print(label_dic)

source_dir = 'test/'
target_test_dir = 'test/'
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)

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
        output_pkl = os.path.join(target_test_dir, output_pkl)
        tag += 1
        f = open(output_pkl, 'wb')
        pickle.dump((vid, label_dic[key], frame), f, -1)
        f.close()

data_dir = 'test/'

train_data = os.listdir(data_dir)
train_data = [x for x in train_data if not x.startswith('.')]
print(len(train_data))

num = 0
f = open('test/test.txt', 'w')
for line in train_data:
    if '.pkl' in line:
        f.write(data_dir + line.split('_')[0] + '_' + str(num) + '.pkl' + '\n')
        num += 1
f.close()
