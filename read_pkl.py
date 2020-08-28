# Author by CRS-club and wizard

import pickle
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def decode_pickle(sample, mode, seg_num, seglen, short_size,
                  target_size, img_mean, img_std):
    pickle_path = sample
    try:
        data_load = pickle.load(
            open(pickle_path, 'rb'), encoding='bytes'
        )
        vid, label, frames = data_load
    except:
        print('Error when loading', pickle_path)
        return None, None

    if mode == 'train' or mode == 'valid' or mode == 'test':
        ret_label = label
    elif mode == 'infer':
        ret_label = vid

    imgs = video_loader(frames, seg_num, seglen, mode)
    return imgs_transform(imgs, ret_label, mode, seg_num, seglen,
                          short_size, target_size, img_mean, img_std)


def video_loader(frames, nsample, seglen, mode):
    video_len = len(frames)
    average_dur = int(video_len / nsample)

    imgs = []
    for i in range(nsample):
        idx = 0
        if mode == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - seglen) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
            imgbuf = frames[int(jj % video_len)]
            img = imageloader(imgbuf)
            imgs.append(img)
    return imgs


def imageloader(imgbuf):
    img = Image.open(imgbuf)
    return img.convert('RGB')


def imgs_transform(imgs, label, mode, seg_num, seglen, short_size,
                   target_size, img_mean, img_std):
    imgs = group_scale(imgs, short_size)

    if mode == 'train':
        imgs = group_random_crop(imgs, target_size)
        imgs = group_random_flip(imgs)
    else:
        imgs = group_center_crop(imgs, target_size)

    np_imgs = (np.array(imgs[0]).astype('float32').transpose(
        (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
    for i in range(len(imgs) - 1):
        img = (np.array(imgs[i + 1]).astype('float32').transpose(
            (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
        np_imgs = np.concatenate((np_imgs, img))
    imgs = np_imgs
    # imgs -= img_mean
    # imgs /= img_std
    imgs = np.reshape(imgs,
                      (seg_num, seglen * 3, target_size, target_size))

    return imgs, label


def group_scale(imgs, target_size):
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            oh = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        else:
            oh = target_size
            ow = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

    return resized_imgs


def group_random_crop(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
          "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images


def group_random_flip(img_group):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    else:
        return img_group


def group_center_crop(img_group, target_size):
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size
        assert (w >= target_size) and (h >= target_size), \
             "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop


def reader(file_path, batch_size, seg_num, seglen, short_size, target_size):
    pkl_list = os.listdir(file_path)
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1]).astype(np.float32)
    std = np.array([1, 1, 1]).reshape([3, 1, 1]).astype(np.float32)
    random.shuffle(pkl_list)
    batch = 0
    for ii in pkl_list:
        temp_path = os.path.join(file_path, ii)
        imgs, label = decode_pickle(temp_path, 'train', seg_num, seglen, short_size, target_size, mean, std)
        batch += 1
        if batch == batch_size:
            yield imgs, label
            batch = 0
        else:
            pass


if __name__ == '__main__':
    data_dir = 'hmdb_data_demo/train/'
    for list in reader(data_dir, 1, 3, 1, 240, 224):
        print(list[0])
        label = list[1]
        imgs = list[0]
        print(label)
        k = 1
        for i in imgs:
            temp = np.transpose(i, [1, 2, 0])
            plt.figure(str(k))
            k += 1
            plt.imshow(temp)
            plt.axis('off')
        plt.show()
        os.system("pause")

