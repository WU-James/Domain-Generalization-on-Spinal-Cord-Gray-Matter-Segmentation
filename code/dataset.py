import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import os
import re
import SimpleITK as stik
from collections import OrderedDict
from torchvision import transforms
import random
import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, random_split

import numbers
import torch.nn.functional as F
from functools import wraps



def makeDataset(phase='train', path='/home/hlli/project/yufei/Medical_Segmentation_Dataset/GGM_spinal_cord_challenge',
                specific_domain=None, transform_train=None, transform_eval=None):
    """
    :param transform_train:
    :param phase: train or infer
        train: return slice, gt
        infer: return a
    :param path:
    :param specific_domain:
        None: return all domains
        list of str "site%d" : return specified one or several datasets
    :return:
    """
    # assert phase in ['train', 'infer', 'train_nips']

    path1, path2 = os.path.join(path, 'train'), os.path.join(path, 'test')
    if phase == 'train' or phase == 'train_nips':
        imageFileList = [os.path.join(path1, f) for f in os.listdir(path1) if 'site' in f and '.txt' not in f]
    elif phase == 'infer':
        imageFileList = [os.path.join(path2, f) for f in os.listdir(path2) if 'site' in f and '.txt' not in f]
    data_dict = {'site1': OrderedDict(), 'site2': OrderedDict(), 'site3': OrderedDict(), 'site4': OrderedDict()}
    for file in imageFileList:
        res = re.search('site(\d)-sc(\d*)-(image|mask)', file).groups()
        if res[1] not in data_dict['site' + res[0]].keys():
            data_dict['site' + res[0]][res[1]] = {'input': None, 'gt': []}
        if res[2] == 'image':
            data_dict['site' + res[0]][res[1]]['input'] = file
        if res[2] == 'mask':
            data_dict['site' + res[0]][res[1]]['gt'].append(file)
    datasets = {}
    print('Making dataset...')
    resolution = {
        'site1': [5, 0.5, 0.5],
        'site2': [5, 0.5, 0.5],
        'site3': [2.5, 0.5, 0.5],
        'site4': [5, 0.29, 0.29],
    }
    for domain, data_list in data_dict.items():
        if specific_domain is None or domain in specific_domain:
            datasets[domain] = SpinalCordDataset(data_list, domain=domain, phase=phase, transform_train=transform_train,
                                                 transform_eval=transform_eval, resolution=resolution[domain], )
    print('Dataset finished')
    return datasets


class SpinalCordDataset(dataset.Dataset):
    def __init__(self, data_list, domain, phase, transform_train=None, transform_eval=None, **kwargs):
        self.phase = phase
        self.reader = stik.ImageFileReader()
        self.reader.SetImageIO("NiftiImageIO")
        self.data_list = self.__read_dataset_into_memory(data_list)
        self.map_list = self.__get_index_map()
        self.info_dict = kwargs
        self.domain = domain

        self.input_transform = None  # transforms.Compose(transforms.ToPILImage,transforms.ToTensor)
        self.gt_transform = None
        # self.transform_train = T.ComposedTransform()
        if transform_train is None:
            transform_train = ComposedTransform([Resize(160), CenterCrop(160)])  # 144 y  128 n # ,
        if transform_eval is None:
            transform_eval = ComposedTransform([CenterCrop(144)])  # 没有用目前
        self.transform_train = transform_train

        self.flat_img, self.flat_spinal_cord_mask, self.flat_gm_mask = self.__get_flat_data(self.map_list)

        self.real_sample_num = len(self.data_list) if phase == 'infer' else len(self.map_list)

    def __get_flat_data(self, map_list):
        flat_img, flat_spinal_cord_mask, flat_gm_mask = [], [], []

        for i in range(len(map_list)):
            x, gt_list = self.map_list[i]
            x = x / (x.max() if x.max() > 0 else 1)
            gt_list = torch.tensor(gt_list, dtype=torch.uint8)
            spinal_cord_mask = (torch.mean((gt_list > 0).float(), dim=0) > 0.5).float()
            gm_mask = (torch.mean((gt_list == 1).float(), dim=0) > 0.5).float()
            x, spinal_cord_mask, gm_mask = self.transform_train(x, spinal_cord_mask, gm_mask)

            if torch.sum(spinal_cord_mask) == 0 or torch.sum(gm_mask) == 0:
                continue
            else:
                flat_img.append(x)
                flat_spinal_cord_mask.append(spinal_cord_mask)
                flat_gm_mask.append(gm_mask)

        return flat_img, flat_spinal_cord_mask, flat_gm_mask

    def __get_index_map(self):
        map_list = []
        total_slice_num = 0
        for data in self.data_list.values():
            slice_num = data['input'].shape[0]
            for i in range(slice_num):
                map_list.append([data['input'][i], np.stack([data['gt'][idx][i] for idx in range(4)], axis=0)])
            total_slice_num += slice_num
        return map_list

    def __read_dataset_into_memory(self, data_list):
        for val in data_list.values():
            val['input'] = self.read_numpy(val['input'])
            for idx, gt in enumerate(val['gt']):
                val['gt'][idx] = self.read_numpy(gt)
        return data_list

    def __getitem__(self, idx):
        if self.phase == 'train' or self.phase == 'train_nips':
            if self.phase == 'train_nips':
                idx = random.randint(0, self.real_sample_num - 1)

            # x, gt_list = self.map_list[idx]
            # x = x / (x.max() if x.max() > 0 else 1)
            # gt_list = torch.tensor(gt_list, dtype=torch.uint8)
            # spinal_cord_mask = (torch.mean((gt_list > 0).float(), dim=0) > 0.5).float()
            # gm_mask = (torch.mean((gt_list == 1).float(), dim=0) > 0.5).float()
            # # a1 = [torch.sum(spinal_cord_mask), torch.sum(gm_mask)]
            # x, spinal_cord_mask, gm_mask = self.transform_train(x, spinal_cord_mask, gm_mask)

            # a2 = [torch.sum(spinal_cord_mask), torch.sum(gm_mask)]
            # if a1 != a2:
            #     print(a1, a2)
            return self.flat_img[idx], self.flat_spinal_cord_mask[idx], self.flat_gm_mask[idx], self.domain
        elif self.phase == 'infer':
            list_temp = list(self.data_list.values())[idx]
            x, gt_list = list_temp['input'], list_temp['gt']
            return x, gt_list, self.domain

    def __len__(self):
        # if self.phase == 'train' or self.phase == 'train_nips':
        #     return len(self.map_list)
        # elif self.phase == 'infer':
        #     return len(self.data_list)

        # elif self.phase == 'train_nips':
        #     return self.train_nips_sample_num

        return len(self.flat_img)

    def read_numpy(self, file_name):
        self.reader.SetFileName(file_name)
        data = self.reader.Execute()
        return stik.GetArrayFromImage(data)

    def set_phase(self, phase):
        assert phase in ['train', 'test', 'infer']


def get_dataset(data_path, test_domain, train_split_ratio):
    train_domains = ["site1", "site2", "site3", "site4"]
    del train_domains[int(test_domain[4]) - 1]
    train_dataset = makeDataset(phase='train', path=data_path, specific_domain=train_domains,
                                transform_train=None, transform_eval=None)

    trains = []
    vals = []
    for dataset in train_dataset.values():
        train_size = int(train_split_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train, val = random_split(dataset, [train_size, test_size])
        trains.append(train)
        vals.append(val)

    train_dataset = ConcatDataset(trains)
    val_dataset = ConcatDataset(vals)

    test_dataset = \
        makeDataset(phase='train', path=data_path, specific_domain=test_domain, transform_train=None,
                    transform_eval=None)[test_domain]

    return train_dataset, val_dataset, test_dataset


# Transformation

class Rand:
    def __init__(self, seed=1234):
        self.seed = seed
        self.m = 2 ^ 31
        self.a = 1103515245
        self.c = 12345

        self.x_n = seed
        self.x_n_backup = seed  # record the x_n value before reset

    def step(self):
        self.x_n = self.x_n_backup

    def rand(self, num=1):
        x_n = self.x_n
        result = []
        for i in range(num):
            self.x_n = (self.a * self.x_n + self.c) % self.m
            result.append(self.x_n / self.m)
        self.x_n_backup = self.x_n
        self.x_n = x_n
        return result if num > 1 else result[0]

def pre_post_deal(func):
    """
    The wrap converts the tensor into numpy and reshape the input from 1*1*w*h to w*h
    after transform, convert the output to the tenosr with 1*1*w*h
    :param func:
    :return: tensor
    """

    @wraps(func)
    def checked(self, img, gt, gt2):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if isinstance(gt2, torch.Tensor):
            gt2 = gt2.detach().cpu().numpy()
        img = img.squeeze()
        gt = gt.squeeze()
        gt2 = gt2.squeeze()
        if random.random() > 0.5 or func.__qualname__ == 'RandomCrop.__call__' or func.__qualname__ == 'CenterCrop.__call__' or func.__qualname__ == 'Resize.__call__':
            img_, gt_, gt2_ = func(self, img, gt, gt2)
        else:
            img_, gt_, gt2_ = img, gt, gt2
        shape = gt_.shape
        return torch.tensor(img_).reshape([1, *shape]).float(), torch.tensor(gt_).reshape(
            [1, *shape]).float(), torch.tensor(gt2_).reshape(
            [1, *shape]).float()

    return checked


syn_rand = Rand()


def pad(img, val):
    """
    :param val: int or list with length of 4 (w1,w2,h1,h2)
    :return: padded img
    """
    if isinstance(val, numbers.Number):
        val = [val for _ in range(4)]
    assert len(img.shape) in [2]
    w, h = img.shape
    new_img = np.zeros([w + val[0] + val[1], h + val[2] + val[3]])
    new_img[val[0]:w + val[0], val[2]:h + val[2]] = img
    return new_img


class RandomCrop:
    def __init__(self, size=224):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size

    def get_params(self, img):
        w, h = img.shape
        th, tw = self.size  # int(h * scale), int(w * scale)
        if w <= tw:  # and h <= th:
            tw = w
        if h <= th:
            th = h
        # return 0, 0, h, w

        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)
        # print(w, h, i, j, th, tw)
        return i, j, th, tw

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        h, w = self.size
        img_h, img_w = img.shape
        pad_val = [0, 0, 0, 0]
        if h > img_h:
            pad_val[0] = (h - img_h) // 2
            pad_val[1] = (h - img_h) - pad_val[0]
        if w > img_w:
            pad_val[2] = (w - img_w) // 2
            pad_val[3] = (w - img_w) - pad_val[2]
        if sum(pad_val) != 0:
            img = pad(img, pad_val)
            gt = pad(gt, pad_val)
            gt2 = pad(gt2, pad_val)
        i, j, h, w = self.get_params(img)
        return img[i:i + h, j:j + w], gt[i:i + h, j:j + w], gt2[i:i + h, j:j + w]


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            self.size = size

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        h, w = self.size
        img_h, img_w = img.shape
        pad_val = [0, 0, 0, 0]
        if h > img_h:
            pad_val[0] = (h - img_h) // 2
            pad_val[1] = (h - img_h) - pad_val[0]
        if w > img_w:
            pad_val[2] = (w - img_w) // 2
            pad_val[3] = (w - img_w) - pad_val[2]
        if sum(pad_val) != 0:
            img = pad(img, pad_val)
            gt = pad(gt, pad_val)
            gt2 = pad(gt2, pad_val)
        img_h, img_w = img.shape
        i_0, i_1 = img_h // 2 - h // 2, img_h // 2 + h // 2
        j_0, j_1 = img_w // 2 - w // 2, img_w // 2 + w // 2
        return img[i_0:i_1, j_0:j_1], gt[i_0:i_1, j_0:j_1], gt2[i_0:i_1, j_0:j_1]


class Sharpness:
    def __init__(self, strength_range=None):
        if strength_range is None:
            strength_range = [10, 30]
        self.strength_range = strength_range

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        center_val = random.uniform(*self.strength_range)
        kernel = np.ones([3, 3], dtype=np.float32) * (-(center_val - 1) / 8)
        kernel[1, 1] = center_val
        out = cv2.filter2D(img, kernel=kernel, ddepth=-1)
        return out, gt, gt2


class Blurriness:
    def __init__(self, ksize=3, sigma_range=None):
        if sigma_range is None:
            sigma_range = [0.25, 1.5]
        self.ksize = ksize
        self.sigma_range = sigma_range

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        out = cv2.GaussianBlur(img, ksize=(self.ksize, self.ksize), sigmaX=random.uniform(*self.sigma_range))
        return out, gt, gt2


class Noise:
    def __init__(self, std_range=None):
        if std_range is None:
            std_range = [0.1, 1.0]
        self.std_range = std_range

    def add_gaussian_noise(self, img, std, mean=0):
        noise = np.random.normal(mean, std, img.shape)
        out = noise + img
        out = np.clip(out, a_min=0, a_max=1)
        return out

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        out = self.add_gaussian_noise(img, random.uniform(*self.std_range))
        return out, gt, gt2


class Brightness:
    def __init__(self, scale_shift_range=None, bias_range=None):
        if bias_range is None:
            bias_range = [-0.1, 0.1]
        self.bias_range = bias_range
        if scale_shift_range is None:
            scale_shift_range = [-0.1, 0.1]
        self.scale_shift_range = scale_shift_range

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        out = img * (1 + random.uniform(*self.scale_shift_range)) + random.uniform(*self.bias_range)
        out = np.clip(out, a_min=0, a_max=1)
        return out, gt, gt2


class Rotation:
    def __init__(self, angle_range=None):
        if angle_range is None:
            angle_range = [-20, 20]
        self.angle_range = angle_range

    @staticmethod
    def rotate(image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        random_angle = random.uniform(*self.angle_range)
        out = Rotation.rotate(img, angle=random_angle)
        gt_ = Rotation.rotate(gt, angle=random_angle)
        gt2_ = Rotation.rotate(gt2, angle=random_angle)
        return out, gt_, gt2_


class Scale:
    def __init__(self, magnitude_range=None):
        if magnitude_range is None:
            magnitude_range = [0.4, 1.6]
        self.magnitude_range = magnitude_range

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        scale_x = random.uniform(*self.magnitude_range)
        scale_y = random.uniform(*self.magnitude_range)
        img_ = cv2.resize(img, dsize=(0, 0), fx=scale_x, fy=scale_y)
        gt_ = cv2.resize(gt, dsize=(0, 0), fx=scale_x, fy=scale_y)
        gt2_ = cv2.resize(gt2, dsize=(0, 0), fx=scale_x, fy=scale_y)
        return img_, gt_, gt2_


class Resize:
    def __init__(self, size):
        self.size = size

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        if img.shape[0] < self.size:
            img = cv2.resize(img, dsize=(self.size, self.size))
            gt = cv2.resize(gt, dsize=(self.size, self.size))
            gt2 = cv2.resize(gt2, dsize=(self.size, self.size))
        return img, gt, gt2


class ComposedTransform:
    def __init__(self, transform_list=None):
        if transform_list is None:
            transform_list = [CenterCrop(160), Sharpness([0, 30]), Blurriness(), Noise([0., 0.05]), Brightness(),
                              Rotation(), Scale([0.7, 1.3]), RandomCrop(144)]
        self.transform_list = transform_list

    def __call__(self, img, gt, gt2):
        for transform in self.transform_list:
            img, gt, gt2 = transform(img, gt, gt2)
        return img, gt, gt2



if __name__ == '__main__':
    datasets = makeDataset()
    for d in datasets.values():
        sample_num = len(d)
        for i in range(sample_num):
            d[i]
    # dataloader.DataLoader()
