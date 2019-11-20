import os
import os.path as osp

import torch
import torch.utils.data as data
import numpy as np
import numpy.linalg as LA

__all__ = ['HMDB51DataLayer']

def find_classes(dir_):
    classes = sorted([
        d for d in os.listdir(dir_) if osp.isdir(osp.join(dir_, d))
    ])
    class_index = {classes[i]: i for i in range(len(classes))}
    return class_index

def build_dataset(lr_spatial_path, lr_temporal_path,
                  hr_spatial_path, hr_temporal_path,
                  class_index, splits):
    inputs = []
    max_length = -1
    for class_name in sorted(os.listdir(lr_spatial_path)):
        lr_class_spatial_path = osp.join(lr_spatial_path, class_name)
        lr_class_temporal_path = os.path.join(lr_temporal_path, class_name)
        hr_class_spatial_path = os.path.join(hr_spatial_path, class_name)
        hr_class_temporal_path = os.path.join(hr_temporal_path, class_name)

        for video_name in sorted(os.listdir(lr_class_spatial_path)):
            lr_video_spatial_path = os.path.join(lr_class_spatial_path, video_name)
            lr_video_temporal_path = os.path.join(lr_class_temporal_path, video_name)
            hr_video_spatial_path = os.path.join(hr_class_spatial_path, video_name)
            hr_video_temporal_path = os.path.join(hr_class_temporal_path, video_name)

            length = np.load(lr_video_spatial_path).shape[0]
            if 'lr' in splits:
                inputs.append((lr_video_spatial_path, lr_video_temporal_path, length, class_index[class_name]))
            if 'hr' in splits:
                inputs.append((hr_video_spatial_path, hr_video_temporal_path, length, class_index[class_name]))
            max_length = max(max_length, length)

    return inputs, max_length

def default_loader(path):
    c3d_feat = np.load(path).astype(np.float32)
    norm = LA.norm(c3d_feat, axis=1)
    c3d_feat /= norm[:, None]
    return torch.as_tensor(c3d_feat)

class HMDB51DataLayer(data.Dataset):
    def __init__(self, data_root, phase=None, loader=default_loader):
        if 'Test' in phase:
            splits = 'lr'
        else:
            splits = 'lrhr'
        class_index = find_classes(data_root)
        inputs, max_length = build_dataset(data_root+'_lr_features', data_root+'_lr_optfl_features',
                                           data_root+'_hr_features', data_root+'_hr_optfl_features',
                                           class_index, splits)

        self.data_root = data_root
        self.inputs = inputs
        self.max_length = max_length
        self.class_index = class_index
        self.loader = loader

    def pad_zeros(self, tensor, max_length):
        return torch.cat([tensor, tensor.new_zeros(max_length - tensor.shape[0], *tensor.shape[1:])])

    def __getitem__(self, index):
        spatial_path, temporal_path, length, target = self.inputs[index]
        spatial_feature = self.pad_zeros(self.loader(spatial_path), self.max_length)
        temporal_feature = self.pad_zeros(self.loader(temporal_path), self.max_length)
        return spatial_feature, temporal_feature, length, target

    def __len__(self):
        return len(self.inputs)
