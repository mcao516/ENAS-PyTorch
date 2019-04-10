# -*- coding: utf-8 -*-

"""This python file contains mothods for cifar10 image sdata preparation.
   
   Author: Meng Cao
"""

import os
import random
import numpy as np
import pickle
import torch

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def _read_data(file):
    """Read cifar10 data and return image arrays and labels.
    
    Args:
        file: str, file path.
    Return:
        images: [N, C, H, W]
        labels: [N]
    """
    data = unpickle(file)
    images = data[b'data'].astype(np.float32) / 255.0
    images = np.reshape(images, [-1, 3, 32, 32])
    labels = np.array(data[b'labels'], dtype=np.int32)
    
    return images, labels


def get_mean_and_std(images):
    """
    Args:
        images: numpy array with shape [N, C, H, W]
        
    Return:
        mean: numpy array with shape [1, C, 1, 1]
        std: numpy array with shape [1, C, 1, 1]
    """
    mean = np.mean(images, axis=(0, 2, 3), keepdims=True)
    std = np.std(images, axis=(0, 2, 3), keepdims=True)
    
    return mean, std


def read_cifar10(data_dir, valid_num=5000):
    """Read cifar10 data and return train, valid and test dataset.
    
    Args:
        data_dir: str, cifar10 data directory.
    """
    train_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    test_file = "test_batch"
    
    image_batches, label_batches = [], []
    for f in train_files:
        image_batch, label_batch = _read_data(os.path.join(data_dir, f))
        image_batches.append(image_batch)
        label_batches.append(label_batch)
        
    image_batches = np.concatenate(image_batches, axis=0)
    label_batches = np.concatenate(label_batches, axis=0)
    
    # build train, valid and test set
    images, labels = {}, {}
    if valid_num:
        images["valid"] = image_batches[-valid_num:]
        labels["valid"] = label_batches[-valid_num:]

        images["train"] = image_batches[:-valid_num]
        labels["train"] = label_batches[:-valid_num]
    else:
        images["valid"] = None
        labels["valid"] = None
        
        images["train"] = image_batches
        labels["train"] = label_batches
    
    # mean, std = get_mean_and_std(images["train"])
    # images["train"] = (images["train"] - mean) / std
    # images["valid"] = (images["valid"] - mean) / std
    
    images["test"], labels["test"] = _read_data(os.path.join(data_dir, test_file))
    
    return images, labels


class WrappedDataLoader:
    """For preprocessing purpose.
    """
    def __init__(self, data_loader, func):
        self.data_loader = data_loader
        self.func = func
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        batches = iter(self.data_loader)
        for b in batches:
            yield (self.func(b['image'], b['label']))

    # def get_batch_size(self):
    #     return len(self)

    # def get_sample_size(self):
    #     return len(self.data_loader.dataset)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Args:
            sample: {image: numpy image C * H * W, 
                     label: numpy.int32, label}
        """
        image, label = sample['image'], sample['label']

        return {'image': torch.from_numpy(image),
                'label': torch.tensor(label, dtype=torch.int64)}
    
class DatasetBuilder(object):
    """Class for dataset building. 
    
       Given the path of the train, dev and test directory, return iterators 
       for mini-batch training. Each picture is processed using defined transform
       objects.
    """
    
    def __init__(self, input_size, mean=None, std=None):
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.data_transforms = self._build_transforms(self.input_size, self.mean, self.std)
        
    def build_dataset(self, images, labels, data_type, batch_size, shuffle=False, num_workers=4, device=None):
        assert data_type in ['train', 'dev', 'test']
        
        image_dataset = MyDataset(images, labels, self.data_transforms[data_type])
        dl = DataLoader(image_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        
        if device is not None:
            return WrappedDataLoader(dl, lambda x, y: (x.to(device), y.to(device)))
        return dl

    def _build_transforms(self, input_size, mean, std):
        assert mean is not None and std is not None
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(input_size),
                # transforms.RandomHorizontalFlip(),
                ToTensor(),
                # transforms.Normalize(mean, std)
            ]),
            'dev': transforms.Compose([
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                ToTensor(),
                # transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                ToTensor(),
                # transforms.Normalize(mean, std)
            ]),
        }
        return data_transforms


class MyDataset(Dataset):
    """Customerize dataset."""

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: numpy array [N, C, H, W]
            labels: numpy array [N]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.images[idx], 'label': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample