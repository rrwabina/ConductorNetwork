'''
Module loads the dataset into memory
'''
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from medpy.io import load
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from slicesdataset import SlicesDataset

def med_reshape(image, new_shape):
    reshaped_image = np.zeros(new_shape)
    reshaped_image[:image.shape[0], :image.shape[1], :image.shape[1]] = reshaped_image
    return reshaped_image

def load_data(y_shape = 64, z_shape = 64):
    image_dir = os.path.join('dataset/images', 'train')
    label_dir = os.path.join('dataset', 'labels')

    images = [f for f in listdir(image_dir) if (isfile(join(image_dir, f)) and f[0] != '.')]
    data = {}

    for f in images[0:20]:
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))
        image = image.astype('float')
        image /= np.max(image)

        image = med_reshape(image, new_shape = (image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape = (label.shape[0], y_shape, z_shape)).astype(int)
        data['image'] = image
        data['seg'] = label
        data['filename']: f
    return data 


def split_init():
    os.chdir('C:/Users/Renan/Desktop/dsai-thesis')
    directory = 'data_simon/SIMON_BIDS/sub-032633'
    FOLDERS = []
    FOLDERS_PATH = []
    COMPLETE_DATA = []
    COMPLETE_PATH = []

    for folder in os.listdir(directory):
        FOLDERS.append(folder)
        FOLDERS_PATH.append(os.path.join(directory, folder))

    for folder in FOLDERS_PATH:
        if len(os.listdir(folder)) > 1:
            path = os.path.join(folder, 'anat')
            ITEMS = []
            for item in os.listdir(path):
                ITEMS.append(item)
                for item in ITEMS:
                    if item.endswith('T2star.nii.gz'):
                        COMPLETE_DATA.append(folder)
    COMPLETE_DATA = list(set(COMPLETE_DATA))                 
    for complete in COMPLETE_DATA:
        COMPLETE_PATH.append(os.path.join(complete, 'anat'))

    T1W_FILES = []
    T2W_FILES = []
    T2STAR_FILES = []
    for folder in COMPLETE_PATH:
        for items in os.listdir(folder):
            if items.endswith('run-1_T1w.nii.gz') and not items.endswith('acq-10iso_run-1_T1w.nii.gz'):
                T1W_FILES.append(items)
            elif items.endswith('run-1_T2w.nii.gz'):
                T2W_FILES.append(items)
            elif items.endswith('run-1_T2star.nii.gz') and not items.endswith('acq-ph_run-1_T2star.nii.gz') and not items.endswith('acq-pmri_run-1_T2star.nii.gz'):
                T2STAR_FILES.append(items)

    T1W_PATH, T2W_PATH, T2STAR_PATH = [], [], []
    for T1W, T2W, T2STAR, complete in zip(T1W_FILES, T2W_FILES, T2STAR_FILES, COMPLETE_DATA):
        T1W_PATH.append(os.path.join(complete, 'anat', T1W))
        T2W_PATH.append(os.path.join(complete, 'anat', T2W))
        T2STAR_PATH.append(os.path.join(complete, 'anat', T2STAR))
    assert len(T1W_PATH) and len(T2W_PATH) and len(T2STAR_PATH)

    T1W_IMAGES, T2W_IMAGES, T2S_IMAGES = {}, {}, {}

    for complete, t1w, t2w, t2s in zip(COMPLETE_PATH, T1W_PATH, T2W_PATH, T2STAR_PATH):
        path1 = 'data_simon/SIMON_BIDS/sub-032633/ses'
        path2 = '/anat/sub-032633_ses'

        T1W_IMAGES['T1W' + complete[36:40]] = path1 + complete[36:40] + path2 + complete[36:40] + '_run-1_T1w.nii.gz'
        T2W_IMAGES['T2W' + complete[36:40]] = path1 + complete[36:40] + path2 + complete[36:40] + '_run-1_T2w.nii.gz'
        T2S_IMAGES['T2S' + complete[36:40]] = path1 + complete[36:40] + path2 + complete[36:40] + '_run-1_T2star.nii.gz'

    T1W_KEYS, T2W_KEYS, T2S_KEYS = [], [], []

    for T1W, T2W, T2S in zip(T1W_IMAGES.keys(), T2W_IMAGES.keys(), T2S_IMAGES.keys()):
        T1W_KEYS.append(T1W[4:7])
        T2W_KEYS.append(T2W[4:7])
        T2S_KEYS.append(T2S[4:7])
    T1W_KEYS = T1W_KEYS[1:21]
    T2W_KEYS = T2W_KEYS[1:21]
    T2S_KEYS = T2S_KEYS[1:21] 

    assert T1W_KEYS == T2W_KEYS == T2S_KEYS

    split = dict()
    split['train'] = T1W_KEYS[0:int(0.7 * len(T1W_KEYS))] 
    split['val'] = T1W_KEYS[int(0.7 * len(T1W_KEYS)) : int(0.9 * len(T1W_KEYS))]
    split['test'] = T1W_KEYS[int(0.9 * len(T1W_KEYS)) : ]

    assert(not bool(set(split['train']) & set(split['val'])))
    assert(not bool(set(split['val']) & set(split['test'])))
    return split, T1W_IMAGES, T2W_IMAGES, T2S_IMAGES

def load_image(filename: str, T1W_IMAGES, T2W_IMAGES, T2S_IMAGES):
    t1_weighted = sitk.ReadImage(T1W_IMAGES['T1W-' + filename], sitk.sitkFloat32)
    t2_weighted = sitk.ReadImage(T2W_IMAGES['T2W-' + filename], sitk.sitkFloat32)
    t2_star = sitk.ReadImage(T2S_IMAGES['T2S-' + filename], sitk.sitkFloat32)
    return t1_weighted, t2_weighted, t2_star

def load_split_data(data_type, preprocess = True):
    split, T1W_IMAGES, T2W_IMAGES, T2S_IMAGES = split_init()
    T1W_load, T2W_load, T2S_load = [], [], []
    for data in split[data_type]:
        T1W, T2W, T2S = load_image(data, T1W_IMAGES, T2W_IMAGES, T2S_IMAGES)
        if preprocess == True:
            T1W, T2W, T2S = torch.tensor(sitk.GetArrayFromImage(T1W)), torch.tensor(sitk.GetArrayFromImage(T2W)), torch.tensor(sitk.GetArrayFromImage(T2S))
        T1W_load.append(T1W), T2W_load.append(T2W), T2S_load.append(T2S) 
    type_loader = {}
    type_loader['T1W'] = T1W_load
    type_loader['T2W'] = T2W_load
    type_loader['T2S'] = T2S_load
    return type_loader