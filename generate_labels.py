import nibabel as nib 
import os 
import SimpleITK as sitk
import load_nii_hdr as load
import numpy as np
from numpy import pi, exp, sqrt
import warnings
import cv2
import scipy.ndimage
import math


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
    return T1W_IMAGES, T2W_IMAGES, T2S_IMAGES

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def matlab_style_gauss2D(shape=(3, 3), sigma = 0.5):
    '''
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    '''
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2. * sigma * sigma) )
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def generate_labels(root, plot = False):
    os.chdir(root)
    T1W_IMAGES, _, _ = split_init()
    cases, files = T1W_IMAGES.keys(), T1W_IMAGES.values()
    for case, file in zip(cases, files):
        h1dr, h2dr = nib.load(file), nib.load(file)
        h1dr_header, h2dr_header = h1dr.header['scl_slope'], h2dr.header['scl_slope']
        v1, v2 = h1dr.get_data(), h2dr.get_data()
        im1, im2 = v1, v2
        slices = im1.shape[0]

        c1, c2, c3 = 0.286, 1.526e-05, 11.852
        w1, w2 = 1.525, 1.443
        CondRange, WaterRange, InstensRange = range(0, 2), range(60, 100), range(100, 1400)

        porcenThld = 0.03
        tmp1x = np.abs(im2)
        RegionInterest = np.ones(tmp1x.shape)
        tmp2x = tmp1x

        for slice in np.arange(0, slices).reshape(-1):
            tmp1 = cv2.medianBlur(tmp1x[slice, :, :], 1)
            s, k = 1, 2 
            H = matlab_style_gauss2D()
            tmp1 = scipy.ndimage.convolve(tmp1, H, mode = 'nearest')
            tmp1x[slice, :, :] = tmp1x[slice, :, :] / np.amax(np.amax(tmp1x))
            tmp2x[slice, :, :] = tmp1 / np.amax(tmp1)
        
        Thld = np.amax(np.abs(tmp2x)) * porcenThld
        RegionInterest[np.abs(tmp1x) <= Thld] = 0
        obj_v = [RegionInterest == 1]

        ImRatio = np.abs(im1 / im2)
        ImRatio[np.isnan(ImRatio)] = 0
        ImRatio[np.isinf(ImRatio)] = 0
        Ir = ImRatio
        IrC = np.abs(np.multiply((Ir), RegionInterest))

        IW = np.multiply(w1, np.exp(np.multiply(-w2, IrC)))
        IW = np.abs(np.multiply(IW, RegionInterest))
        v1 = np.abs(np.multiply(IW, RegionInterest) * 10000)
        IW[IW > 1] = 1
        v1 = np.abs(np.multiply(IW, RegionInterest) * 10000)

        ImCond = (c1 + np.multiply(c2, np.exp(np.multiply(c3, IW))))
        ROIept = IW
        ROIept[ROIept != 0] = 1
        v1 = np.abs(np.multiply(ImCond, ROIept) * 10000)
        ImCond = np.multiply(ImCond, ROIept)

        if plot == True:
            plt.figure(figsize = (8, 8))
            plt.imshow(tmp1x[sliceToShow, :, :], extent = [0, 1, 0, 1], cmap = 'inferno')
            plt.colorbar()
            plt.clim(0, 1.2)
            plt.title(f'{case} (S/m) at slice {sliceToShow}', size = 15)
            plt.show()
        with_results = [52, 55, 25, 38, 71, 
                        26, 18, 32, 56, 37, 
                        30, 35, 20, 23, 73, 
                        19, 21, 13, 41, 12,
                        17, 22]
        if case[5:7] in str(with_results):
            img = nib.Nifti1Image(tmp1x, h1dr.affine)
            nib.save(img, os.path.join('labels', case + '.nii.gz'))
        
        
# root = 'C:/Users/Renan/Desktop/dsai-thesis'
# generate_labels(root)