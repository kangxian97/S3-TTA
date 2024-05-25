import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
from pathlib import Path
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import torch
import cv2
import glob
import logging
models_logger = logging.getLogger(__name__)



def rotational_consistency_pair_wise(input_list):
        lens = len(input_list)
        flatten = [(i[:,:,2]>0).flatten() for i in input_list]
        dices = []
        for i in range(lens):
            j=i+1
            while j<lens:
                intersection = flatten[i] * flatten[j]
                dices.append((2. * intersection.sum() + 1) / (flatten[i].sum() + flatten[j].sum() + 1))
                j+=1
        return sum(dices)/len(dices)
        
def scale_augmentation(img,scales):
    """ 
    perform scale augmentation
    
    Parameters
    ------------------
        img: original image as a np array
        scales: list of scaling options
    
    Return
    ------------------
    A list of images at all scales 
    """
    imgs_o = img.copy()
    h_ori,w_ori = img.shape[0], img.shape[1]
    #get heights and widths according to scale options
    sizes_heights = [int(h_ori*scale) for scale in scales]
    sizes_widths = [int(w_ori*scale) for scale in scales]
    #resize the original image
    resized_images = [cv2.resize(img.copy(), dsize=(w_curr, h_curr), interpolation=cv2.INTER_CUBIC) for (h_curr, w_curr) in zip(sizes_heights,sizes_widths)]
    #add the original image too
    resized_images.append(imgs_o)
    return resized_images
                
def rotation_augmentation(imgs):
    """ 
    perform rotation augmentation
    
    Parameters
    ------------------
        imgs: 2D list of np array images at different scales (no rotation)
              [
               [img_Scale1],
               [img_Scale2],
                ...,
               [img_ScaleN]
              ]
        
    Return
    ------------------
        2D list of np array images 
        1st dimension represents scale
        2nd dimension represents rotated images at various angles
        [
         [img_Scale1_Rotation1, ..., img_Scale1_RotationN], 
          ..., 
         [img_ScaleN_Rotation1, ..., img_ScaleN_RotationN]
        ]
    """
    for idx, s in enumerate(imgs):
        s.append(cv2.rotate(s[0], cv2.ROTATE_90_CLOCKWISE))
        s.append(cv2.rotate(s[0], cv2.ROTATE_180))
        s.append(cv2.rotate(s[0], cv2.ROTATE_90_COUNTERCLOCKWISE))
    return imgs

def find_best_aug_index(dice_scores):
    """ 
    find the combination of scale and style with highest dice score
    
    Parameters
    ------------------
        dice_scores: 2D list of dice scores
        
    Return
    ------------------
        indices of optimal scale, style 
    """
    best_scale_ind, best_style_ind = 0, 0
    max_dice = 0
    for scale_ind, scale in enumerate(dice_scores):
        for style_ind, scale_style in enumerate(scale):
            if scale_style > max_dice:
                best_scale_ind = scale_ind
                best_style_ind = style_ind
                max_dice = scale_style
    return best_scale_ind, best_style_ind
    
