import tensorflow as tf
from segmentation_models import Unet, get_preprocessing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, model_from_json
#from segmentation_models.metrics import iou_score, dice_score
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import regularizers
from PIL import Image
from random import randint
import numpy as np
import os
import pandas as pd
import glob
import math
import warnings
import tensorflow.keras.backend as K
import pdb
from tensorflow.keras.models import load_model
import tensorflow.keras.losses

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

BACKBONE = 'resnet50'
preprocess_input = get_preprocessing(BACKBONE)


def read_imgs_keraspp_stacked(imgs_df):
    imgs_tensor = np.zeros((len(imgs_df), 224, 224, 9))
    imgs_labels = np.zeros((len(imgs_df), 224, 224, 1))
    for index, (_, row) in enumerate(imgs_df.iterrows()):
        im_name = row['image']
        im_name_prior = row['image'].replace('SENTINEL', '0SENTINEL')
        im_name_anterior = row['image'].replace('SENTINEL', '2SENTINEL')
        img_central = np.array(Image.open(im_name).resize([224, 224]).convert('RGB'))
        img_prior = img_central
        img_anterior = img_central
        if os.path.exists(im_name_prior):
          img_prior = np.array(Image.open(im_name_prior).resize([224, 224]).convert('RGB')) 
        if os.path.exists(im_name_anterior):
          img_anterior = np.array(Image.open(im_name_anterior).resize([224, 224]).convert('RGB'))

        imgs_tensor[index, :, :, 0:3] = img_prior
        imgs_tensor[index, :, :, 3:6] = img_central
        imgs_tensor[index, :, :, 6:9] = img_anterior
        imgs_labels[index, :, :, 0] = np.array(Image.open(row['mask']).resize([224, 224]))/255.0

    return imgs_tensor, imgs_labels

def read_imgs_keraspp_multi_input(imgs_df):
    imgs_tensor = np.zeros((len(imgs_df), 224, 224, 9))
    imgs_labels = np.zeros((len(imgs_df), 224, 224, 1))
    for index, (_, row) in enumerate(imgs_df.iterrows()):
        im_name_1 = row['image1']
        im_name_2 = row['image2']
        im_name_3 = row['image3']
        img_1 = np.array(Image.open(im_name_1).resize([224, 224]).convert('RGB'))
        img_2 = img_1
        img_3 = img_1
        if os.path.exists(im_name_2):
          img_2 = np.array(Image.open(im_name_2).resize([224, 224]).convert('RGB')) 
        if os.path.exists(im_name_3):
          img_3 = np.array(Image.open(im_name_3).resize([224, 224]).convert('RGB'))

        imgs_tensor[index, :, :, 0:3] = img_1
        imgs_tensor[index, :, :, 3:6] = img_2
        imgs_tensor[index, :, :, 6:9] = img_3
        imgs_labels[index, :, :, 0] = np.array(Image.open(row['mask']).resize([224, 224]))/255.0

    return imgs_tensor, imgs_labels

def read_imgs_keraspp_output(imgs_df):
    imgs_tensor = np.zeros((len(imgs_df), 224, 224, 3))
    imgs_labels = np.zeros((len(imgs_df), 224, 224, 1))
    for index, (_, row) in enumerate(imgs_df.iterrows()):
        imgs_labels[index, :, :, 0] = np.array(Image.open(row['mask']).resize([224, 224]))/255.0

    return imgs_labels
    

def read_imgs_keraspp(imgs_df):
    imgs_tensor = np.zeros((len(imgs_df), 224, 224, 3))
    imgs_labels = np.zeros((len(imgs_df), 224, 224, 1))
    for index, (_, row) in enumerate(imgs_df.iterrows()):
        imgs_tensor[index, :, :, :] = np.array(Image.open(row['image']).resize([224, 224]).convert('RGB'))
        imgs_labels[index, :, :, 0] = np.array(Image.open(row['mask']).resize([224, 224]))/255.0

    return imgs_tensor, imgs_labels

def read_imgs_keraspp_labels(imgs_df, input_shape=224):
    imgs_labels = np.zeros((len(imgs_df), input_shape, input_shape, 1))
    for index, (_, row) in enumerate(imgs_df.iterrows()):
        imgs_labels[index, :, :, 0] = np.array(Image.open(row['mask']).resize([input_shape, input_shape]))/255.0

    return imgs_labels


def read_imgs_keraspp_DG(imgs_df):
    imgs_tensor = np.zeros((len(imgs_df), 512, 512, 3))
    imgs_labels = np.zeros((len(imgs_df), 512, 512, 1))
    for index, (_, row) in enumerate(imgs_df.iterrows()):
        imgs_tensor[index, :, :, :] = np.array(Image.open(row['image']).resize([512, 512]).convert('RGB'))
        imgs_labels[index, :, :, 0] = np.array(Image.open(row['mask']).resize([512, 512]))/255.0

    return imgs_tensor, imgs_labels

def batch_generator_multi_input(df, batch_size=64, stacked=True, is_imagenet=True):
    while True:
        for batch_index in range(round(len(df)/batch_size)):
            if stacked:
                x_train, y_train = read_imgs_keraspp_multi_input(df[batch_index*batch_size:batch_index*batch_size+batch_size])
            else:
                x_train, y_train = read_imgs_keraspp(df[batch_index*batch_size:batch_index*batch_size+batch_size])

            if is_imagenet:
                x_train = preprocess_input(x_train)
            yield x_train, y_train


def batch_generator(df, batch_size=64, stacked=True, is_imagenet=True):
    while True:
        for batch_index in range(round(len(df)/batch_size)):
            if stacked:
                x_train, y_train = read_imgs_keraspp_stacked(df[batch_index*batch_size:batch_index*batch_size+batch_size])
            else:
                x_train, y_train = read_imgs_keraspp(df[batch_index*batch_size:batch_index*batch_size+batch_size])

            if is_imagenet:
                x_train = preprocess_input(x_train)
            yield x_train, y_train


def batch_generator_DG(df, batch_size=64, is_imagenet=True):
    while True:
        for batch_index in range(round(len(df)/batch_size)):
            x_train, y_train = read_imgs_keraspp_DG(df[batch_index*batch_size:batch_index*batch_size+batch_size])

            if is_imagenet:
                x_train = preprocess_input(x_train)
            yield x_train, y_train
