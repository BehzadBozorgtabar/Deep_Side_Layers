# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:23:23 2017

@author: Behzad
"""

#import analytics.util as au
caffe_root = '/home/behzad/hed/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from PIL import Image
import numpy as np
import os
import cv2
from mss_segmentation import MSS


def demon(im, out):
    """
    Overlay an image with the segmentation mask
    :param im: input image
    :param out: segmentation mask
    :return: overlayed image
    """
    im = np.asarray(im, np.uint8)
    out = np.asarray(out, np.uint8)
    out = out / 255
    mask3 = np.transpose([out, out, out], [1, 2, 0])
    invmask3 = 1 - mask3

    label = [out, np.multiply(out, 255), out]  # green
    label = np.transpose(label, [1, 2, 0])
    imreg = np.multiply(im, mask3)

    x = np.multiply(imreg, 0.7) + np.multiply(label, 0.3)
    x = np.asarray(x, np.uint8)
    y = np.multiply(im, invmask3)
    z = np.asarray(x + y, np.uint8)

    return z



def segment_lesion(seg, im,overlay=False):
    """
    Lesion segmentation using MSS method
    :param im: input image
    :param seg: segmentation model
    :return: overlayed image or lesion mask
    """
    lesion_mask = seg.detect_lesion_border(im)

    if overlay:
        vizim = np.asarray(im)
        lesion_mask = demon(vizim, lesion_mask)


    return lesion_mask


if __name__ == "__main__":

    # Load the model
    model_root = os.getcwd()
    proto = model_root+'/resources/deploy_mss.prototxt'
    model = model_root+'/resources/mss.caffemodel'
    seg = MSS(proto, model)

    #Test a sample
    input_im = Image.open('/home/behzad/segmentation/sample_image.jpg')

    #Segment skin lesion
    lesion_mask=segment_lesion(seg,input_im,overlay=True)
    Image.fromarray(lesion_mask.astype(np.uint8)).save('lesion_map2.jpeg')





