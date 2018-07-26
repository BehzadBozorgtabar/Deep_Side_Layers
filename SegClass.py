# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:40:24 2017

@author: Behzad
"""

import numpy as np
from PIL import Image
import PIL.ImageOps
caffe_root = '/home/behzad/hed/'
import caffe
import os

class CaffeModel:
    """ Initialization """
    def __init__(self, proto_file, model_weights):
        """
        Initializes the model
        :param proto_file:  prototxt file 
        :param model_weights: weight model
        """
        self.net = caffe.Net(proto_file, model_weights, caffe.TEST)


    def resize_image(self,im, maxSize=800, isMask=False,LowerBound=False):
        """
        resize the image to  the given maximum size (either width or height)
        This function is copied from https://github.ibm.com/aur-mma/optic-disc-cup-segmentation
        :param im:
        :param maxSize:  the maximum size (either height or width) of the image after resize.
        :param isMask:   if true, the input image (im) is a binary mask image
        :return:
        """
        imsize = np.asarray(im.size)
        if(not LowerBound or (imsize[0] > maxSize or imsize[1] > maxSize)):
            scale = maxSize / max(imsize)
            newsize = np.round(imsize * scale)
            newsize = np.asarray(newsize, np.int64)
            if (not isMask):
                im = im.resize(newsize, Image.BILINEAR)
            else:
                im = im.resize(newsize, Image.NEAREST)
            return im
        else:
            return im

    def compute_map(self, im):
        """
        Compute the confidence map of the segmentation
        :param self.net: network model
        :param im: input image (RGB format)
        :return: final confidence map
        """
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        #mean 
        in_ -= np.array((145.4160,158.4950,184.6047)) 
        in_ = in_.transpose((2,0,1))
        self.net.blobs['data'].reshape(1, *in_.shape)
        self.net.blobs['data'].data[...] = in_
        self.net.forward()
        out2 = self.net.blobs['dsn2_loss'].data[0][0,:,:]
        out3 = self.net.blobs['dsn3_loss'].data[0][0,:,:]
        out4 = self.net.blobs['dsn4_loss'].data[0][0,:,:]
        out5 = self.net.blobs['dsn5_loss'].data[0][0,:,:]
        fuse = self.net.blobs['fuse-loss'].data[0][0,:,:]
        return fuse, out2, out3, out4, out5
        

    def segment_lesion(self, skin_image):
        """
        Apply threshold
        :param net: network parameter
        :param skin_image: input skin image (RGB format)
        :return: segmentation map
        """
        im=self.resize_image(skin_image)
        out = self.compute_map(im)
        out=out[0]
        out=np.ones((out.shape[0],out.shape[1]))-out
        out = (out > 0.6).astype(np.uint8)*255

        print np.shape(out), type(out)
        out = Image.fromarray(out).resize(skin_image.size, Image.NEAREST)



        return np.asarray(out)
