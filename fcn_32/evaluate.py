# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:40:52 2016

@author: Behzad
"""


import numpy as np
from PIL import Image

import caffe
import matplotlib.pyplot as plt
import os
#from sklearn.metrics import jaccard_similarity_score
#from sklearn.metrics import f1_score


caffe.set_mode_cpu()
#caffe.set_device(0)
#caffe.set_mode_gpu()

# load net
#net = caffe.Net('vgg16_fcnn_deploy.prototxt', '/data/HeB/FC/share/fc8_iter_24000.caffemodel',caffe.TEST)
net = caffe.Net('vgg16_fcnn_deploy.prototxt', '/data/HeB/FC/share/fcn_iter_100000.caffemodel',caffe.TEST)

# Resize image
def resizeimage(im):
    imsize=np.asarray(im.size)
    scale=800.0/max(imsize)
    newsize=np.round(imsize*scale)
    newsize=np.asarray(newsize,np.int64)
    return im.resize(newsize)

# Overlay an image with segmentation prediction
def  demon(im,out):
    im=np.asarray(im,np.uint8)
    mask3=np.transpose([out, out, out],[1,2,0])
    invmask3=1-mask3
   
    
    label=[out, np.multiply(out,255), out] #green 
    label = np.transpose(label,[1,2,0])
    imreg=np.multiply(im,mask3)
    
    x=np.multiply(imreg,0.7)+ np.multiply(label,0.3)
    x=np.asarray(x,np.uint8)
    y=np.multiply(im,invmask3)
    z=np.asarray(x+y,np.uint8)
    
    return z


# Resize image
def dispFilters(net,filt):
    
    resp= net.blobs[filt].data[0]
    for i in range(1,128):
        plt.subplot(2,10,i)
        plt.imshow(resp[i,:,:], cmap='Greys_r')

 # Segmentation
def  doSeg(net,im):
    in_ = np.array(im, dtype=np.float32)
    #transformed_image = transformer.preprocess('data', im)
    in_ = in_[:,:,::-1]
    #in_ -= np.array((124.5789, 147.0185, 192.1043))
    in_ -= np.array((145.4160,158.4950,184.6047))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data

   # net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].reshape(1, *in_.shape)
    #net.blobs['data'].data[...] = in_
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    #return out
    return out
    
def doSegDisp(net, impath):
    im=Image.open(impath)
    im=resizeimage(im)
    in_ = np.array(im, dtype=np.float32)
    out= doSeg(net, in_)
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.subplot(1,2,2)
    #plt.imshow(demon(im,out))
    return out
 

def layer_name(net):
    
    list=[]
    for layer_name, blob in net.blobs.iteritems():
        list.append(layer_name)
    return list    
          
def col(feature):
    combined_x     = np.array([])
    for fmap in feature:
            combined_x = np.concatenate((combined_x,fmap),axis=1) 
    return combined_x
    

def list_files(path):
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files
    
def test(net):
    #Path to original images
    #impath  = '/data/For Suman/new/Skin_Training_Data'
    impath='/data/HeB/FC/share/all'
    #gtpath='/data/For Suman/new/test-gt'
    #gtpath= '/data/For Suman/new/Skin_Training_GT_bw'
    #Path for saving prediction scores
    out_score='/data/HeB/FC/out_score'
    #Path to save overlaied images
    overlayed='/data/HeB/FC/overlayed'
    #Path to save binary maps
    out_map='/data/HeB/FC/out_map'

    images=list_files(impath)
    jaccard=[]
    for imf in images:
        im=Image.open(os.path.join(impath,imf))
        print imf
        gt=Image.open(os.path.join(gtpath,imf[:-4]+'_Segmentation'+'.png'))   
        gt=Image.open(os.path.join(gtpath,imf[:-4]+'.png'))        
        gt=np.asarray(gt,np.uint8)       
       # Prediction
        out=doSeg(net,im)
        out=np.asarray(out,np.uint8)
        resp=net.blobs['score'].data[0]
        output=net.blobs['score'].data[0].argmax(axis=0)
        imreg=np.multiply(output,resp[1,:,:])
        #Jaccard similarity index
        jaccard_index = jaccard_similarity_score(gt, out, normalize=True)
        print "\nJaccard similarity score: " +str(jaccard_index)
        jaccard.append(jaccard_index)
        #F1 score
        #F1_score = f1_score(gt,out, labels=None, average='binary', sample_weight=None)
        #print "\nF1 score (F-measure): " +str(F1_score)
         
        #Saving outputs 
        outim=Image.fromarray(demon(im,out))
        outim.save(os.path.join(overlayed,imf[:-3]+'png'))
        out= Image.fromarray(out)
        imreg=Image.fromarray(imreg)
        imreg.save(os.path.join(out_score,imf[:-3]+'tiff'))
        out.save(os.path.join(out_map,imf[:-3]+'png'))
        
        
    #Print Jaccard index
    print 'jaccard' + str(np.mean(jaccard))


    
if __name__ == '__main__':
  test(net)

