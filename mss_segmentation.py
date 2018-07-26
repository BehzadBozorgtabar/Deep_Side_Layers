# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:37:55 2017

@author: Behzad
"""
from SegClass import CaffeModel
from PIL import Image
import numpy as np
import os


class MSS:
    def __init__(self, proto_file, model_weights):
        """
        Initializes the model

        :param proto_file:  prototxt file of the trained model
        :param model_weights: weight file of the trained model
        """
        self.model_mss = CaffeModel(
            proto_file,model_weights)



    def detect_lesion_border(self,skin_image):
        """
        Detect the lesion border
        :param skin_image:
        :return: mask
        """
        lesion_mask= self.model_mss.segment_lesion(skin_image)
        return lesion_mask