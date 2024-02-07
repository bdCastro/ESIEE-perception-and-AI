#! pip install scikit-image
#%pip install ultralytics
#%pip install roboflow

import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from PIL import Image
import glob

from ultralytics import YOLO

# Importing yolo model with our pretrained weights
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model.load('bdCastro/ESIEE-perception-and-AI/final-project/data/best.pt')

#for filename in glob.glob('./data/WALK_Y-F0-B1_P287619_20200619084759_250/*.png'):
#    frame=cv2.imread(filename)
#    # Normalize the image
#    img_normalized = cv2.normalize(frame, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#    cv2.imwrite(filename, img_normalized)

for filename in glob.glob('./data/WALK_Y-F0-B1_P287619_20200619084759_250/*.png'):
    res = model.predict(filename)
    res_plotted = res[ 0 ].plot() # get results
    cv2.imshow('People', res_plotted)