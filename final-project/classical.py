import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from scipy import ndimage
from PIL import Image
import glob

directory = 'data/CROSS_X-F1-B1_P880043_20200625111459_225-2'
file_root = 'CROSS_X-F1-B1_P880043_20200625111459_225_cs001_'

# directory = 'data/PASS-2LBO_X-F3-B0_P848995_20210303145112_275'
# file_root = 'PASS-2LBO_X-F3-B0_P848995_20210303145112_275_cs013_'

# directory = 'data/FLOOR_Y-F1-B0_P220533_20201203141353_275'
# file_root = 'FLOOR_Y-F1-B0_P220533_20201203141353_275_cs010_'

def generate_background(path):
    image = cv2.imread(f'{directory}/{file_root}{path:05}.png')

    # Normalize the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
   
    # threshold the image
    # ret, image = cv2.threshold(image, 0.45, 1, cv2.THRESH_BINARY)

    # closing
    kernel = np.ones((7,7),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    image[0:150, 200:400] = 0
    image[400:512, 170:240] = 0

    # image[200:300, 300:500] = 0

    return image

background = generate_background(112)
cv2.imwrite(f'results/backgroung.png', background)

def preprocess(frame):
    image = cv2.imread(frame)

    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result = cv2.absdiff(image, background)

    # remove small noise
    kernel = np.ones((3,3),np.uint8)
    result = cv2.erode(result,kernel,iterations = 2)

    # threshold
    # ret, result = cv2.threshold(result, 0.05, 1, cv2.THRESH_BINARY)

    # closing
    kernel = np.ones((3,3),np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)

    return result

def main():
    # running the script on all images
    for filename in glob.glob(f'{directory}/*.png'):
        tag = filename.split('_')[-1] # get the tag of the image
        
        result = preprocess(filename)

        cv2.imwrite(f'results/no_background/{tag}', result)

        # select only the largest contour
        contours, hierarchy = cv2.findContours(result[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours by area
        contours = [c for c in contours if cv2.contourArea(c) > 500 and cv2.contourArea(c) < 15000]

        result = result.astype(np.uint8)

        # draw the largest contours
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        image = cv2.imread(filename)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # draw the bounding box
        for i, c in enumerate(contours):
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

            if(y < image.shape[1]/2):
                # rectangle on the top of the image
                cv2.rectangle(image, (x, y+h), (x+85, y+h+20), (0, 255, 0), -1)
                cv2.putText(image, f'person {i}', (x+5, y+h+15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            else:
                # rectangle on the bottom of the image
                cv2.rectangle(image, (x, y), (x+85, y-20), (0, 255, 0), -1)
                cv2.putText(image, f'person {i}', (x+5, y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.imwrite(f'results/{tag}', image)

if __name__ == "__main__":
    main()