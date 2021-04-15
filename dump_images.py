#!/usr/bin/python3
import sys
import cv2
from skimage.metrics import structural_similarity
import imutils
import numpy

def similarity(image_a, image_b):
    grayA = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    #print(score)
    return score

def is_similar_enough(imageA, imageB):
    return similarity(imageA, imageB) > 0.95
    
# 1500, 230, (330, 330)
x=1500
y=230
w=330
h=330

x1=210
y1=85
w1=40
h1=30

# read video
def dump_images():
    video_capture = cv2.VideoCapture(sys.argv[1])
    success,image = video_capture.read()
    previous_image = image
    count = 0
    success = True
    while success:
        success,image = video_capture.read()
        if not success:
            print('Done!')
            break
        # calculate similarity
        if not is_similar_enough(image[y:y+h, x:x+w], previous_image[y:y+h, x:x+w]):
            print('write file frame%000d.jpg'%count)
            cv2.imwrite("frame%000d.jpg" % count, image[y:y+h, x:x+w])     # save frame as JPEG file
            #cv2.imwrite("FirstSocket%000d.jpg" % count, image[y+y1:y+y1+h1, x+x1:x+x1+w1])     # save frame as JPEG file
            #cv2.imwrite("SecondSocket%000d.jpg" % count, image[y+y1:y+y1+h1, x+x1+w1:x+x1+w1+w1])     # save frame as JPEG file
            #cv2.imwrite("ThirdSocket%000d.jpg" % count, image[y+y1:y+y1+h1, x+x1+w1+w1:x+x1+w1+w1+w1])     # save frame as JPEG file
        count += 1
        previous_image = image

def extract_text(image):
    print(image)


if __name__ == "__main__":
    dump_images()
    