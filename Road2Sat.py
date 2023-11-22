# 21030005
# Rizwan Ahmad Bhatti

import cv2
import numpy as np
import os
import imutils
import argparse
from datetime import datetime

# Load the images

#rootPath = 'D:\mscs_lums\dev\mscs_cv\programming_assignments\PA3\project_files'

# satelliteImage = imutils.resize(cv2.imread(os.path.join(rootPath, 'satelliteImage1.jpg')), width = 600)
# roadImage = imutils.resize(cv2.imread(os.path.join(rootPath, 'roadImage1.jpg')), width = 600)

pointsInput = []

def onClickCallback(event, x, y, p1, p2):
    if event == cv2.EVENT_LBUTTONDOWN:
        pointsInput.append([x, y])

def point_reader(img):
    '''
    img - The image to be marked points on
    '''
    pointsInput.clear()
    cv2.imshow('point_reader', img)
    cv2.setMouseCallback('point_reader', onClickCallback)
    cv2.waitKey(0)

def roadHomography(roadImagePath, satelliteImagePath, rootPath):

    roadImage = imutils.resize(cv2.imread(os.path.join(roadImagePath)), width = 600)
    satelliteImage = imutils.resize(cv2.imread(os.path.join(satelliteImagePath)), width = 600)
    
    point_reader(roadImage)
    roadPoints = np.array(pointsInput)
    print(roadPoints)
    point_reader(satelliteImage)
    satellitePoints = np.array(pointsInput)
    print(satellitePoints)

    # Estimate the homography matrix
    homography, _ = cv2.findHomography(roadPoints, satellitePoints, cv2.RANSAC, 5.0)

    # Warp the first image using the homography
    result = cv2.warpPerspective(roadImage, homography, (2*roadImage.shape[1], 2*roadImage.shape[0]))
    outFilename = 'homography_'+roadImagePath.split('\\')[-1].split('.')[0]+'_'+satelliteImagePath.split('\\')[-1].split('.')[0]+'_'+datetime.now().strftime("%H-%M-%S")+'.jpg'
    print('File '+outFilename+' written')
    cv2.imwrite(os.path.join(rootPath, outFilename), result)

def main():
    currentWorkingDirectory = os.getcwd()
    parser = argparse.ArgumentParser(description='Road Homography calculator')
    parser.add_argument('-r','--roadImage', help='road image path', required=True)
    parser.add_argument('-s','--satelliteImage', help='satellite image path', required=True)
    args = vars(parser.parse_args())

    roadHomography(args['roadImage'], args['satelliteImage'], currentWorkingDirectory)
    

main()