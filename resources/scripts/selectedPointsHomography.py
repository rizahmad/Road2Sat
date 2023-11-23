import cv2
import numpy as np
import os
from datetime import datetime
import NumpyArrayEncoder
from NumpyArrayEncoder import NumpyArrayEncoder
import json
import glob

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

def selectedPointsHomography():
    rootPath = '.\\dataset'
    genPath = '.\\gen'
    
    if not os.path.exists(rootPath):
        os.makedirs(genPath)

    roadImagePath = glob.glob(os.path.join(rootPath, 'roadref', '*'))[0]
    satelliteImagePath = glob.glob(os.path.join(rootPath, 'satref', '*'))[0]

    roadImage = cv2.imread(roadImagePath)
    satelliteImage = cv2.imread(satelliteImagePath)
    
    point_reader(roadImage)
    roadPoints = np.array(pointsInput)
    point_reader(satelliteImage)
    satellitePoints = np.array(pointsInput)

    # Estimate the homography matrix
    homography, _ = cv2.findHomography(roadPoints, satellitePoints, cv2.RANSAC, 5.0)
    print(homography)

    # Serialization and saving
    numpyData = {"homography": homography}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder, indent=4)
    f = open(os.path.join(genPath, "road2sat_homography.json"), "w")
    f.write(encodedNumpyData)
    f.close()

if __name__ == "__main__":
    selectedPointsHomography()