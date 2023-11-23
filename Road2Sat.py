# 21030005
# Rizwan Ahmad Bhatti

import cv2
import numpy as np
import os
import imutils
import argparse
from datetime import datetime
import json
import glob
from resources.scripts.NumpyArrayEncoder import NumpyArrayEncoder

def getCorrespondingPoints(srcPath, dstPath):
    # Load the images
    image1 = cv2.imread(srcPath)
    image2 = cv2.imread(dstPath)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize the feature detector and extractor (e.g., SIFT)
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Initialize the feature matcher using brute-force matching
    bf = cv2.BFMatcher()

    # Match the descriptors using brute-force matching
    matches = bf.match(descriptors1, descriptors2)

    # Select the top N matches
    num_matches = 50
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

    # Extract matching keypoints
    src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    return src_points, dst_points

def roadHomography():
    genPath = '.\\gen'
    datasetPath = ".\\dataset\\frames"

    f = open(os.path.join(genPath, "road2sat_homography.json"), "r")
    encodedNumpyData =f.read()
    f.close()
    decodedArrays = json.loads(encodedNumpyData)
    finalNumpyArray = np.asarray(decodedArrays["homography"])
    homography = finalNumpyArray
    print(homography)

    # read all frames
    # loop over and get corresponding points and calculate homography
    framePaths = glob.glob(os.path.join(datasetPath, '*'))
    framePaths.sort()

    frameHomography = list()
    runningHomography = homography
    for i, _ in enumerate(framePaths):
        frameName = framePaths[i].split('\\')[-1]
        if i == 0:
            frameHomography.append({frameName:runningHomography})
        else:
            srcPoints, dstPpoints = getCorrespondingPoints(framePaths[i], framePaths[i-1])
            h, _ = cv2.findHomography(srcPoints, dstPpoints, cv2.RANSAC, 5.0)
            runningHomography = np.matmul(runningHomography, h)
            frameHomography.append({frameName:runningHomography})

    # Serialization and saving
    encodedNumpyData = json.dumps(frameHomography, cls=NumpyArrayEncoder, indent=4)
    f = open(os.path.join(genPath, "interframes_homography.json"), "w")
    f.write(encodedNumpyData)
    f.close()

def main():
    roadHomography()

main()