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
from resources.scripts.roadSegmentation import roadSegmentation


class Road2Sat:
    def __init__(self):
        self.genFolder = '.\\gen'
        self.datasetFolder = '.\\dataset'
        self.framesPath = os.path.join(self.datasetFolder, 'frames')
        self.roadrefPath = os.path.join(self.datasetFolder, 'roadref')
        self.satrefPath = os.path.join(self.datasetFolder, 'satref')
        self.p_framesPath = os.path.join(self.genFolder, 'p_frames')
        self.rs_framesPath = os.path.join(self.genFolder, 'rs_frames')
        self.road2SatHomographyFilename = "road2sat_homography.json"
        self.interframesHomographyFilename = "interframes_homography.json"

    def calculateCorrespondingPoints(self, srcPath, dstPath):
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

    def CalculateInterFrameHomography(self):
        f = open(os.path.join(self.genPath, self.road2SatHomographyFilename), "r")
        encodedNumpyData =f.read()
        f.close()
        decodedArrays = json.loads(encodedNumpyData)
        finalNumpyArray = np.asarray(decodedArrays["homography"])
        homography = finalNumpyArray
        print(homography)

        # read all frames
        # loop over and get corresponding points and calculate homography
        framePaths = glob.glob(os.path.join(self.framesPath, '*'))
        framePaths.sort()

        interframeHomography = list()
        runningHomography = homography
        for i, _ in enumerate(framePaths):
            frameName = framePaths[i].split('\\')[-1]
            if i == 0:
                interframeHomography.append({frameName:runningHomography})
            else:
                srcPoints, dstPpoints = self.CalculateCorrespondingPoints(framePaths[i], framePaths[i-1])
                h, _ = cv2.findHomography(srcPoints, dstPpoints, cv2.RANSAC, 5.0)
                runningHomography = np.matmul(runningHomography, h)
                interframeHomography.append({frameName:runningHomography})

        # Serialization and saving
        encodedNumpyData = json.dumps(interframeHomography, cls=NumpyArrayEncoder, indent=4)
        f = open(os.path.join(self.genPath, self.interframesHomographyFilename), "w")
        f.write(encodedNumpyData)
        f.close()

        self.interframeHomography = interframeHomography

        return self

    def CreateRoadSegmentedFrames(self):
        framePaths = glob.glob(os.path.join(self.framesPath, '*'))
        framePaths.sort()
        if not os.path.exists(self.rs_framesPath):
            os.makedirs(self.rs_framesPath)
        for p in framePaths:
            segmentedImage = roadSegmentation(p)
            frameName = p.split('\\')[-1]
            cv2.imwrite(os.path.join(self.rs_framesPath, 'rs_'+frameName), segmentedImage)
        return self
        
    def CreateProjectedFrames(self, roadSegmented=False):
        srcImagesPath = ''
        if roadSegmented:
            srcImagesPath = self.rs_framesPath
        else:
            srcImagesPath = self.framesPath
        
        framePaths = glob.glob(os.path.join(srcImagesPath, '*'))
        for i, p in enumerate(framePaths):
            img = cv2.imread(p)
            frameName = frameName = p.split('\\')[-1]
            h = self.interframeHomography[i]
            result = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))
            cv2.imwrite(os.path.join(self.p_framesPath, 'p_'+frameName), result)
        
        return self

    def CreateMosaic():
        pass

if __name__ == "__main__":
    r2s = Road2Sat()
    r2s.CalculateInterFrameHomography().CreateProjectedFrames()
    