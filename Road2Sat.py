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
import shutil

from resources.scripts.NumpyArrayEncoder import NumpyArrayEncoder

class Road2Sat:
    def __init__(self):
        # Dataset files
        self.datasetFolder = '.\\dataset'
        self.framesPath = os.path.join(self.datasetFolder, 'frames')
        self.roadrefPath = os.path.join(self.datasetFolder, 'roadref')
        self.satrefPath = os.path.join(self.datasetFolder, 'satref')
        self.rs_framesPath = os.path.join(self.datasetFolder, 'rs_frames')
        
        # Generated files
        self.genFolder = '.\\gen'
        self.p_framesPath = os.path.join(self.genFolder, 'p_frames')
        self.road2SatHomographyFilename = "road2sat_homography.json"
        self.interframesHomographyFilename = "interframes_homography.json"

        # Create gen folder directories if they do not exist
        if os.path.exists(self.p_framesPath):
            shutil.rmtree(self.p_framesPath)
        os.makedirs(self.p_framesPath)

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
        f = open(os.path.join(self.genFolder, self.road2SatHomographyFilename), "r")
        encodedNumpyData =f.read()
        f.close()
        decodedArrays = json.loads(encodedNumpyData)
        finalNumpyArray = np.asarray(decodedArrays["homography"])
        homography = finalNumpyArray
        print('Road to Satellite view homography loaded')

        interframeHomography = list()
        runningHomography = homography
        
        if os.path.isfile(os.path.join(self.genFolder, self.interframesHomographyFilename)):
            # File exits, load it
            print(self.interframesHomographyFilename, 'was found, loading contents')
            f = open(os.path.join(self.genFolder, self.interframesHomographyFilename), "r")
            encodedNumpyData =f.read()
            f.close()
            homographyList = json.loads(encodedNumpyData)
            finalNumpyArray = np.asarray(decodedArrays["homography"])

            for h in homographyList:
                k, v = list(h.items())[0]
                interframeHomography.append({k:np.asarray(v)})
            
        else:
            # File does not exist, calculation needed
            # read all frames
            # loop over and get corresponding points and calculate homography
            print(self.interframesHomographyFilename, 'was not found, calculating')
            framePaths = glob.glob(os.path.join(self.framesPath, '*'))
            framePaths.sort()
            for i, _ in enumerate(framePaths):
                frameName = framePaths[i].split('\\')[-1]
                if i == 0:
                    interframeHomography.append({frameName:runningHomography})
                else:
                    srcPoints, dstPpoints = self.calculateCorrespondingPoints(framePaths[i], framePaths[i-1])
                    h, _ = cv2.findHomography(srcPoints, dstPpoints, cv2.RANSAC, 5.0)
                    runningHomography = np.matmul(runningHomography, h)
                    interframeHomography.append({frameName:runningHomography})

            # Serialization and saving
            encodedNumpyData = json.dumps(interframeHomography, cls=NumpyArrayEncoder, indent=4)
            f = open(os.path.join(self.genFolder, self.interframesHomographyFilename), "w")
            f.write(encodedNumpyData)
            f.close()

        self.interframeHomography = interframeHomography

        return self
        
    def CreateProjectedFrames(self, roadSegmented=False):
        srcImagesPath = ''
        if roadSegmented:
            srcImagesPath = self.rs_framesPath
        else:
            srcImagesPath = self.framesPath
        
        print('Generating projections for images in', srcImagesPath)
        framePaths = glob.glob(os.path.join(srcImagesPath, '*'))
        for i, p in enumerate(framePaths):
            img = cv2.imread(p)
            frameName = frameName = p.split('\\')[-1]
            h = list(self.interframeHomography[i].values())[0]
            result = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))
            cv2.imwrite(os.path.join(self.p_framesPath, 'p_'+frameName), result)
        
        return self

    def CreateMosaic():
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Road2Sat',
                    description='Creates a mosaic from dash cam images',
                    epilog='V1.0')
    parser.add_argument('-rs', '--roadsegmentation', action='store_true', required=False)
    args = vars(parser.parse_args())

    r2s = Road2Sat()
    r2s.CalculateInterFrameHomography().CreateProjectedFrames(roadSegmented=args['roadsegmentation'])
        
