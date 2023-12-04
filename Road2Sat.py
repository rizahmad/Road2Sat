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
import torch
import sys

from resources.scripts.NumpyArrayEncoder import NumpyArrayEncoder

sys.path.append(os.path.join(os.getcwd(),'resources','models', 'Stitch-images-using-SuperGlue-GNN'))
from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

class Road2Sat:
    def __init__(self, model, roadSegmentation, clean, verbose):
        # Dataset files
        self.datasetFolder = '.\\dataset'
        self.framesPath = os.path.join(self.datasetFolder, 'frames')
        self.roadrefPath = os.path.join(self.datasetFolder, 'roadref')
        self.satrefPath = os.path.join(self.datasetFolder, 'satref')
        self.rs_framesPath = os.path.join(self.datasetFolder, 'rs_frames')
        self.objectsFilename = "objects.json"
        self.objectsFilePath = os.path.join(self.rs_framesPath, self.objectsFilename)
        
        # Generated files
        self.genFolder = '.\\gen'
        self.p_framesPath = os.path.join(self.genFolder, 'p_frames')
        self.road2SatHomographyFilename = "road2sat_homography.json"
        self.interframesHomographyFilename = "interframes_homography.json"
        self.mosaicPath = os.path.join(self.genFolder, 'mosaic.jpg')
        self.transformedObjectsFilename = "projectedObjects.json"

        # Create gen folder directories if they do not exist
        if os.path.exists(self.p_framesPath):
            shutil.rmtree(self.p_framesPath)
        os.makedirs(self.p_framesPath)

        # Set configuration
        self.model = int(model)
        self.roadSegmentation = roadSegmentation
        self.clean = clean
        self.verbose = verbose

    def calculateCorrespondingPoints(self, srcImagePath, dstImagePath, srcRoiPath, dstRoiPath):
        
        # Load the images
        image1 = cv2.imread(srcImagePath)
        image2 = cv2.imread(dstImagePath)

        # Load the masks
        # https://stackoverflow.com/questions/42346761/opencv-python-feature-detection-how-to-provide-a-mask-sift
        srcRoi = cv2.imread(srcRoiPath)
        dstRoi = cv2.imread(dstRoiPath)

        # Convert images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        srcMask = cv2.cvtColor(srcRoi, cv2.COLOR_BGR2GRAY)
        dstMask = cv2.cvtColor(dstRoi, cv2.COLOR_BGR2GRAY)

        if self.model == 1:

            # Initialize the feature detector and extractor (e.g., SIFT)
            sift = cv2.SIFT_create()

            # Detect keypoints and compute descriptors for both images
            keypoints1, descriptors1 = sift.detectAndCompute(gray1, mask=srcMask)
            keypoints2, descriptors2 = sift.detectAndCompute(gray2, mask=dstMask)

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
        
        elif self.model == 2 :
            
            device ='cuda' if torch.cuda.is_available() else 'cpu'

            # Read images using cv2.imread
            # image0 = cv2.imread('dataset/frames/frame_000000001.jpg')
            # image1 = cv2.imread('dataset/frames/frame_000000011.jpg')
            image0,inp0,_ = read_image(path = srcImagePath , device = device, resize = [640, 480] , rotation = 0, resize_float = False)
            image1,inp1,_ = read_image(path = dstImagePath , device = device, resize = [640, 480], rotation = 0, resize_float = False)
            
            config = {
                'superpoint': {
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': -1
                },
                'superglue': {
                    'weights':'outdoor',
                    'sinkhorn_iterations': 100,
                    'match_threshold': 0.2,
                    }
                }
            matching = Matching(config).eval().to(device)

            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].detach().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            valid = matches > -1

            s_points = kpts0[valid]
            d_points = kpts1[matches[valid]]

            roi_src_points = []
            roi_dst_points = []
            for i, p in enumerate(s_points):
                x, y = int(p[0]), int(p[1])
                if srcMask[y, x] != 0:
                    roi_src_points.append([[x, y]])
                    roi_dst_points.append([[int(d_points[i][0]), int(d_points[i][1])]])
            
            src_points = np.array(roi_src_points)
            dst_points = np.array(roi_dst_points)
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
        
        rsframePaths = glob.glob(os.path.join(self.rs_framesPath, '*.jpg'))
        framePaths = glob.glob(os.path.join(self.framesPath, '*.jpg'))
        framePaths.sort()
        rsframePaths.sort()
        
        if self.clean:
            if os.path.exists(os.path.join(self.genFolder, self.interframesHomographyFilename)):
                os.remove(os.path.join(self.genFolder, self.interframesHomographyFilename))

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
            # read all frames
            # loop over and get corresponding points and calculate homography
            print(self.interframesHomographyFilename, 'shall be calculated')

            for i, _ in enumerate(framePaths[0:len(rsframePaths)]):
                frameName = framePaths[i].split('\\')[-1]
                if i == 0:
                    interframeHomography.append({frameName:runningHomography})
                else:
                    sourceImagePath = framePaths[i]
                    targetImagePath = framePaths[i-1]
                    srcRoiPath = rsframePaths[i]
                    dstRoiPath = rsframePaths[i-1]
                    if self.verbose:
                        print('Source image: {}', sourceImagePath)
                        print('Target image: {}', targetImagePath)
                        print('Source ROI image: {}', srcRoiPath)
                        print('Target ROI image: {}', dstRoiPath)
                    srcPoints, dstPoints = self.calculateCorrespondingPoints(sourceImagePath, targetImagePath, srcRoiPath, dstRoiPath, model)
                    h, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
                    runningHomography = runningHomography @ h
                    interframeHomography.append({frameName:runningHomography})

            # Serialization and saving
            encodedNumpyData = json.dumps(interframeHomography, cls=NumpyArrayEncoder, indent=4)
            f = open(os.path.join(self.genFolder, self.interframesHomographyFilename), "w")
            f.write(encodedNumpyData)
            f.close()

        self.interframeHomography = interframeHomography

        self.createProjectedFrames(framePaths, rsframePaths)

        return self
        
    def createProjectedFrames(self, framePaths, rsframePaths):
        print('Creating frame projections')
        objectsList = list()
        transformedObjectsLists = list()
        paths = None

        if os.path.exists(os.path.join(self.genFolder, self.transformedObjectsFilename)):
            os.remove(os.path.join(self.genFolder, self.transformedObjectsFilename))

        if self.roadSegmentation:
            paths = rsframePaths
            f = open(self.objectsFilePath, "r")
            encodedObjectsData =f.read()
            f.close()
            decodedObjectsData = json.loads(encodedObjectsData)
            for i in decodedObjectsData:
                frameId, obj = list(i.items())[0]
                location = [int(obj['x']), int(obj['y'])]
                label = obj['label']
                confidence = obj['confidence']
                objectsList.append({frameId:[location, label, confidence]})
        else:
            paths = framePaths[0:len(rsframePaths)]
        for i, p in enumerate(paths):
            # get the transformation
            h = list(self.interframeHomography[i].values())[0]
            frameName = frameName = p.split('\\')[-1]

            # transform the object locations
            point =list(objectsList[i].values())[0][0]
            point.extend([1])
            if point[0] != -1:
                transformedLocation = h@np.array(point)
                transformedLocation = (transformedLocation/transformedLocation[2]).astype('uint32')
            else:
                transformedLocation = point
            transformedObjectsLists.append({frameName:[transformedLocation, list(objectsList[i].values())[0][1], list(objectsList[i].values())[0][2]]})

            # transform images
            img = cv2.imread(p)
            resultWidth = 2*(img.shape[1])
            resultHeight = 2*(img.shape[0])
            result = cv2.warpPerspective(img, h, (resultWidth, resultHeight))
            cv2.circle(result, (transformedLocation[0], transformedLocation[1]), 5, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(self.p_framesPath, 'p_'+frameName), result)

        # put transformed files into a json file
        encodedtransformedObjectsLists = json.dumps(transformedObjectsLists, cls=NumpyArrayEncoder, indent=4)
        f = open(os.path.join(self.genFolder, self.transformedObjectsFilename), "w")
        f.write(encodedtransformedObjectsLists)
        f.close()

        self.createMosaic(self.p_framesPath)
        return self

    def createMosaic(self, srcImagesPath):
        print('Generating mosaic for images in', srcImagesPath)
        framePaths = glob.glob(os.path.join(srcImagesPath, '*'))
        mosaic = cv2.imread(framePaths[0])
        for i, p in enumerate(framePaths[1:-1]):
            gray1 = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
            img2 = cv2.imread(framePaths[i+1])
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
            mask1 = cv2.threshold(gray1, 0, 255,
	                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            mask2 = cv2.threshold(gray2, 0, 255,
	                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            mask = mask2-mask1
            mosaic[mask>5] = img2[mask>5]
            if self.verbose:
                cv2.imshow('Mosaic',mosaic)
                #cv2.imshow('mask', mask)
                cv2.waitKey()
            
           # mosaic = cv2.addWeighted(img1, 1,img2, 1, 0)
        cv2.imwrite(self.mosaicPath, mosaic)
        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Road2Sat',
                    description='Creates a mosaic from dash cam images',
                    epilog='V1.0')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-c', '--clean', action='store_true', required=False)
    parser.add_argument('-v', '--verbose', action='store_true', required=False)
    parser.add_argument('-rs', '--road_segmentation', action='store_true', required=False)
    args = vars(parser.parse_args())

    r2s = Road2Sat(int(args['model']),
                        args['road_segmentation'],
                        args['clean'],
                        args['verbose'])
    r2s.CalculateInterFrameHomography()
    
