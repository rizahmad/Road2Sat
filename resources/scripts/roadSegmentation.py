import cv2
import os
import argparse
from IPython.display import HTML
from base64 import b64encode
from PIL import Image
from torchvision import transforms
import os, sys
import numpy as np
import cv2
import pickle
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# from transformers import AutoProcessor, CLIPSegForImageSegmentation
import torchvision
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
import shutil

# https://stackoverflow.com/questions/21259070/struggling-to-append-a-relative-path-to-my-sys-path
# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..\models', 'YOLOP'))

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane

import torch
from torchvision import transforms
from PIL import Image

from glob import glob

# import evaluate



def yolopRoadSegmention(imagePath, openingIterations, closingIterations):
    # @Faizan make any changes needed. You can download model weights in ../model/
    source = imagePath
    original_image =  cv2.imread(source)
    
    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger,'cpu')
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    half = device.type != 'cpu'  # half precision only supported on CUDA
    transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])



    model = get_net(cfg)
    checkpoint = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..\models', 'YOLOP', 'weights', 'End-to-end.pth'), map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16


    dataset = LoadImages(source, img_size=640)

    # Choose one image from the dataset
    path, img, img_det, vid_cap, shapes = next(iter(dataset))

    # Transform the image
    img = transform(img).to(device)
    img = img.half() if half else img.float()

    # If the image has 3 dimensions, add one more dimension (batch dimension)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Run the model
    det_out, da_seg_out, ll_seg_out = model(img)

    # Process the segmentation outputs
    inf_out, _ , _ = det_out
    det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
    det = det_pred[0]

    _, _, height, width = img.shape
    h, w, _ = img_det.shape

    pad_w, pad_h = shapes[1][1]
    pad_w = int(pad_w)
    pad_h = int(pad_h)
    ratio = shapes[1][0][1]

    da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

    ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    # Process the segmentation masks and overlay them on the original image
    palette = np.random.randint(0, 255, size=(2, 2))
    palette[0] = [0, 0]
    palette[1] = [0, 255]
    palette = np.array(palette)
    assert palette.shape[0] == 2  # len(classes)
    assert palette.shape[1] == 2
    assert len(palette.shape) == 2

    result = (da_seg_mask, ll_seg_mask)
    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    color_area[result[0] == 1] = [0, 255, 0]
    color_area[result[1] == 1] = [0, 255, 0]

    color_seg = color_area[..., ::-1]
    color_mask = np.mean(color_seg, 2)


    # img_r = cv2.resize(img_r, (1280, 720), interpolation=cv2.INTER_LINEAR)

    # Display the result
    color_mask = cv2.resize(color_mask, (original_image.shape[1], original_image.shape[0]))

    k = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=np.uint8)
    # remove the tiny artefacts
    color_mask = cv2.erode(color_mask, k, iterations=openingIterations)
    color_mask = cv2.dilate(color_mask, k, iterations=openingIterations)
    # close gaps in road
    color_mask = cv2.dilate(color_mask, k, iterations=closingIterations)
    color_mask = cv2.erode(color_mask, k, iterations=closingIterations)

    result_image = original_image.copy()
    result_image[color_mask == 0] = 0

    return result_image

def createRoadSegmentedFrames(framesPath, rs_framesPath, openingIterations, closingIterations, model=1 ):
        framePaths = glob(os.path.join(framesPath, '*'))
        framePaths.sort()
        if model == 1:
            rsAlgo = yolopRoadSegmention
        for p in framePaths:
            segmentedImage = rsAlgo(p, int(openingIterations), int(closingIterations))
            frameName = p.split('\\')[-1]
            cv2.imwrite(os.path.join(rs_framesPath, 'rs_oi'+openingIterations+'_ci'+closingIterations+'_'+frameName), segmentedImage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='roadSegmentation',
                    description='Segment road from dash cam view',
                    epilog='V1.0')
    parser.add_argument('-m', '--model', help='Model selection', required = False)
    parser.add_argument('-oi', '--opening_iterations', help='Opening iterations', required=True)
    parser.add_argument('-ci', '--closing_iterations', help='Closing iterations', required=True)
    args = vars(parser.parse_args())
    
    source = '.\\dataset\\frames'

    destinationPath = '.\\dataset\\rs_frames'
    if os.path.exists(destinationPath):
         shutil.rmtree(destinationPath)
    os.makedirs(destinationPath)

    createRoadSegmentedFrames(source,
                              destinationPath, 
                              args['opening_iterations'], 
                              args['closing_iterations'], 
                              model=1)

