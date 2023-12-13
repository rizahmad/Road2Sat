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
import onnxruntime as ort


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
import json
from NumpyArrayEncoder import NumpyArrayEncoder

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
from lib.core.general import non_max_suppression


import torch
from torchvision import transforms
from PIL import Image

from glob import glob

# import evaluate
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)

import cv2
import numpy as np

def getObjects(imagePath, original_image, det_out):
    img_bgr = original_image.copy()
    height, width, _ = img_bgr.shape

    ort.set_default_logger_severity(4)
    weight="yolop-640-640.onnx"
    onnx_path = f"./resources/models/YOLOP/weights/{weight}"
    ort_session = ort.InferenceSession(onnx_path)


    img_bgr = cv2.imread(imagePath)
    height, width, _ = img_bgr.shape

    # convert to RGB
    img_rgb = img_bgr[:, :, ::-1].copy()

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

    img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, 0)  # (1, 3,640,640)

    # inference: (1,n,6) (1,2,640,640) (1,2,640,640)
    det_out = ort_session.run(['det_out'], input_feed={"images": img})[0]

    det_out = torch.from_numpy(det_out).float()
    boxes = non_max_suppression(det_out, conf_thres=0.75)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)

    if boxes.shape[0] == 0:
        print(f"No bounding boxes detected above 0.75 confidence.")
        return

    # scale coords to original size.
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    # print(f"Detect {boxes.shape[0]} bounding boxes above 0.75 confidence.")

    # Calculate the center pixel
    center_x = width / 2
    center_y = height / 2

    best_box = None
    best_distance = float('inf')

    best_confidence = None  # Variable to store the confidence of the best box
    frameName = imagePath.split('\\')[-1]
    
    
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)

        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2

        distance = calculate_distance(center_x, center_y, box_center_x, box_center_y)

        # Check if a box is detected and if it has higher confidence
        if best_confidence is None or (conf is not None and conf > best_confidence):
            best_box = boxes[i]
            best_confidence = conf
            best_distance = distance

    object_info = {frameName: {'x': -1, 'y': -1, 'confidence': 0, 'label': 0}}

    if best_box is not None:
        x1, y1, x2, y2, conf, label = best_box
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_det = img_rgb[:, :, ::-1].copy()
        base_center_x = (x1 + x2) // 2
        base_center_y = y2
        cv2.circle(img_det, (base_center_x, base_center_y), 5, (255, 0, 0), -1)

        object_info[frameName]['x'] = base_center_x
        object_info[frameName]['y'] = base_center_y
        object_info[frameName]['confidence'] = int(conf)
        object_info[frameName]['label'] = int(label)
    
    return object_info


def yolopRoadSegmention(imagePath, openingIterations, closingIterations):   
    source = imagePath
    original_image =  cv2.imread(source)
    

    device = torch.device('cpu')
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

    object = getObjects(imagePath, original_image, det_out)
    return result_image, object

def createRoadSegmentedFrames(framesPath, rs_framesPath, openingIterations, closingIterations, nframes):
    framePaths = glob(os.path.join(framesPath, '*'))
    framePaths.sort()
    objectsList = list()
    rsAlgo = yolopRoadSegmention

    nMax = 1000000
    if nframes != None:
        nMax = int(nframes)
    if len(framePaths) < nMax:
        nMax = len(framePaths)
    for p in framePaths[0:nMax]:
        segmentedImage, object = rsAlgo(p, int(openingIterations), int(closingIterations))
        name = p.split('\\')[-1].split('.')[0]
        ext = p.split('\\')[-1].split('.')[-1]
        cv2.imwrite(os.path.join(rs_framesPath, 'rs_'+name+'_oi_{}_ci_{}'.format(openingIterations,closingIterations)+'.{}'.format(ext)), segmentedImage)
        objectsList.append(object)
    
    # Save bounding boxes to a JSON file
    path_to_save  = '.\\dataset\\rs_frames'
    encodedNumpyData = json.dumps(objectsList, cls=NumpyArrayEncoder ,indent=4)
    f = open(os.path.join(path_to_save, "objects.json"), "w")
    f.write(encodedNumpyData)
    f.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='roadSegmentation',
                    description='Segment road from dash cam view',
                    epilog='V1.0')
    parser.add_argument('-oi', '--opening_iterations', help='Opening iterations', required=True)
    parser.add_argument('-ci', '--closing_iterations', help='Closing iterations', required=True)
    parser.add_argument('-n', '--nframes', help='number of frames', required=False)
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
                              args['nframes'])

