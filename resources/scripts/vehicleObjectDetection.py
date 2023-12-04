import cv2
import onnxruntime as ort
import numpy as np
import torch
import os
import sys
import json 
sys.path.append(os.path.join(os.getcwd(),'resources','models', 'YOLOP'))

from lib.core.general import non_max_suppression


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

def infer_yolop(weight=r"resources\models\YOLOP\weights\yolop-640-640.onnx", img_path = None, confidence_threshold=0.75):
    ort.set_default_logger_severity(4)
    onnx_path = fr".\{weight}"
    output_json = r".\\gen\\bounding_boxes.json"
    ort_session = ort.InferenceSession(onnx_path)
    print(f"Load {onnx_path} done!")

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    # for ii in inputs_info:
    #     print("Input: ", ii)
    # for oo in outputs_info:
    #     print("Output: ", oo)

    print("num outputs: ", len(outputs_info))

    save_det_path = f"./pictures/detect_onnx.jpg"

    img_bgr = cv2.imread(img_path)
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
    boxes = non_max_suppression(det_out, conf_thres=confidence_threshold)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)

    if boxes.shape[0] == 0:
        print(f"No bounding boxes detected above {confidence_threshold} confidence.")
        return

    # scale coords to original size.
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    print(f"Detect {boxes.shape[0]} bounding boxes above {confidence_threshold} confidence.")

    bounding_boxes = []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        bounding_box = {
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'confidence': float(conf),
            'label': int(label)
        }
        bounding_boxes.append(bounding_box)

    # Save bounding boxes to a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(bounding_boxes, json_file, indent=2)

    img_det = img_rgb[:, :, ::-1].copy()
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    cv2.imshow('Bounded Boxes', img_det)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
infer_yolop(img_path=r'C:\Users\Faizan\Road2Sat\dataset\frames\frame_000002661_w640_s10_bm1200_tm1000.jpg')
