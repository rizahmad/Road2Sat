import cv2
import os
import argparse


def roadSegmentation(imagePath, modelChoice=1):
    # @Faizan make any changes needed. You can download model weights in ../model/
    # you can include an option to choose from different models or processes.
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='roadSegmentation',
                    description='Segment road from dash cam view',
                    epilog='V1.0')
    parser.add_argument('-i', '--input', help='Input frame path', requried=True)
    parser.add_argument('-m', '--model', help='Model selection', required=False)
    args = parser.parse_args()

    roadSegmentation(args['input'], modelChoice=1)
