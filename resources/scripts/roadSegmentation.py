import cv2
import os
import argparse


def roadSegmentation(videoPath, destinationPath):
    # @Faizan make any changes needed. You can download model weights in ../model/
    # you can include an option to choose from different models or processes.
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='roadSegmentation',
                    description='Segment road from dash cam view',
                    epilog='V1.0')
    parser.add_argument('-i', '--input', help='Input frame')
    args = parser.parse_args()

    roadSegmentation(args['input'],)
