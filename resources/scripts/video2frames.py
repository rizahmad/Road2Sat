import cv2
import os
import argparse


def createFrames(videoPath, destinationPath):
    if not os.path.exists(destinationPath):
        os.makedirs(destinationPath)
    vidcap = cv2.VideoCapture(videoPath)
    success,image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite(os.path.join(destinationPath, "frame_%d.jpg" % count), image) 
        success,image = vidcap.read()
        if success:
            print("frame_%d.jpg generated" % count)
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Video2frames',
                    description='Creates frames from video',
                    epilog='V1.0')
    parser.add_argument('-v', '--video', help='Full path to video', required=True)
    parser.add_argument('-d', '--destination', help='Directory to generate frames into', required=True)
    args = vars(parser.parse_args())

    createFrames(args['video'], args['destination'])
