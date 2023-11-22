import cv2
import os
import argparse
import imutils


def createFrames(videoPath, width):
    destinationPath = '.\\dataset\\frames'
    if not os.path.exists(destinationPath):
        os.makedirs(destinationPath)
    vidcap = cv2.VideoCapture(videoPath)
    success,image = vidcap.read()
    count = 1
    while success:
        resizedImage = imutils.resize(image, width)
        cv2.imwrite(os.path.join(destinationPath, "frame_%d.jpg" % count), resizedImage) 

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
    parser.add_argument('-w', '--width', help='Resize to the width', required=True)
    args = vars(parser.parse_args())

    createFrames(args['video'], int(args['width']))
