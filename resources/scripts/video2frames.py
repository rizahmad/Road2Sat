import cv2
import os
import argparse
import imutils


def createFrames(videoPath, width, stride):
    destinationPath = '.\\dataset\\frames'
    if not os.path.exists(destinationPath):
        os.makedirs(destinationPath)
    vidcap = cv2.VideoCapture(videoPath)
    success,image = vidcap.read()
    count = 1
    success = True
    while success:
        success,image = vidcap.read()
        if success:
            resizedImage = imutils.resize(image, width)
            cv2.imwrite(os.path.join(destinationPath, "frame_%09d.jpg" % count), resizedImage) 
            print("frame_%09d.jpg generated" % count)
            count += 1
        
        skipCounter = stride - 1
        while skipCounter > 0:
            skipCounter = skipCounter - 1
            _1,_2 = vidcap.read()
            count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Video2frames',
                    description='Creates frames from video',
                    epilog='V1.0')
    parser.add_argument('-v', '--video', help='Full path to video', required=True)
    parser.add_argument('-w', '--width', help='Resize to the width', required=True)
    parser.add_argument('-s', '--stride', help='Stride for skipping frames', required=True)
    args = vars(parser.parse_args())

    createFrames(args['video'], int(args['width']), int(args['stride']))
