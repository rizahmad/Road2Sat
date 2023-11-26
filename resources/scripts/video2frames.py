import cv2
import os
import argparse
import imutils
import shutil


def createFrames(videoPath, width, stride, bottom_margin=0, top_margin=0):
    destinationPath = '.\\dataset\\frames'
    if os.path.exists(destinationPath):
         shutil.rmtree(destinationPath)
    os.makedirs(destinationPath)
    
    vidcap = cv2.VideoCapture(videoPath)
    success,image = vidcap.read()
    count = 1
    success = True
    while success:
        success,image = vidcap.read()
        if success:
            if bottom_margin != 0:
                image = image[0:image.shape[0]-bottom_margin,:,:]
            if top_margin != 0:
                image = image[top_margin:,:,:]
            resizedImage = imutils.resize(image, width)
            outfilename = "frame_%09d_w%d_s%d_bm%d_tm%d.jpg" % (count, width, stride, bottom_margin, top_margin)
            cv2.imwrite(os.path.join(destinationPath, outfilename), resizedImage) 
            print(outfilename, "generated")
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
    parser.add_argument('-bm', '--bottom_margin', help='Bottom margin to be cropped out', required=True)
    parser.add_argument('-tm', '--top_margin', help='Top margin to be cropped out', required=True)
    args = vars(parser.parse_args())

    createFrames(args['video'], int(args['width']), int(args['stride']), int(args['bottom_margin']), int(args['top_margin']))
