from __future__ import print_function
import cv2 as cv
import argparse

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    roosts = roost_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in roosts:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        roostROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        # eyes = eyes_cascade.detectMultiScale(roostROI)
        # for (x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2)*0.25))
        #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    cv.imshow('Capture - Roost detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--roost_cascade', help='Path to roost cascade.', default='cascade-training/roost-train-cascade-3/cascade.xml')
parser.add_argument('--input', help='Path to image file or camera device number.', default='cascade-training/images/test_image2.jpg')
#parser.add_argument('--input', help='Path to image file or camera device number.', default='0')
args = parser.parse_args()

roost_cascade_name = args.roost_cascade
#eyes_cascade_name = args.eyes_cascade

roost_cascade = cv.CascadeClassifier()
#eyes_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not roost_cascade.load(cv.samples.findFile(roost_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
# if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
#     print('--(!)Error loading eyes cascade')
#     exit(0)

# Check if input is a number (camera device) or a path
try:
    camera_device = int(args.input)
    is_camera = True
except ValueError:
    is_camera = False
    image_path = args.input

if is_camera:
    #-- 2. Read the video stream
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        detectAndDisplay(frame)

        if cv.waitKey(10) == 27:
            break
else:
    # Read single image
    frame = cv.imread(image_path)
    if frame is None:
        print(f'--(!)Error loading image {image_path}')
        exit(0)
    
    detectAndDisplay(frame)
    # Wait for key press before closing
    cv.waitKey(0)