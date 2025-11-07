import cv2
import sys
import random
# code adapted from tutorial here: https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2021-12-06--opencv-object-tracking/2021-12-06/#load-video-file
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = tracker_types[6]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy.TrackerCSRT_create()

# Change tracker_type index to check if objects are created:
print(tracker)

video = cv2.VideoCapture('test_face_tracking.mp4')
# load video
if not video.isOpened():
    print('[ERROR] video file not loaded')
    sys.exit()
# capture first frame
ok, frame = video.read()
if not ok:
    print('[ERROR] no frame captured')
    sys.exit()
print('[INFO] video loaded and frame capture started')

bbox = cv2.selectROI(frame)
print('[INFO] select ROI and press ENTER or SPACE')
print('[INFO] cancel selection by pressing C')
print(bbox)

ok = tracker.init(frame, bbox)
if not ok:
    print('[ERROR] tracker not initialized')
    sys.exit()
print('[INFO] tracker was initialized on ROI')
# random generate a colour for bounding box
colours = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
# loop through all frames of video file
while True:
    ok, frame = video.read()
    if not ok:
        print('[INFO] end of video file reached')
        break
    # update position of ROI based on tracker prediction
    ok, bbox = tracker.update(frame)
    # test print coordinates of predicted bounding box for all frames
    print(ok, bbox)

    if ok == True:
        (x, y, w, h) = [int(v) for v in bbox]
        # use predicted bounding box coordinates to draw a rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), colours, 3)
        cv2.putText(frame, str(tracker_type), (10, 30), cv2.QT_FONT_NORMAL, 1, (255, 255, 255))

    else:
        # if prediction failed and no bounding box coordinates are available
        cv2.putText(frame, 'No Track', (10, 30), cv2.QT_FONT_NORMAL, 1, (0, 0, 255))

    # display object track
    cv2.imshow('Single Track', frame)
    # press 'q' to break loop and close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video_codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    prefix = 'recording/'
    basename = "object_track.mp4"
    video_output = cv2.VideoWriter("_".join([prefix, basename]), video_codec, fps, (frame_width, frame_height))
    video_output.write(frame)