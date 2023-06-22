# This script will serve as a revision of what we had previously accomplished..
# According to results, main_test2.py seems to be giving the best results on single
# lane as was acheived previously. So we are going to review that over here and 
# then carry forward to enhancing the script even further.

from re import match
import time
import math

from numpy.lib.twodim_base import mask_indices
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import numpy as np

from yolov3_tf2.models import YoloV3, YoloV3Tiny

path_to_classes = './data/coco.names'
path_to_weights = './checkpoints/yolov3.tf'

path_to_output_vid = 'output.avi'
num_classes = 80

yolo = YoloV3(classes = num_classes)
yolo.load_weights(path_to_weights)
print('Weights Loaded')

class_names = [c.strip() for c in open(path_to_classes).readlines()]
print('Classes Loaded')

times = []

cap = cv2.VideoCapture('highway.mp4')
# available options in video
# highway
# tracking_ex -- For testing tracking

success, frame = cap.read()

if success:
    frame = cv2.resize(frame, (0,0), fx=1.0, fy=1.0)
    roi = cv2.selectROI('Select ROI', frame, showCrosshair=False)

# getting roi
# FOUR Points are given, x1 and x2 and y1 and y2
roi_x = roi[0]
roi_w = roi[0] + roi[2]

roi_y = roi[1]
roi_h = roi[1] + roi[3]

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)

mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, (roi_x, roi_y), (roi_w, roi_h), 255, -1)

centroid_prev = 0

# Initializing bounding boxes
bbox_with_ids = {
    0: [100, 100, 100, 100]
}
box_prev = [100, 100, 100, 100]
count = 0

while True:
    success, frame = cap.read()

    if not success:
        print('Video Ended')
        break

    t1 = time.time()
    frame = cv2.resize(frame, (0,0), fx=1.0, fy=1.0)

    # using a mask to block unwanted area which may in turn increase FPS
    # BUT it doesn't
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    frame_p = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    frame_p = tf.expand_dims(frame_p, axis=0)    # adds a new dimension at specified axis/column - in this case it is 0
    frame_p = transform_images(frame_p, 416)    # dimesion for applying YOLO

    boxes, scores, classes, nums = yolo.predict(frame_p)
    t2 = time.time()
    times.append(t2-t1)
    fpss = int(1/(t2-t1))
    fpss = str(fpss)

    frame, bbox = draw_outputs(frame, (boxes, scores, classes, nums), class_names)

    if not bbox and count < 5:
        count += 1
        print("#_#_#_#_#_#")
        print(count)
        print("#_#_#_#_#_#")
        box = box_prev

        centroid_x = int((box[0] + box[2])/2)
        centroid_y = int((box[1] + box[3])/2)

        for id in bbox_with_ids:
            box_previous = bbox_with_ids[id]
            centroid_x_pre = int((box_previous[0] + box_previous[2])/2)
            centroid_y_pre = int((box_previous[1] + box_previous[3])/2)

            cent_diff_x = centroid_x - centroid_x_pre
            cent_diff_y = centroid_y - centroid_y_pre
            
            centroid_sq = cent_diff_x**2 + cent_diff_y ** 2

            # treating the centroid as a vector, we can then calculate the magnitude
            # of the centroid and then compare it to out previous one.
            # this will also give us a direction for the detected object which can
            # also be used for tracking purposes.

            centroid_magnitude = math.sqrt(centroid_sq)
            print('Magnitude of CENTROID')
            print(centroid_magnitude)

            # centroid should only be used to track a single object.
            # so work first on isolating an object!

            # Trying BETTER COMMENTS
            # ! ALERT
            # TODO This is a todo
            # ? Do a query with this...
            # * AWESOME!
            
            try:
                centroid_dir = math.atan(cent_diff_y/cent_diff_x)
                centroid_dir = math.degrees(centroid_dir)
            except ZeroDivisionError:
                print('Same Object since they seem to be moving in the same direction')

            print('Direction/Angle of Centroid (Absolute): {}'.format(str(abs(centroid_dir))))

            diff_centroid = centroid_magnitude - centroid_prev
            if diff_centroid > 50:
                id += 1
            
        centroid_prev = centroid_magnitude


        bbox_with_ids[id] = box
        print('THE DICT')
        print(bbox_with_ids)
        print('-------------------------')
        print('')

        cv2.circle(frame, (centroid_x, centroid_y), 3, (0, 255, 0), thickness=-1)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(frame, 'id: {}'.format(id), (box[0], box[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    predictions = []
    preds_l = []
    centroids = []

    i = 0

    for box in bbox:
        # separate each detected object for tracking
        print('Detected Object')

        id_bbox = []

        
        centroid_x = int((box[0] + box[2])/2)
        centroid_y = int((box[1] + box[3])/2)

        for id in bbox_with_ids:
            box_previous = bbox_with_ids[id]
            centroid_x_pre = int((box_previous[0] + box_previous[2])/2)
            centroid_y_pre = int((box_previous[1] + box_previous[3])/2)

            cent_diff_x = centroid_x - centroid_x_pre
            cent_diff_y = centroid_y - centroid_y_pre

            centroid_sq = cent_diff_x**2 + cent_diff_y**2

            # again calculating centroid as vector to determine magnitude and direction
            centroid_magnitude = math.sqrt(centroid_sq)
            print('Magnitude of Centroid: {}'.format(centroid_magnitude))


            try:
                centroid_dir = math.atan(cent_diff_y/cent_diff_x)
                centroid_dir = math.degrees(centroid_dir)
            except ZeroDivisionError:
                print('Same Object?')

            print('Direction/Angle of Centroid (ABS): {}'.format(str(abs(centroid_dir))))

            diff_centroid = centroid_magnitude - centroid_prev

            query_im = frame[box_previous[1]:box_previous[1]+box_previous[3], box_previous[0]:box_previous[0]+box_previous[2]]
            test_im = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

            cv2.imshow('QUERY', test_im)
            cv2.waitKey(10)

            query_gray = cv2.cvtColor(query_im, cv2.COLOR_BGR2GRAY)
            test_gray = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)

            orb = cv2.ORB_create()

            query_keypoints, query_descriptors = orb.detectAndCompute(query_im, None)

            train_keypoints, train_descriptors = orb.detectAndCompute(test_im, None)

            matcher = cv2.BFMatcher()
            matches = matcher.match(query_descriptors, train_descriptors)

            print('MATCHES: {}'.format(len(matches)))

            # So matching is working good for sure...
            # Need to isolate each and every detection and their descriptors and then assign them to a variable such that, same object has same descriptions attached to it.
            # A dictionary might do the trick maybe?

            if diff_centroid > 50:
                id += 1
        
        centroid_prev = centroid_magnitude

        bbox_with_ids[id] = box
        print('THE DICT')
        print(bbox_with_ids)
        print('----------------')
        print('')

        cv2.circle(frame, (centroid_x, centroid_y), 3, (0, 255, 0), thickness=-1)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(frame, 'id: {}'.format(id), (box[0], box[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        box_prev = box

    frame = cv2.putText(frame, 'FPS: {}'.format(fpss), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Output', frame)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

