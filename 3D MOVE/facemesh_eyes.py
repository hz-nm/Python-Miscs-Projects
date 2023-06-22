import cv2
import mediapipe as mp
import numpy as np

def get_relative(eye_landmarks, frame):
    eye_points = []
    for landmarks in eye_landmarks:
        for points in landmarks:
            x = points.x
            y = points.y
            shape = frame.shape
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])
            eye_points.append([relative_x, relative_y])
            cv2.circle(frame, (relative_x, relative_y), radius=1, color=(225, 0, 100), thickness=1)

    return(eye_points)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# The points that determine the eyes according to the facemesh library.
points_right_eye = [390, 373, 374, 380, 381, 382, 398, 384, 385, 386, 387, 388, 249]
points_left_eye = [246, 161, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163]
blinks_right = []
blinks_left = []
blinks = []
blink_count = 0

right_eye_points = []
left_eye_points = []
drowsiness_thresh = 10
ALERT = False

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print('Ignoring empty frame')
            # if loading a video use break
            continue
        # Flip the image horizontally for a later selfie-view display, and perform BGR to RGB conversion
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # cv2.imshow('Frame', frame)
        # to improve performance
        frame.flags.writeable = False
        result = face_mesh.process(frame)

        # Draw the face mesh annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Get the landmarks for both eyes.
        if result.multi_face_landmarks:
            right_eye = [[result.multi_face_landmarks[0].landmark[p]] for p in points_right_eye]
            left_eye = [[result.multi_face_landmarks[0].landmark[p]] for p in points_left_eye]

            # Drawing and finding out the relative co-ordinates.
            # RIGHT EYE
            right_eye_points = get_relative(eye_landmarks=right_eye, frame=frame)

        
            # LEFT EYE
            left_eye_points = get_relative(eye_landmarks=left_eye, frame=frame)

            # FOR THE RIGHT EYE
            points_r = np.array(right_eye_points, np.int32)
            points_r = points_r.reshape((-1, 1, 2))
            # cv2.polylines(frame, [points_r], True, (0, 255, 100), thickness=1)
            cv2.drawContours(frame, [points_r], -1, (0,255,255), 1)
            bbox = cv2.boundingRect(points_r)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color=(0,0,255), thickness=1)
            
            # FOR THE LEFT EYE
            points_l = np.array(left_eye_points, np.int32)
            points_l = points_l.reshape((-1, 1, 2))
            # cv2.polylines(frame, [points_l], True, (0, 255, 100), thickness=1)
            cv2.drawContours(frame, [points_l], -1, (0,255,255), 1)
            bbox = cv2.boundingRect(points_l)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color=(0,0,255), thickness=1)
            
            # calculating and printing the areas.
            poly_area_r = int(cv2.contourArea(points_r))
            # print("AREA OF RIGHT EYE = {}".format(poly_area_r))
            cv2.putText(frame, "Right Eye: {}".format(poly_area_r), (400, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1)

            poly_area_l = int(cv2.contourArea(points_l))
            # print("AREA OF LEFT EYE = {}".format(poly_area_l))
            cv2.putText(frame, "Left Eye: {}".format(poly_area_l), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1)

            # we'll start by counting the blinking.
            if poly_area_r < 400 or poly_area_l < 400:
                # blink_count += 1
                # print('blinked')
                cv2.putText(frame, 'BLINKED', (250, 150), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 100), 2)
            
            # To check for drowsiness we have to check consecutive frames to see if
            # all of them had the eyes closed..
            # Thanks to PYIMAGESEARCH for this idea! You the BOSS!!
            if poly_area_r < 400 or poly_area_l < 400:
                blink_count += 1
                if blink_count > drowsiness_thresh:
                    if not ALERT:
                        ALERT = True
                    cv2.putText(frame, 'WAKE UP!!!', (250, 250),
                                cv2.FONT_ITALIC, 2.0, (0,0,255), 3)
            else:                      
                blink_count = 0
                ALERT = False

            # print(blink_count)
            

            cv2.imshow('Are Ya Drowsy Son?', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()


# Points observed,
# In my camera resolution, the area of the polygon that detects the eyes 
# was found to be between less than 100 to a max of 110