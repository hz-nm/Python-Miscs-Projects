import cv2
import mediapipe as mp
import numpy as np
import random

mp_drawing_styles = mp.solutions.drawing_styles

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

def random_polygon(points_all, shape):
    new_polygon = []
    while(len(new_polygon) <= shape):
        mark = random.randint(0, 441)
        if mark not in new_polygon:
            new_polygon.append([points_all[mark]])

    np_polygon = np.array(new_polygon, np.int32)
    np_polygon = np_polygon.reshape((-1, 1, 2))

    return np_polygon
        


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

all_points = []

# The points that determine the eyes according to the facemesh library.
points_right_eye = [390, 373, 374, 380, 381, 382, 398, 384, 385, 386, 387, 388, 249]
points_left_eye = [246, 161, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163]
eye_points = points_right_eye + points_left_eye

all_points = [i for i in range(468)]
print(len(all_points))

right_eye_points = []
left_eye_points = []

ALERT = False

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()

        # frame of shape = (480, 640, 3)
        frame_shape = (480, 640, 3)
        empty_frame = np.zeros(frame_shape)
        

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
        
        # all_landmarks = []
        # Get the landmarks for both eyes.
        if result.multi_face_landmarks:
            # print(result.multi_face_landmarks[0].landmark[0])
            
            all_landmarks = [[result.multi_face_landmarks[0].landmark[i]] for i in all_points if i not in eye_points]
            
            # Drawing and finding out the relative co-ordinates.
            all_landmarks_rel = get_relative(eye_landmarks=all_landmarks, frame=frame)
            # print(f'The Length = {len(all_landmarks_rel)}\n')
            # print('Frame: ')
            # print(all_landmarks_rel)
            # print('\n\n')
            
            for l in all_landmarks_rel:
                with open('facial_landmarks.txt', 'a') as f:
                    f.write(str(l))
                

            # shape = [[x, y], [x, y], [x, y]]
            polygon_1 = random_polygon(all_landmarks_rel, 3)
            polygon_2 = random_polygon(all_landmarks_rel, 4)

            # Getting the relative points, turning them into a numpy array and reshaping them into x and y points.
            points_all = np.array(all_landmarks_rel, np.int32)
            points_all = points_all.reshape((-1, 1, 2))     # -1 means that it is an unknown shape and we want numpy to figure it out


            # cv2.drawContours(frame, [points_all], -1, (0, 255, 0), 1)
            # bbox = cv2.boundingRect(points_all)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color=(255, 0, 100), thickness=1)
            # ! drawing the polygons
            cv2.polylines(frame, [polygon_1], color=(255, 0, 0), isClosed=True, thickness=1)
            cv2.polylines(frame, [polygon_2], color=(150, 0, 255), isClosed=True, thickness=1)
            cv2.fillPoly(frame, [polygon_1], color=(255, 0, 0))
            cv2.fillPoly(frame, [polygon_2], color=(150, 0, 255))
            cv2.imshow('FaceMESH', frame)
            
            for face_landmarks in result.multi_face_landmarks:
                points = mp_face_mesh.FACEMESH_TESSELATION
                print('\n\t\tPoints:')
                # frozenset({(x, y), (x, y) .... (x, y)})
                print(points)
                print('\n\n')

                mp_drawing.draw_landmarks(
                image=empty_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

            
            # cv2.drawContours(empty_frame, [points_all], -1, (0, 0, 255), 1)
            cv2.imshow('Black', empty_frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()


# Points observed,
# In my camera resolution, the area of the polygon that detects the eyes 
# was found to be between less than 100 to a max of 110