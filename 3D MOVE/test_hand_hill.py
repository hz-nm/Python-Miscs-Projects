from sys import displayhook
import cv2
import mediapipe as mp

from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt

from scipy.fftpack import diff
import time

import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# First we will perform the required tasks for generating the hillshading.
demo_data = cbook.get_sample_data('jacksboro_fault_dem.npz', np_load=True)
z = demo_data['elevation']
nrows, ncols = z.shape

x = np.linspace(demo_data['xmin'], demo_data['xmax'], ncols)
y = np.linspace(demo_data['ymin'], demo_data['ymax'], nrows)
x, y = np.meshgrid(x, y)

region = np.s_[5:50, 5:50]
x, y, z = x[region], y[region], z[region]

# let's now set up a plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ls = LightSource(270, 45)

# using a custom hillshading mode
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)




# for static images
cap = cv2.VideoCapture(0)
image_cap = True

while image_cap:
    success, frame = cap.read()

    image_height, image_width, _ = frame.shape

    if success:
        image_cap = False
    else:
        print("Caught an Empty frame trying again")
    
print("Put HAND in front of camera for calibration of initial point.")
time.sleep(5)

avg_index_x = []
avg_index_y = []

count = 0

# Now for the VIDEO
with mp_hands.Hands(
    model_complexity = 1,
    min_detection_confidence = 0.6,
    min_tracking_confidence = 0.6,
    max_num_hands = 2) as hands:
    
    while cap.isOpened():
        success, frame = cap.read()

        # image_height, image_width, _ = frame.shape
        
        if not success:
            print("Ignoring Empty Camera Frame.")
            continue        # use break when loading a video from disk

        # To improve performance, mark the frames as unwriteable
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        # Draw the hand annotations in the image
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # print(hand_landmarks)
                # print(
                # f'Index finger Tip Co-ordinates: (',
                # f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                # f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                # )
                current_index_x = float('{:.2f}'.format(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width))
                current_index_y = float('{:.2f}'.format(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height))

                print(current_index_x)
                print(current_index_y)

                avg_index_x.append(current_index_x)
                avg_index_y.append(current_index_y)


        # flip the image horizontally

        cv2.imshow('Hands in Mediapipe', cv2.flip(frame, 1))
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break
        count += 1
        if count > 10:
            break



# Now find the average to find out the initial position of the index finger.

initial_x = float('{:.2f}'.format(sum(avg_index_x)/len(avg_index_x)))
initial_y = float('{:.2f}'.format(sum(avg_index_y)/len(avg_index_y)))
ini_x = 10
ini_y = 180

print("Initial Points")
print(initial_x, initial_y)

ax.view_init(ini_x, ini_y)

display = True



# Now for the VIDEO
with mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.6,
    min_tracking_confidence = 0.6,
    max_num_hands = 1) as hands:
    
    while cap.isOpened():
        success, frame = cap.read()

        # image_height, image_width, _ = frame.shape
        
        if not success:
            print("Ignoring Empty Camera Frame.")
            continue        # use break when loading a video from disk

        # To improve performance, mark the frames as unwriteable
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        # Draw the hand annotations in the image
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                current_index_x = float('{:.2f}'.format(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width))
                current_index_y = float('{:.2f}'.format(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height))

                print(current_index_x)
                print(current_index_y)

                print()
                print('X-INDEX_FINGER: {}'.format(current_index_x))
                print('Y-INDEX_FINGER: {}'.format(current_index_y))

            diff_x, diff_y = int(initial_x - current_index_x), int(initial_y - current_index_y)


            print(diff_x)
            print(diff_y)
            if diff_x > 180:
                diff_x = 180
            if diff_y > 180:
                diff_y = 180

            ax.view_init(ini_y + diff_y, ini_x + diff_x)

            plt.draw()
            plt.pause(0.1)


        # flip the image horizontally

        cv2.imshow('Hands in Mediapipe', cv2.flip(frame, 1))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

plt.show()

cap.release()