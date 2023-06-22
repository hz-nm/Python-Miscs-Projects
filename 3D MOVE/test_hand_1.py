# This is for holistic and not just the hand.

from sys import getdefaultencoding
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hollistic = mp.solutions.holistic

with mp_hollistic.Holistic(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera Frame")
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame)

        # Draw landmarks annotation on the image
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_hollistic.FACEMESH_CONTOURS,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles
            .get_default_face_mesh_contours_style()
        )

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_hollistic.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Flip the image horizontally for a selfie view
        cv2.imshow('MediaPipe Holistic', cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()