import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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
    
# Now the mediapipe stuff
# with mp_hands.Hands(
#     static_image_mode = True,
#     max_num_hands = 2,
#     min_detection_confidence = 0.5) as hands:

#     image = cv2.flip(frame, 1)
#     # Convert from BGR to RGB for processing
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # print handedness and draw landmarks on the image.
#     print("Handedness: ", results.multi_handedness)
#     if not results.multi_hand_landmarks:
#         pass

#     image_height, image_width, _ = image.shape

#     annotated_image = image.copy

#     for hand_landmarks in results.multi_hand_landmarks:
#         print('hand_landmarks: ', hand_landmarks)
#         print(
#             f'Index finger Tip Co-ordinates: (',
#             f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#             f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#         )

#         mp_drawing.draw_landmarks(
#             annotated_image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style()
#         )

#         cv2.imwrite(
#             '/tmp/annotated_image' + str(1) + '.png', cv2.flip(annotated_image, 1)
#         )

#         # draw hand landmarks
#         if not results.multi_hand_world_landmarks:
#             continue

#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#         mp_drawing.plot_landmarks(
#             hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth = 5)

#     cv2.imshow('An Image', annotated_image)


prev_index_x = 0
prev_index_y = 0

# Now for the VIDEO
with mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:
    
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




        # flip the image horizontally

        cv2.imshow('Hands in Mediapipe', cv2.flip(frame, 1))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()