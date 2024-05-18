import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)


with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(results)


        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imwrite(
            os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image
        )

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ('z'):
            break

cap.release()
cv2.destroyAllWindows()

mp_hands.HAND_CONNECTIONS

#So now I can track the image with a flipped image. Now I just need to add the angles between the fingers and maybe some
#if statements to generate a passcode things haha. Hmm. I can use the mediapipe landmarks to do that. Let's do it.
