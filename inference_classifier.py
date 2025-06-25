#using a trained model to make predictions on new, unseen data.
import cv2
import pickle
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labes_dict = {0: 'A', 1: 'B', 2: 'V' , 3: 'L'}  


while True:
    data_aux = []  # for each frame we will have an array of landmarks
    x_ = []
    y_ = []
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        padding = 40
        x1 = int(min(x_) * W) - padding
        y1 = int(min(y_) * H) - padding
        x2 = int(max(x_) * W) + padding
        y2 = int(max(y_) * H) + padding


        prediction = model.predict([np.array(data_aux)]) #list of one element
        predicted_character = labes_dict[int(prediction[0])]
        print('Predicted character:', predicted_character)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
