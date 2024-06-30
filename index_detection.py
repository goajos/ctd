import cv2 as cv
import mediapipe as mp
mp_hands = mp.solutions.hands

cap  = cv.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        succ,img = cap.read()
        if not succ:
            print("Empty frame")
            continue

        img.flags.writeable = False
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        res = hands.process(img)

        xs = []
        ys = []
        # index finger tip detection check
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                xs.append(int(hand_landmarks.landmark[8].x * img.shape[1]))
                ys.append(int(hand_landmarks.landmark[8].y * img.shape[0]))

        img = cv.flip(img,1)
        for x,y in zip(xs,ys):
            x = img.shape[1] - x - 25
            y += 25
            cv.putText(img, 'Index', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
        cv.imshow("Hands",cv.cvtColor(img,cv.COLOR_RGB2BGR))
        if cv.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()