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
        # hand detection check
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[0].x * img.shape[1])
                y = int(hand_landmarks.landmark[0].y * img.shape[0])
                xs.append(x)
                ys.append(y)
                #cv.putText(img, 'Hand detected', (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)

        img = cv.flip(img,1)
        for x,y in zip(xs,ys):
            # position text properly with flip at base of each hand
            x = img.shape[1] - x - 100
            cv.putText(img, 'Hand detected', (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
        cv.imshow("Hands",cv.cvtColor(img,cv.COLOR_RGB2BGR))
        if cv.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()