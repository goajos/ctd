from typing import Tuple, List, NoReturn

import cv2 as cv
import time
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands

scores = [0,0]
last_dot = time.time()
dot_interval = 3
dot_positions = [(0,0), (0,0)]

top_margin = left_margin = right_margin = 25
bottom_margin = 50

def process_image(img: cv.UMat, hands: mp.solutions.hands.Hands) -> Tuple[mp.solutions.hands.Hands, cv.UMat]:
    img.flags.writeable = False
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    res = hands.process(img)
    return res, img

def get_landmarks(res: mp.solutions.hands.Hands, img: cv.UMat) -> Tuple[List[int], List[int]]:
    xs = []
    ys = []
    # index finger tip detection check
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            xs.append(int(hand_landmarks.landmark[8].x * img.shape[1]))
            ys.append(int(hand_landmarks.landmark[8].y * img.shape[0]))
    return xs, ys

def update_dot_position(w: int, h: int) -> Tuple[Tuple[int], Tuple[int]]:
    global last_dot
    global dot_positions
    if time.time() - last_dot > dot_interval:
        dot_positions = [(np.random.randint(left_margin,w//2-right_margin),np.random.randint(top_margin,h-bottom_margin)) for _ in range(2)]
        last_dot = time.time()
    return dot_positions

def draw_dots(img: cv.UMat, dot_positions: Tuple[Tuple[int], Tuple[int]], w: int) -> NoReturn:
    for i,(x,y) in enumerate(dot_positions):
        if x != (0,0) and y != (0,0):
            cv.circle(img, (x if i==0 else w//2+x, y), 10, (0,0,255), -1)

def draw_index(img: cv.UMat, xs: List[int], ys: List[int]) -> NoReturn:
    for x,y in zip(xs,ys):
        x = img.shape[1] - x - 25
        y += 25
        cv.putText(img, 'Index', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

def draw_scores(img: cv.UMat, scores: List[int], w: int, h: int) -> NoReturn:
    cv.putText(img, f"Score: {scores[0]}", (10,h-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
    cv.putText(img, f"Score: {scores[1]}", (w-80,h-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

def draw_grid(img: cv.UMat, w: int, h:int) -> NoReturn:
    cv.line(img, (w//2, 0), (w//2,h), (255,255,255), 2)

def main() -> NoReturn:
    cap = cv.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            succ, img = cap.read()
            if not succ:
                print("Empty frame")
                continue

            res, img = process_image(img, hands)
            xs, ys = get_landmarks(res,img)
            img = cv.flip(cv.cvtColor(img,cv.COLOR_RGB2BGR),1)
            h,w,_ = img.shape
            dot_positions = update_dot_position(w,h)
            draw_dots(img, dot_positions, w)
            draw_index(img, xs, ys)
            draw_scores(img, scores, w, h)
            draw_grid(img, w, h)

            cv.imshow("Catch the Dot",img)
            if cv.waitKey(5) & 0xFF == ord('q'):
                break  

    cap.release()

if __name__ == "__main__":
    main()