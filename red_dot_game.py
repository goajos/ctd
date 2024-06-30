from typing import Tuple, List, NoReturn

import cv2 as cv
import time
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands

final_score = 5
scores = [0,0]
dot_positions = [(0,0), (0,0)]
dot_interval = 3
dot_update = [time.time(),time.time()]
countdown = 3
countdown_start = None
game_started = False

top_margin = left_margin = right_margin = 50
bottom_margin = 100

def process_image(img: cv.UMat, hands: mp.solutions.hands.Hands) -> Tuple[mp.solutions.hands.Hands, cv.UMat]:
    img.flags.writeable = False
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    res = hands.process(img)
    return res, img

def get_landmarks(res: mp.solutions.hands.Hands, img: cv.UMat) -> Tuple[List[int], List[int]]:
    xs = []
    ys = []
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            xs.append(int(hand_landmarks.landmark[8].x * img.shape[1]))
            ys.append(int(hand_landmarks.landmark[8].y * img.shape[0]))
    return xs, ys

def update_dot_position(i: int, w: int, h: int) -> NoReturn:
    global dot_positions
    global dot_update
    if time.time() - dot_update[i] > dot_interval and dot_positions[i] == (0,0):
        dot_positions[i] = (np.random.randint(left_margin if i==0 else w//2+left_margin, w//2-right_margin if i==0 else w-right_margin),
                            np.random.randint(top_margin,h-bottom_margin))
        dot_update[i] = time.time()

def draw_dots(img: cv.UMat, i: int, w: int) -> NoReturn:
    x,y = dot_positions[i]
    if x != 0 and y != 0:
        cv.circle(img, (x, y), 10, (0,0,255), -1)

def draw_index(img: cv.UMat, xs: List[int], ys: List[int]) -> NoReturn:
    for x,y in zip(xs,ys):
        x = img.shape[1] - x - 25
        y += 25
        cv.putText(img, 'Index', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

def draw_scores(img: cv.UMat, w: int, h: int) -> NoReturn:
    cv.putText(img, f"Score: {scores[0]}", (10,h-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
    cv.putText(img, f"Score: {scores[1]}", (w-80,h-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

def draw_grid(img: cv.UMat, w: int, h:int) -> NoReturn:
    cv.line(img, (w//2, 0), (w//2,h), (255,255,255), 2)

def draw_countdown(img: cv.UMat, countdown: int, w: int, h: int) -> NoReturn:
    cv.putText(img, str(countdown), (w//4,h//2), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)
    cv.putText(img, str(countdown), (3*w//4,h//2), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)

def catch_the_dot(i: int, x: int, y: int, w: int) -> int:
    global scores
    global dot_positions
    if abs(w - x - dot_positions[i][0]) < 5 and abs(y-dot_positions[i][1]) < 5:
        scores[i] += 1
        dot_positions[i] = (0,0)
    return scores[i]

def start_game(i: int, w: int, h: int, img: cv.UMat) -> NoReturn:
    global countdown
    global countdown_start
    global game_started
    if countdown_start is None:
        countdown_start = time.time()
    if time.time() - countdown_start >= 1:
        countdown -= 1
        countdown_start = time.time()
    draw_countdown(img, countdown, w, h)
    if countdown == 0:
        game_started = True

def play_game(i: int, xs: List[int], ys: List[int], w: int, h: int, img: cv.UMat, cap: cv.VideoCapture) -> NoReturn:
    global scores
    update_dot_position(i,w,h)
    draw_dots(img,i,w)
    scores[i] = catch_the_dot(i, xs[i], ys[i], w)
    if scores[i] >= final_score:
        print(f"Player {i+1} won the game!")
        cap.release()

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
            draw_grid(img, w, h)
            if len(xs) == 2:
                for i in range(2):
                    if not game_started:
                        start_game(i, w, h, img)
                    else:
                        play_game(i, xs, ys, w, h, img, cap)
                draw_index(img, xs, ys)
                draw_scores(img, w, h)

            cv.imshow("Catch the Dot",img)
            if cv.waitKey(5) & 0xFF == ord('q'):
                break  

    cap.release()

if __name__ == "__main__":
    main()