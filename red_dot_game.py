from typing import Tuple, List

import cv2 as cv
import time
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands

# global variables: to keep track of the game state
final_score = 5
scores = [0,0]
dot_positions = [(0,0), (0,0)]
dot_interval = 3
dot_update = [time.time(),time.time()]
countdown = 3
countdown_start = None
game_started = False

# global margings: to ensure the index fingers don't have to move off screen
top_margin = left_margin = right_margin = 50
bottom_margin = 100

"""
Process an image for hand detection, required by Mediapipe

Parameters:
img: the image to process
hands: the mediapipe hands solution

Return:
Tuple of the processed hands and image
"""
def process_image(img: cv.UMat, hands: mp.solutions.hands.Hands) -> Tuple[mp.solutions.hands.Hands, cv.UMat]:
    img.flags.writeable = False
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    res = hands.process(img)
    return res, img

"""
Get the correct position of the index fingers

Parameters:
res: the processed hands to determine the index positions
img: the image the hands are displayed on

Return:
Tuple of lists for the x and y coordinates
"""
def get_landmarks(res: mp.solutions.hands.Hands, img: cv.UMat) -> Tuple[List[int], List[int]]:
    xs = []
    ys = []
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            xs.append(int(hand_landmarks.landmark[8].x * img.shape[1]))
            ys.append(int(hand_landmarks.landmark[8].y * img.shape[0]))
    return xs, ys

"""
Update the on screen position of the red dots, added a small interval to not have the dots updated instantly

Parameters:
i: index of the dot
w: width of the image
h: height of the image

Global variables:
dot_positions: the positions of each dot drawn on the screen, i=0 represents the left side and i=1 represents the right side of the screen
dot_udpate: tuple that keeps track of the last time a dot was drawn, i=0 represents the left dot and i=1 represents the right dot 
"""
def update_dot_position(i: int, w: int, h: int) -> None:
    global dot_positions
    global dot_update
    if time.time() - dot_update[i] > dot_interval and dot_positions[i] == (0,0):
        dot_positions[i] = (np.random.randint(left_margin if i==0 else w//2+left_margin, w//2-right_margin if i==0 else w-right_margin),
                            np.random.randint(top_margin,h-bottom_margin))
        dot_update[i] = time.time()

"""
Draw the red dots on each side of the image

Parameters:
img: image to draw the dots on
i: index of the dot
"""
def draw_dots(img: cv.UMat, i: int) -> None:
    x,y = dot_positions[i]
    if x != 0 and y != 0:
        cv.circle(img, (x, y), 10, (0,0,255), -1)

"""
Draw the word "Index" under each detected index finger on the image

Parameters:
img: image to draw the index on
xs: list with x coordinates of detected index fingers
ys: list with y coordinates of detected index fingers
"""
def draw_index(img: cv.UMat, xs: List[int], ys: List[int]) -> None:
    for x,y in zip(xs,ys):
        x = img.shape[1] - x - 25
        y += 25
        cv.putText(img, 'Index', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

"""
Draw the scores at each side on the bottom of the image

Parameters:
img: image to draw the scores on
w: width of the image
h: heigh of the image
"""
def draw_scores(img: cv.UMat, w: int, h: int) -> None:
    cv.putText(img, f"Score: {scores[0]}", (10,h-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
    cv.putText(img, f"Score: {scores[1]}", (w-80,h-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

"""
Draw the gridline at the middle of the image

Parameters:
img: image to draw the gridline on
w: width of the image
h: heigh of the image
"""
def draw_gridline(img: cv.UMat, w: int, h:int) -> None:
    cv.line(img, (w//2, 0), (w//2,h), (255,255,255), 2)


"""
Draw the countdown on each side of the image

Parameters:
img: image to draw the countdown on
countdown: number to be displayed
w: width of the image
h: heigh of the image
"""
def draw_countdown(img: cv.UMat, countdown: int, w: int, h: int) -> None:
    cv.putText(img, str(countdown), (w//4,h//2), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)
    cv.putText(img, str(countdown), (3*w//4,h//2), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)

"""
Logic for catching the red dot with the index fingers

Parameters:
i: index of the dot
x: x position of the index finger
y: y position of the index finger
w: width of the image

Global variables:
scores: keeps track of the score of each index finger, i=0 represents the left and i=1 represents the right index finger
dot_positions: the positions of each dot drawn on the screen, i=0 represents the left side and i=1 represents the right side of the screen

Return:
Updated score
"""
def update_score_if_dot_caught(i: int, x: int, y: int, w: int) -> int:
    global scores
    global dot_positions
    if abs(w - x - dot_positions[i][0]) < 5 and abs(y-dot_positions[i][1]) < 5:
        scores[i] += 1
        dot_positions[i] = (0,0)
    return scores[i]


"""
Logic to start the game with a countdown once two index fingers are detected

Parameters:
w: width of the image
h: height of the image
img: image to draw the countdown on

Global variables:
countdown: keeps track of the current countdown integer, starts at 3 and reduces to 0
countdown_start: keeps track of the time to ensure the countdown is lowered every second and not every frame
game_started: keeps track if the game is started (after countdown reaches 0) or not
"""
def start_game(w: int, h: int, img: cv.UMat) -> None:
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

"""
Logic to play the game once the game has started, releases the webcam when the game is over after a player reaches 5 points and prints the winning player to the console

Parameters:
i: index of the dot
xs: list with x coordinates of detected index fingers
ys: list with y coordinates of detected index fingers
w: width of the image
h: height of the image
img: image to drawn the game elements on
cap: videocapture object to close if game is over

Global variables:
scores: keeps track of the score of each index finger, i=0 represents the left and i=1 represents the right index finger
"""
def play_game(i: int, xs: List[int], ys: List[int], w: int, h: int, img: cv.UMat, cap: cv.VideoCapture) -> None:
    global scores
    update_dot_position(i,w,h)
    draw_dots(img,i)
    scores[i] = update_score_if_dot_caught(i, xs[i], ys[i], w)
    if scores[i] >= final_score:
        print(f"Player {i+1} won the game!")
        cap.release()

# main loop: to play the red dot game
def main() -> None:
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
            draw_gridline(img, w, h)
            if len(xs) == 2:
                for i in range(2):
                    if not game_started:
                        start_game(w, h, img)
                    else:
                        play_game(i, xs, ys, w, h, img, cap)
                draw_index(img, xs, ys)
                draw_scores(img, w, h)
            else:
                cap.release()
                raise Exception("A finger moved offscreen. Please restart a new game")

            cv.imshow("Catch the Dot",img)
            if cv.waitKey(5) & 0xFF == ord('q'):
                break  

    cap.release()

if __name__ == "__main__":
    main()