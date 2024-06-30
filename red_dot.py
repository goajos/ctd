import cv2 as cv
import time
import numpy as np

cap = cv.VideoCapture(0)
scores = [0,0]
last_dot = time.time()
dot_interval = 3
dot_positions = [(0,0), (0,0)]

top_margin = left_margin = right_margin = 25
bottom_margin = 50

while cap.isOpened():
    succ, img = cap.read()
    if not succ:
        print("Empty frame")
        continue

    h,w,_ = img.shape
    left = img[:, :w//2]
    right = img[:, w//2:]

    if time.time() - last_dot > dot_interval:
        dot_positions = [(np.random.randint(left_margin,w//2-right_margin),np.random.randint(top_margin,h-bottom_margin)) for _ in range(2)]
        last_dot = time.time()

    for i,(x,y) in enumerate(dot_positions):
        if x != (0,0) and y != (0,0):
            cv.circle(img, (x if i==0 else w//2+x, y), 10, (0,0,255), -1)

    img = cv.flip(img,1)
    cv.putText(img, f"Score: {scores[0]}", (10,h-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
    cv.putText(img, f"Score: {scores[1]}", (w-80,h-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
    cv.line(img, (w//2, 0), (w//2,h), (255,255,255), 2)

    cv.imshow("Catch the Dot",img)
    if cv.waitKey(5) & 0xFF == ord('q'):
        break  

cap.release()