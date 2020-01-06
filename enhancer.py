import cv2

cap = cv2.VideoCapture(0)

kernel = [[-1, 2, -1],[-1, 2, -1],[-1, 2, -1]]

while 1:
    
    _, frame = cap.read()
    frame = cv2.filter2D(frame, -1, kernel)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
