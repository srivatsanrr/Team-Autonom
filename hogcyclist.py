import cv2
import numpy as np 
import imutils 

winStride= (4,2)
meanShift= False
padding= (0,0)
scale= 1.05

cap = cv2.VideoCapture('crash2.mp4')

while(cap.isOpened()):
	ret,frame = cap.read()
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	frame = imutils.resize(frame, 750, 1200)
	(rects, weights) = hog.detectMultiScale(frame, winStride=winStride,
		padding=padding, scale=scale, useMeanshiftGrouping=meanShift)
	for(x,y,w,h) in rects: 
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
	cv2.imshow('fr', frame)
	# gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=1)
	# gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=1)
	# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
	# cv2.waitKey(0)
cap.release() 
cv2.destroyAllWindows() 