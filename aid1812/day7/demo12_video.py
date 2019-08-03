"""
demo12_video.py  捕获视频
"""
import cv2 as cv
# 视频头
video_capture = cv.VideoCapture(0)
while True:
	frame = video_capture.read()[1]
	cv.imshow('Frame', frame)
	if cv.waitKey(33) == 27:
		break

video_capture.release()
cv.destroyAllWindows()