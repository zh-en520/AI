"""
demo13_cascade.py  级联定位
"""
import cv2 as cv

# 定义哈尔级联人脸定位器
fd = cv.CascadeClassifier('../ml_data/haar/face.xml')
ed = cv.CascadeClassifier('../ml_data/haar/eye.xml')
nd = cv.CascadeClassifier('../ml_data/haar/nose.xml')

# 视频头
video_capture = cv.VideoCapture(0)
while True:
	frame = video_capture.read()[1]
	# 根据哈尔定位器 找到人脸的位置 并绘制
	faces = fd.detectMultiScale(frame, 1.5, 2)
	for l, t, w, h in faces:
		a, b = int(w/2), int(h/2)
		cv.ellipse(frame, 
			(l+a, t+b), 
			(a, b), 
			0, 0, 360, 
			(255,0,255), 2)

		# 找鼻子找眼
		face = frame[t:t+h, l:l+w]
		eyes = ed.detectMultiScale(face, 1.5, 2)
		for l, t, w, h in eyes:
			a, b = int(w/2), int(h/2)
			cv.ellipse(face, 
				(l+a, t+b), 
				(a, b), 
				0, 0, 360, 
				(0,255,255), 2)

		noses = nd.detectMultiScale(face, 1.2, 1)
		for l, t, w, h in noses:
			a, b = int(w/2), int(h/2)
			cv.ellipse(face, 
				(l+a, t+b), 
				(a, b), 
				0, 0, 360, 
				(255,255,0), 2)

	cv.imshow('Frame', frame)
	if cv.waitKey(33) == 27:
		break

video_capture.release()
cv.destroyAllWindows()