# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo12_vc.py 视频捕获
"""
import cv2 as cv
vc = cv.VideoCapture(0)
while True:
    frame = vc.read()[1]
    cv.imshow('VideoCapture', frame)
    # 阻塞时摁了ESC键  则返回27
    if cv.waitKey(33) == 27:   
        break
vc.release()
cv.destroyAllWindows()