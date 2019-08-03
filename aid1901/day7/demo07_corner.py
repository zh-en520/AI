# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import cv2 as cv

original = cv.imread('../ml_data/box.png')
cv.imshow('Original', original)
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
corners = cv.cornerHarris(gray, 7, 5, 0.04)
mixture = original.copy()
# 把角点位置的像素颜色改为红色
mixture[corners > corners.max() * 0.01] = [0, 0, 255]
cv.imshow('Corner', mixture)
cv.waitKey()


