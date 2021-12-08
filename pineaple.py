#import libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#read image
pine = cv.imread('pineaple.jpg')
pine_gray = cv.cvtColor(pine, cv.COLOR_BGR2GRAY)
blur_pine = cv.GaussianBlur(pine_gray, (9,9), cv.BORDER_DEFAULT)
#cv.imshow('PINE', blur_pine)

#adaptive threshold
#pine_adapt = cv.adaptiveThreshold(blur_pine, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 255, 13)
#pine_adapt = cv.adaptiveThreshold(blur_pine, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 255, 13)
ret, pine_adapt = cv.threshold(blur_pine, 252, 255, cv.THRESH_BINARY_INV)
#ret, pine_adapt = cv.threshold(blur_pine, 252, 255,0)
#cv.imshow('PINE-ADAPT', pine_adapt)
#contour detection
import imutils
#contour = cv.findContours(pine_adapt.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# = cv.findContours(pine_adapt.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#contour, hierarchy = cv.findContours(pine_adapt, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

contour = cv.findContours(pine_adapt.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(contour)
cv.drawContours(pine, cont, 0, (150, 1, 159), 5)

print('Number of contours =' + str(len(contour)))
print(contour[0])

for c in cont:
    area = cv.contourArea(c)
    Moments = cv.moments(c)
    cX = int(Moments["m10"] / Moments["m00"])
    cY = int(Moments["m10"] / Moments["m00"])
    cv.circle(pine, (cX, cY), 5, (255, 255, 255, -1))
    cv.imshow('Pine', pine)
print("area of pineaple contour is:", area)
print("centroid of image is:", cX, cY)

cv.waitKey(0)