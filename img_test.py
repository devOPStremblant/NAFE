import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test.jpg',0)

height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))

plt.show()

def decimal_range(start, stop, increment):
    while start < stop: # and not math.isclose(start, stop): Py>3.5
        yield start
        start += increment
Rangelow=0.11
Rangehigh=1
Delta=0.001
for i in decimal_range(Rangelow, Rangehigh, Delta):
	print i,Rangelow,Rangehigh,Delta