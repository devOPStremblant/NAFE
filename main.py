from __future__ import division
import cv2
from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
from PIL import Image



sd = 1
coeffecients = []
runningTotal = 0
a = b = 0


def initialize():
    imgHelper = ImageHelper()
    imgHelper.readImage()



def readImage():
    return Image.open('sample.png')

def displayImageDimensions(im):
    w, h = im.size
    print w
    print h
    # im.show()

def iterateThroughImagePixels(im):
    w, h = im.size
    for x in range(0, w):
        for y in range(0, h):
            print "Pixel %d %d" % (x, y)

def findPolarCoordinates(im):
    w, h = im.size
    a = w / 2
    b = h / 2
    print a, b
    for x in range (0, w):
        for y in range (0, h):
            xl = x - a
            yl = b - y

            r = findRadius(xl, yl)
            varphi = calculateDegrees(xl,yl, findQuadrant(x, y))
            print "Radius of %d, %d is %f" % (x, y, r)
            print "Varphi of %d, %d is %f" % (xl, yl, varphi)
           

def findQuadrant(x, y):
    if x > a and y < b:
        return 1
    elif x > a and y > b:
        return 2
    elif x < a and y > b:
        return 3
    else: 
        return 4
    return 
    

def findRadius(xl, yl):
    r = np.sqrt((xl*xl) + (yl*yl))
    return r

def calculateDegrees(xl, yl, q):
    print "Quadrant %d" % q
    print "(xl, yl) : %d, %d" % (xl, yl)
    if (xl == 0):
        return 0

    degrees = 0
    if(q == 1):
        degrees = np.arctan(abs(yl)/(xl))
    else:
        degrees = (q*90) - np.rad2deg(np.arctan(abs(yl)/abs(xl)))
    print q*90
    print yl/xl
    print np.rad2deg(np.arctan(yl/xl))
    print degrees
    return degrees

def calculateKernel(rho, phi, r, varphi):
    # print "**************"
    # print "Actual Pixels"
    # print "rho %d" % (rho)
    # print "phi %d" % (phi)
    # print "Coefficients"
    # print "r %d" % (r)
    # print "varphi %d" % (varphi)
    # print "rho-phi whole sq %d" % (np.square(r - rho))
    # print "the other part whole sq %d" % (np.square(r * (varphi - phi)))
    numerator = np.square(r - rho) + np.square(r * (varphi - phi))
    # print "Numerator %d" % numerator
    expParam = numerator / 2 * np.square(sd)
    # print "(%d, %d)" % (r, varphi)
    return np.exp((-1) * expParam)
    # print "**************"

def normalize(coeffecients):
    coeffecients.sort()
    denominator = coeffecients[len(coeffecients)-1] - coeffecients[0] #maxValue - minValue
    for c in coeffecients:
        c = (c - coeffecients[0]) / runningTotal
        print c
    print coeffecients
    return

# initialize()
o_img = readImage()
# displayImageDimensions(o_img)
# iterateThroughImagePixels(o_img)
# findPolarCoordinates(o_img)
calculateDegrees(84, 6, 4)
calculateDegrees(-4,-6,3)
calculateDegrees(4,-89, 4)
calculateDegrees(-43,-21,2)

l = [-1, 0, 1]
for x in l:
    tempX = x
    for y in l:
        tempY = y
        # print "Cooefficient (%d, %d)" % (x, y)
        # print "Coefficient (%d, %d)" % (tempX, tempY)
        coefficient = calculateKernel(0, 5, tempX, tempY)
        coeffecients.append(coefficient)
        runningTotal += coefficient



# normalize(coeffecients)
# plt.show()