import cv2
from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
from PIL import Image


sd = 1
coeffecients = []
runningTotal = 0


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

def findQuadrant(im):
    w, h = im.size
    a = w / 2
    b = h / 2
    print a, b
    for x in range (0, w):
        for y in range (0, h):
            if x < a and y < b:
                print "%d and %d First Quadrant" % (x, y)
                # xl = x - a
                # yl = b - y
            elif x > a and y > b:
                print "%d and %d Second Quadrant" % (x, y)
                # xl = x - a
                # yl = b - y
            elif x < a and y > b:
                print "%d and %d Third Quadrant" % (x, y)
                # xl = x - a
                # yl = b - y
            else: 
                print "%d and %d Fourth Quadrant" % (x, y)
                # xl = x - a
                # yl = b - y




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
findQuadrant(o_img)

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