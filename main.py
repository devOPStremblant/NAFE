from __future__ import division
import cv2
from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
from PIL import Image
import time
import math


sd = 2
coeffecients = []
runningTotal = 0
x0 = y0 = 0

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
            print im.getPixelValue(w, h)

def processImage(im) :
    w, h = im.size
    x0 = w / 2
    y0 = h / 2
    print x0, y0
    for x in range (0, w):
        for y in range (0, h):
            # print "starting loop for %d, %d" % (x, y)
            
            q = findQuadrant(x, y,x0, y0)

            xl, yl = calculatePixelOffset(x, y, q, x0, y0)

            r = findRadius(xl, yl)
            varphi = calculateDegrees(xl,yl, q)

            carX, carY = polToCar(r, varphi)
            carX +=x0
            carY = y0-carY
            if q == 2:
                print "Pixel (%d, %d) in %d quadrant is (%d, %d) faraway with polar coordinates (%f, %f) and car coordinates (%d, %d)" % (x, y,q, xl, yl, r, varphi, carX, carY)

            lowerRho = r - 2*sd
            upperRho = r + 2*sd

            lowerPhi = (varphi - ((2 * sd) / r))
            upperPhi = (varphi + ((2 * sd) / r))

            # print "LowerRho Value: %f" %  lowerRho
            # print "Upper Value: %f" % upperRho

            sumA = 0
            rhoIndex = 0.001
        
            # *** WHILE LOOP HERE

            # print "end of loop %d, %d" % (x,y)
            # for rho in range(lowerRho, upperRho):
                # for phi  in range(lowerPhi, upperPhi):
                    # sumA += calculateKernel(rho, phi, r, varphi)

            # print "Radius of %d, %d is %f" % (x, y, r)
            # print "Varphi of %d, %d is %f" % (xl, yl, varphi)
            # get Pixel
            # print im.getpixel((r,varphi))
    print "Origin: %d, %d" % (x0, y0)

def calculatePixelOffset(x, y, q, x0, y0):
    if q == 0:
        return [x,y]
    return [x - x0, y0 - y]
    
def dummyWhileLoop():
    startTime = time.time()    
    x = y = 0
    count = 0
    for x in range(0,100):
        for y in range(0, 100):
            
            rhoIndex = 0.001
            while rhoIndex < 100:
                # print rhoIndex
                rhoIndex += 0.001
            count+=1
            # print "End of Loop #%d" % count
    print time.time() - startTime
    
    
def polToCar(r, varphi):
    # print "varphi in degrees: %f" % varphi
    xCordinate = r * math.cos(np.deg2rad(varphi))
    yCoordinate = r * math.sin(np.deg2rad(varphi))
    return [math.floor(xCordinate), math.floor(yCoordinate)]

def getPixelValue(r, varphi):
    xCordinate = r * np.rad2deg(np.cos(varphi))
    yCoordinate = r * np.rad2deg(np.sin(varphi))
    # pixelValue = 

def findQuadrant(x, y, x0, y0):
    # print x, y, x0, y0
    if x > x0 and y > y0:
        # print "IV Quadrant"
        return 4
    elif x > x0 and y < y0:
        # print "I Quadrant"
        return 1
    elif x < x0 and y > y0:
        # print "III Quadrant"
        return 3
    elif x < x0 and y < y0:        
        # print "II Quadrant"
        return 2
    return 0
    

def findRadius(xl, yl):
    r = np.sqrt((xl*xl) + (yl*yl))
    return r

def calculateDegrees(xl, yl, q):
    # print "Quadrant %d" % q
    # print "(xl, yl) : %d, %d" % (xl, yl)
    if (xl == 0 or q == 0):
        return np.rad2deg(np.arctan(0))

    degrees = 0
    if(q == 1):
        degrees = np.rad2deg(np.arctan(yl/xl))
    elif (q == 2):
        degrees = np.rad2deg(np.arctan(abs(xl)/abs(yl))) + 90
    else:
        degrees = np.rad2deg(np.arctan(yl/xl)) + ((q-1)*90)
    # print q*90
    # print yl/xl
    # print np.arctan(yl/xl)
    # print degrees
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
print "Just a number"
print np.cos(23)
print "Radians to Degrees"
print np.cos(np.rad2deg(23))
print "Degrees to Radians"
print np.cos(np.deg2rad(23))
# displayImageDimensions(o_img)
# iterateThroughImagePixels(o_img)
processImage(o_img)
# calculateDegrees(32, -19, 3)
# calculateDegrees(22, 24,1)
# calculateDegrees(0,0, 4)
# calculateDegrees(-43,-21,2)

# dummyWhileLoop()

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