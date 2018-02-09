from __future__ import division
import cv2
from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
from PIL import Image
import time
import math


sd = 1
coeffecients = []
runningTotal = 0
x0 = y0 = 0
# f = open('adjustedRadius.txt')

def initialize():
    imgHelper = ImageHelper()
    imgHelper.readImage()


def read_image():
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
            print im.get_pixel_value(w, h)


def processImage(im, newImg) :
    w, h = im.size
    x0 = int(w / 2)
    y0 = int(h / 2)
    print x0, y0
    pixels_with_values = []
    dummy = 0
    for x in range (0, w):
        for y in range (0, h):
            # print "starting loop for %d, %d" % (x, y)
            
            q = findQuadrant(x, y, x0, y0)

            xl, yl = get_offset_from_center(x, y, q, x0, y0)

            r = findRadius(xl, yl)
            varphi = calculateDegrees(xl, yl, q)

            # carX, carY = polToCar(r, varphi)
            # carX += x0
            # carY = y0-carY

            lowerRho = r - 2*sd
            upperRho = r + 2*sd

            lowerPhi = varphi - ((2 * sd) / r)
            upperPhi = varphi + ((2 * sd) / r)

            # print "LowerRho Value: %f" %  lowerRho
            # print "Upper Value: %f" % upperRho

            sumA = 0
            sumB = 0
            rhoIndex = lowerRho
            phiIndex = lowerPhi

            while rhoIndex < upperRho:
                while phiIndex < upperPhi:
                    # print "Current Rho is %f and Current Phi is %f " % (rhoIndex, phiIndex)
                    c_x, c_y = polToCar(r + rhoIndex, phiIndex + varphi)
                    c_x += x0
                    c_y = y0 - c_y
                    pv = get_pixel_value(c_x, c_y, im)
                    try:
                        newImg.putpixel((c_x, c_y), (0, 0, 255, 255))
                    except IndexError:
                        print "Out of bounds"
                    # print pv
                    kernel_value = calculateKernel(rhoIndex, phiIndex, r, varphi)
                    sumA += (pv[0] * kernel_value)
                    sumB += kernel_value
                    phiIndex += 0.01
                rhoIndex += 0.01

            original_pixel_value = get_pixel_value(x, y, im)[0]
            if sumB == 0 or original_pixel_value == 0:
                new_pixel_value = 0
            else:
                new_pixel_value = get_pixel_value(x, y, im)[0] - int(sumA / sumB)

            # if new_pixel_value > 0:
            #     print x, y
            # newImg.putpixel((x, y), (new_pixel_value, new_pixel_value, new_pixel_value, 255))
            # print "Radius of %d, %d is %f" % (x, y, r)
            # print "Varphi of %d, %d is %f" % (xl, yl, varphi)
            # get Pixel
            # print im.getpixel((r,varphi))
    newImg.save('output.png')
    newImg.show()


def reverse_offset(x, y, x0, y0):
    return x+x0, y0-y


def get_offset_from_center(x, y, q, x0, y0):
    if q == 0:
        return [x, y]
    return [x - x0, y0 - y]


def polToCar(radius, varphi):
    # print "varphi in degrees: %f" % varphi
    x_coordinate = radius * math.cos(np.deg2rad(varphi))
    y_coordinate = radius * math.sin(np.deg2rad(varphi))

    if x_coordinate < 0:
        x = math.ceil(x_coordinate)
    else:
        x = math.floor(x_coordinate)

    if y_coordinate < 0:
        y = math.ceil(y_coordinate)
    else:
        y = math.floor(y_coordinate)

    return [int(x),int(y)]


def get_pixel_value(x, y, img):
    # print x, y
    try:
        return img.getpixel((x, y))
    except IndexError:
        return (0,0,0, 255)
    print 


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

    if(xl == 0 and yl == 0):
        return 0

    if(xl==0):
        if(yl < 0):
            return 270
        else:
            return 90
    if(yl==0):
        if(xl > 0):
            return 0
        else:
            return 180
    
    if (xl == 0 or q == 0):
        return np.rad2deg(np.arctan(0))

    degrees = 0
    if(q == 1 or q == 4):
        degrees = np.rad2deg(np.arctan(yl/xl))
    elif (q == 2):
        degrees = np.rad2deg(np.arctan(abs(xl)/abs(yl))) + 90
    else:
        degrees = np.rad2deg(np.arctan(abs(yl)/abs(xl))) + ((q-1)*90)
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


def copyImage(o_img):
    return Image.new(o_img.mode, o_img.size)


# initialize()
o_img = read_image()
# displayImageDimensions(o_img)
# iterateThroughImagePixels(o_img)
c_img = copyImage(o_img)
processImage(o_img, c_img)
# calculateDegrees(32, -19, 3)
# calculateDegrees(22, 24,1)
# calculateDegrees(0,0, 4)
# calculateDegrees(-43,-21,2)

# normalize(coeffecients)
# plt.show()