from __future__ import division
import cv2
from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
from PIL import Image
import time
import math
import sys


# x_coordinate = int(sys.argv[1])
# y_coordinate = int(sys.argv[2])

x_coordinate = 20
y_coordinate = 20

sd = 2
coeffecients = []
runningTotal = 0
x0 = y0 = 0
f = open('radius.txt','w')

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
    pixel_count = 0

    q = findQuadrant(x_coordinate, y_coordinate, x0, y0)
    print "Quadrant: %s" % q

    x_offset, y_offset = calculatePixelOffset(x_coordinate, y_coordinate, q, x0, y0)
    print "Offset from Center of Sun: %s, %s" % (x_offset, y_offset)

    
    r = findRadius(x_offset, y_offset)
    varphi = calculateDegrees(x_offset, y_offset, q)
    print "Polar Coordinates: (radius, theta) is (%s, %s)" % (r, varphi)

    carX, carY = polToCar(r, varphi)
    carX += x0
    carY = y0-carY
    print "Cartesian coordinates (x, y): (%s, %s)" % (carX, carY)

    # f.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (x, y, r, varphi, carX, carY))

    lowerRho = r - 2*sd
    upperRho = r + 2*sd
    print "Rho Range: %s, %s" % (lowerRho, upperRho)

    lowerPhi = (varphi - ((2 * sd) / r))
    upperPhi = (varphi + ((2 * sd) / r))
    print "Phi Range: %s, %s" % (lowerPhi, upperPhi)

    # print "LowerRho Value: %f" %  lowerRho
    # print "Upper Value: %f" % upperRho

    sumA = 0
    sumB = 0
    rhoIndex = lowerRho
    phiIndex = lowerPhi

    while rhoIndex < upperRho:
        while phiIndex < upperPhi:
            # print "Current Rho is %f and Current Phi is %f " % (rhoIndex, phiIndex)
            print "Current Rho and Phi Indices: %s, %s" % (rhoIndex, phiIndex)
            print "Polar Coordinates of the f function: %s, %s" % (r + rhoIndex, phiIndex + varphi)
            c_x, c_y = polToCar(r + rhoIndex, phiIndex + varphi)
            c_x += x0
            c_y = y0 - c_y
            print "Cartesian Coordinate for the above pixels (%s, %s)" % (c_x, c_y)
            pv = get_pixel_value(c_x, c_y, im)
            print "Pixel value at the above coordinate is %s" % (pv,)
            # print pv
            kernel_value = calculateKernel(rhoIndex, phiIndex, r, varphi)
            print "Kernel value (C function) for the current combination is: %s" % kernel_value 
            sumA += (pv[0] * kernel_value)
            sumB += kernel_value
            phiIndex += 0.1
        rhoIndex += 0.1

    if sumB == 0:
        new_pixel_value = 0
    else:
        new_pixel_value = get_pixel_value(x_coordinate, y_coordinate, im)[0] - int(sumA / sumB)

    print "Gaussian A: %s" % sumA
    print "Gaussian B: %s" % sumB


    print "Filtered Pixel value (after subtracting): %s" % new_pixel_value
    # if new_pixel_value > 0:
        # print x, y
    # if r < 40:    
    newImg.putpixel((x_coordinate, y_coordinate), (new_pixel_value, new_pixel_value, new_pixel_value, 255))
    print "Pixel Value read again for verification: %s" % (newImg.getpixel((x_coordinate, y_coordinate)),)

    # if int(r) % 2 == 0:
    #     newImg.putpixel((x, y), (new_pixel_value, new_pixel_value, new_pixel_value, 255))
    # else:
    #     newImg.putpixel((x, y), (0, 0, 255, 255))
    # print "Radius of %d, %d is %f" % (x, y, r)
    # print "Varphi of %d, %d is %f" % (x_offset, y_offset, varphi)
    # get Pixel
    # print im.getpixel((r,varphi))
    pixel_count += 1

    print pixel_count
    newImg.save('output%s.png' % time.time())
    # newImg.show()


def reverse_offset(x, y, x0, y0):
    return x+x0, y0-y


def calculatePixelOffset(x, y, q, x0, y0):
    if q == 0:
        return [x, y]
    return [x - x0, y0 - y]


def polToCar(radius, varphi):
    # print "varphi in degrees: %f" % varphi
    x_coordinate = radius * math.cos(np.deg2rad(varphi))
    y_coordinate = radius * math.sin(np.deg2rad(varphi))

    # if y_coordinate < 0:
    #     x = math.ceil(x_coordinate)
    # else:
    #     x = math.floor(x_coordinate)

    # if x_coordinate < 0:
    #     y = math.ceil(y_coordinate)
    # else:
    #     y = math.floor(y_coordinate)

    return [x_coordinate,y_coordinate]


def get_pixel_value(x, y, img):
    # print x, y
    try:
        return img.getpixel((x, y))
    except IndexError:
        return (0,0,0, 255)


def findQuadrant(x, y, x0, y0):
    # print x, y, x0, y0
    
    if x == x0 or y == y0:
        return 'NoQuadrant'
    
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
    

def findRadius(x_offset, y_offset):
    r = np.sqrt((x_offset*x_offset) + (y_offset*y_offset))
    return r


def calculateDegrees(x_offset, y_offset, q):
    # print "Quadrant %d" % q
    # print "(x_offset, y_offset) : %d, %d" % (x_offset, y_offset)

    if(x_offset == 0 and y_offset == 0):
        return 0

    if(x_offset==0):
        if(y_offset < 0):
            return 270
        else:
            return 90
    if(y_offset==0):
        if(x_offset > 0):
            return 0
        else:
            return 180
    
    if (x_offset == 0 or q == 0):
        return np.rad2deg(np.arctan(0))

    degrees = 0
    if(q == 1 or q == 4):
        degrees = np.arctan(y_offset/x_offset)
    elif (q == 2):
        degrees = np.deg2rad(np.rad2deg(np.arctan(abs(y_offset)/abs(x_offset)))+90)
    else:
        degrees = np.arctan(abs(y_offset)/abs(x_offset)) + ((q-1)*90)
    # print q*90
    # print y_offset/x_offset
    # print np.arctan(y_offset/x_offset)
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

def write_img_test(im):
    r = 100
    deg = 360
    out_of_bounds = 0
    while r > 0:
        while deg > 0:
            try:
                x, y = polToCar(r, deg)
                print x, y
                im.putpixel((x,y), (0,0,255,255))
            except IndexError:
                out_of_bounds += 1
            deg = deg - 1
        r = r - 1
    print "Index Out Of Bounds: %d " % out_of_bounds
    im.save('test.png')
    im.show()

def write_img_with_polar_car(img, o_img):
    w, h = img.size
    x0 = int(w / 2)
    y0 = int(h / 2)
    for x in range (0, w):
        for y in range (0, h):
            print "current pixel %d, %d" % (x, y)
            q = findQuadrant(x, y, x0, y0)

            x_offset, y_offset = calculatePixelOffset(x, y, q, x0, y0)

            r = findRadius(x_offset, y_offset)
            varphi = calculateDegrees(x_offset, y_offset, q)

            # print "polar coordinate %s, %s" % (r, varphi)

            carX, carY = polToCar(r, varphi)
            # print "Pol2Car before adjusting %s, %s" % (round(carX), round(carY))
            carX += x0
            carY = y0-carY
            print "recalculated pixel %s, %s" % (int(round(carX)), int(round(carY)))
            # if carX == 0:
            img.putpixel((int(round(carX)), int(round(carY))), (255, 255, 0, 255))
            print "------------------------------------"
    img.save('test%s.png' % time.time())
    # img.show()


# initialize()
o_img = read_image()
# print x_coordinate, y_coordinate
# print o_img.getpixel((x_coordinate, y_coordinate))
# displayImageDimensions(o_img)
# iterateThroughImagePixels(o_img)
c_img = copyImage(o_img)

# write_img_with_polar_car(c_img, o_img)
# write_img_test(c_img)

processImage(o_img, c_img)
# calculateDegrees(32, -19, 3)
# calculateDegrees(22, 24,1)
# calculateDegrees(0,0, 4)
# calculateDegrees(-43,-21,2)

# normalize(coeffecients)
# plt.show()