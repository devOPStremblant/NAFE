from __future__ import division
import cv2
from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
from PIL import Image
import time
import math


sd = 64
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
    min_value = 0.0039215
    max_value = 1.0576288
    difference = max_value - min_value
    for x in range (0, w):
        for y in range (0, h):
            # print "starting loop for %d, %d" % (x, y)
            
            q = findQuadrant(x, y, x0, y0)

            xl, yl = calculatePixelOffset(x, y, q, x0, y0)

            r = findRadius(xl, yl)
            varphi = calculateDegrees(xl, yl, q)
            

            carX, carY = polToCar(r, varphi)
            carX += x0
            carY = y0-carY

            # f.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (xl, yl, r, varphi, carX, carY))
            # print '%s\t%s\t%s\t%s\t%s\t%s\n' % (xl, yl, r, varphi, carX, carY)

            lower_rho = r - 2*sd
            upper_rho = r + 2*sd

            if r == 0:
                lower_phi = varphi
                upper_phi = varphi
            else:
                lower_phi = varphi - ((2 * sd) / r)
                upper_phi = varphi + ((2 * sd) / r)

            # print "LowerRho Value: %f" %  lowerRho
            # print "Upper Value: %f" % upperRho

            sum_a = 0
            sum_b = 0
            rho_index = lower_rho
            phi_index = lower_phi

            while rho_index < upper_rho:
                while phi_index < upper_phi:
                    # print "Current Rho is %f and Current Phi is %f " % (rhoIndex, phiIndex)

                    c_x, c_y = polToCar(r + rho_index, phi_index + varphi)
                    c_x += x0
                    c_y = y0 - c_y

                    f.write('%s\t%s\t%s\t%s\n' % (int(round(c_x)), int(round(c_y)), (r + rho_index), (phi_index + varphi)))
                    pv = get_pixel_value(c_x, c_y, im)
                    # This line has to go away
                    try:
                        newImg.putpixel((int(round(c_x)), int(round(c_y))), pv)
                    except IndexError:
                        print "do nothing"

                    # print pv
                    kernel_value = calculateKernel(rho_index, phi_index, r, varphi)
                    sum_a += (pv[0] * kernel_value)
                    sum_b += kernel_value
                    phi_index += 0.1
                rho_index += 0.1

            print sum_a, sum_b
            quotient = 0
            if sum_b > 0:
                quotient = sum_a / sum_b

            to_subtract = 100 * ((quotient - min_value) / difference)

            original_pixel_value = get_pixel_value(x, y, im)[0]

            # print to_subtract
            # print("%s" % (int(round(to_subtract))))

            # f.write('%s\t%s\t%s\t%s\n' % (x,y,sum_a, sum_b))
            # if round(sum_b) == 0 or original_pixel_value == 0:
            #     new_pixel_value = 0
            # else:
            new_pixel_value = original_pixel_value - int(round(to_subtract))
            # new_pixel_value = get_pixel_value(x, y, im)[0] - int(sum_a / sum_b)

            # if new_pixel_value > 0:
                # print x, y
            # if r < 40:    
            # newImg.putpixel((x, y), (new_pixel_value, new_pixel_value, new_pixel_value, 255))

            # if int(r) % 2 == 0:
            #     newImg.putpixel((x, y), (new_pixel_value, new_pixel_value, new_pixel_value, 255))
            # else:
            #     newImg.putpixel((x, y), (0, 0, 255, 255))
            # print "Radius of %d, %d is %f" % (x, y, r)
            # print "Varphi of %d, %d is %f" % (xl, yl, varphi)
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
    x_coordinate = radius * np.cos(np.deg2rad(varphi))
    y_coordinate = radius * np.sin(np.deg2rad(varphi))

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


def copy_image(o_img):
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

            xl, yl = calculatePixelOffset(x, y, q, x0, y0)

            r = findRadius(xl, yl)
            varphi = calculateDegrees(xl, yl, q)

            # print "polar coordinate %s, %s" % (r, varphi)

            carX, carY = polToCar(r, varphi)
            # print "Pol2Car before adjusting %s, %s" % (round(carX), round(carY))
            carX += x0
            carY = y0-carY
            # print "recalculated pixel %s, %s" % (int(round(carX)), int(round(carY)))

            p_x = int(round(carX))
            p_y = int(round(carY))

            pixel_value = get_pixel_value(p_x, p_y, o_img)
            # if carX == 0:
            img.putpixel((p_x, p_y), pixel_value)
            print "------------------------------------"
    img.save('test%s.png' % time.time())
    # img.show()


# initialize()
o_img = read_image()
# displayImageDimensions(o_img)
# iterateThroughImagePixels(o_img)
c_img = copy_image(o_img)

# write_img_with_polar_car(c_img, o_img)
# write_img_test(c_img)

processImage(o_img, c_img)
# calculateDegrees(32, -19, 3)
# calculateDegrees(22, 24,1)
# calculateDegrees(0,0, 4)
# calculateDegrees(-43,-21,2)

# normalize(coeffecients)
# plt.show()