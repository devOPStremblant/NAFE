from __future__ import division
import cv2
from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
from PIL import Image
import time
import math
import sys
# from libtiff import *
from tifffile import TiffFile

RADIANS_90 = 1.5708
RADIANS_180 = 3.14159
RADIANS_270 = 4.71239

# x_input = int(sys.argv[1])
# y_input = int(sys.argv[2])

sd = 1
coeffecients = []
runningTotal = 0
x0 = y0 = 0
f = open('radius.txt','w')

def initialize():
    imgHelper = ImageHelper()
    imgHelper.readImage()


def read_image():
    return Image.open('sample.tif')


def display_image_dimensions(im):
    w, h = im.size
    print w
    print h
    # im.show()


def iterate_through_image_pixels(im):
    w, h = im.size
    for x in range(0, w):
        for y in range(0, h):
            print im.get_pixel_value(w, h)


def process_image(im, new_img):
    w, h = im.size
    x_suncenter = int(w / 2)
    y_suncenter = int(h / 2)
    print x_suncenter, y_suncenter

    img_array = read_tiff_using_tifffle()

    filtered_pixel_value = (0, 0, 0, 255)
    for x in range (0, w):
        for y in range (0, h):
            filtered_pixel_value = filter_pixel(x, y, img_array)
            new_img.putpixel((x, y), (filtered_pixel_value, filtered_pixel_value, filtered_pixel_value, 255))

    new_img.save('output%s.png' % time.time())
    # newImg.show()


def filter_pixel(x, y, img_array):
    # print "starting loop for %d, %d" % (x, y)
    x0 = 66
    y0 = 66
    q = find_quadrant(x, y, x0, y0)

    xl, yl = calculate_pixel_offset(x, y, q, x0, y0)

    r = find_radius(xl, yl)
    varphi = calculate_degrees(xl, yl, q)

    cartesian_x, cartesian_y = polToCar(r, varphi)
    cartesian_x += x0
    cartesian_y = y0 - cartesian_y
    print "******** Normal Pixel Value ******** %s" % get_16bit_pixel_value(int(round(cartesian_x)), int(round(cartesian_y)), img_array)

    f.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (x, y, r, varphi, cartesian_x, cartesian_y))

    lower_rho = r - 2 * sd
    upper_rho = r + 2 * sd

    if r == 0:
        lower_phi = upper_phi = varphi
    else:
        lower_phi = (varphi - ((2 * sd) / r))
        upper_phi = (varphi + ((2 * sd) / r))

    print "Rho Values range from %s to %s" % (lower_rho, upper_rho)
    print "Phi values range from %s to %s" % (lower_phi, upper_phi)

    sum_a = 0
    sum_b = 0
    rho_index = lower_rho
    phi_index = lower_phi

    while rho_index < upper_rho:
        while phi_index < upper_phi:
            print "Current Rho is %f and Current Phi is %f " % (rho_index, phi_index)
            rho_param_for_f = r + rho_index
            phi_param_for_f = angle_addition(phi_index, varphi)
            print "f() function is done on %s and %s values" % (rho_param_for_f, phi_param_for_f)

            c_x, c_y = polToCar(rho_param_for_f, phi_param_for_f)
            c_x += x0
            c_y = y0 - c_y
            pv = get_16bit_pixel_value(int(round(c_x)), int(round(c_y)), img_array)
            print "f() function returns %s " % (pv,)

            kernel_value = calculate_kernel(rho_index, phi_index, r, varphi)
            print "C() function of %s and %s is %s" % (rho_index, phi_index, kernel_value)
            sum_a += (pv[0] * kernel_value)
            sum_b += kernel_value
            phi_index += 1
        rho_index += 1

    print "Sum Gaussian A: %s" % sum_a
    print "Sum Gaussian B: %s" % sum_b

    if sum_b == 0:
        new_pixel_value = 0
    else:
        new_pixel_value = get_16bit_pixel_value(x, y, img_array)[0] - int(sum_a / sum_b)
    return new_pixel_value


def read_tiff_using_libtiff():
    tiff = TIFF.open('filename')
    image = tiff.read_image()
    return image

def readTIFF16():
    """Read 16bits TIFF"""
    im = Image.open('sample.tif')
    out = np.fromstring(
        im.tobytes(),
        np.ushort
        ).reshape(tuple(list(im.size)))
    return out


def angle_addition(a, b):
    return np.arctan((a + b) / (1 - (a * b)))


def angle_difference(a, b):
    return np.arctan((a - b) / (1 + (a * b)))


def reverse_offset(x, y, x0, y0):
    return x+x0, y0-y


def calculate_pixel_offset(x, y, q, x0, y0):
    if q == 0:
        return [x, y]
    return [x - x0, y0 - y]


def polToCar(radius, varphi):
    # print "varphi in degrees: %f" % varphi
    x_coordinate = radius * np.cos(abs(varphi))
    y_coordinate = radius * np.sin(abs(varphi))

    # if y_coordinate < 0:
    #     x = math.ceil(x_coordinate)
    # else:
    #     x = math.floor(x_coordinate)

    # if x_coordinate < 0:
    #     y = math.ceil(y_coordinate)
    # else:
    #     y = math.floor(y_coordinate)

    return [x_coordinate,y_coordinate]


def get_16bit_pixel_value(x, y, img_array):
    print "Looking up 16 Bit Pixel information for (%s, %s)" % (x, y)
    return img_array[x][y]

def get_pixel_value(x, y, img):
    # print x, y
    try:
        return img.getpixel((x, y))
    except IndexError:
        return (0,0,0, 255)


def find_quadrant(x, y, x0, y0):
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
    

def find_radius(xl, yl):
    r = np.sqrt((xl*xl) + (yl*yl))
    return r


def calculate_degrees(xl, yl, q):
    # print "Quadrant %d" % q
    # print "(xl, yl) : %d, %d" % (xl, yl)

    if xl == 0 and yl == 0:
        return 0

    if xl == 0 :
        if yl < 0:
            return RADIANS_270
        else:
            return RADIANS_90
    if yl == 0:
        if xl > 0:
            return 0
        else:
            return RADIANS_180
    
    if xl == 0 or q == 0:
        return np.arctan(0)

    sign_factor = 1
    if (yl/xl) < 0:
        sign_factor = -1

    # degrees = sign_factor * (np.arctan(abs(yl) / abs(xl)) + ((q - 1) * RADIANS_90))
    if q == 1:
        degrees = np.arctan(abs(yl) / abs(xl)) + ((q-1) * RADIANS_90)
    elif q == 2:
        degrees = sign_factor * (np.arctan(abs(xl) / abs(yl)) + RADIANS_90)
    elif q == 4:
        degrees = np.arctan(abs(xl) / abs(yl)) + ((q-1) * RADIANS_90)
    else:
        degrees = np.arctan(abs(yl) / abs(xl)) + ((q-1) * RADIANS_90)
    return degrees


def calculate_kernel(rho, phi, r, varphi):
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
    exp_param = numerator / 2 * np.square(sd)
    # print "(%d, %d)" % (r, varphi)
    return np.exp((-1) * exp_param)
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
            q = find_quadrant(x, y, x0, y0)

            xl, yl = calculate_pixel_offset(x, y, q, x0, y0)

            r = find_radius(xl, yl)
            varphi = calculate_degrees(xl, yl, q)

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

def read_tiff_using_tifffle():
    with TiffFile('sample.tif', 'rb') as file:
        return file.asarray()


def test_this_pixel(x, y, img):
    print "Input Pixels (%s, %s)" % (x, y)
    q = find_quadrant(x, y, 66, 66)
    print "Quadrant %s" % q
    xl, yl = calculate_pixel_offset(x, y, q, 66, 66)
    print "Pixel Offset: (%s, %s)" % (xl, yl)
    print "Pixel value %s: " % (get_pixel_value(xl, yl, img),)
    radius = find_radius(xl, yl)
    print "Radius %s: " %  radius
    radians = calculate_degrees(xl, yl, q)
    print "Radians %s: " %  radians
    x_new, y_new = polToCar(radius, radians)
    print "Recalculated Cartesian: %s, %s" %  (x_new, y_new)
    x_new += 66
    y_new = 66 - y_new
    print "Recalculated Cartesian: %s, %s" %  (int(round(x_new)), int(round(y_new)))


# initialize()
o_img = read_image()

# test_this_pixel(x_input,y_input, o_img)

# display_image_dimensions(o_img)
# iterate_through_image_pixels(o_img)
c_img = copy_image(o_img)

# readTIFF16()
# read_tiff_using_tifffle()

# read_tiff_using_libtiff()
# filter_pixel(20, 20, o_img)

# write_img_with_polar_car(c_img, o_img)
# write_img_test(c_img)

process_image(o_img, c_img)
# calculate_degrees(32, -19, 3)
# calculate_degrees(22, 24,1)
# calculate_degrees(0,0, 4)
# calculate_degrees(-43,-21,2)

# normalize(coeffecients)
# plt.show()