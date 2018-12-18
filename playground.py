import numpy as np
import cv2 as cv
import sys
from collections import OrderedDict
from functools import partial, reduce

WINDOW_NAME = "SHROOMS RADAR"

g_Transformations = OrderedDict(
    [('grayscale', True),
     ('blur', 7),
     ('norm_histogram', False),
     ('binarize', 128),
     ('circles', {'scale': 1,
                  'param1': 1,
                  'param2': 100,
                  'min_radius': 500,
                  'max_radius': 1500})])


def grayscale(image, param):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def gaussian_blur(image, param):
    p = int(param) 

    if p%2 == 0:
        p += 1
        
    return cv.GaussianBlur(image, (p, p), 0)


def threshold(image, param):
    p = int(param)
    _, th = cv.threshold(image, param, 255, cv.THRESH_BINARY_INV)
    return th


def norm_histogram(image, param):
    p = bool(param)
    if p:
        return cv.equalizeHist(image)
    else:
        return image

def hough_circles(image, params):
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, params['scale'], 400,
                              param1=params['param1'], param2=params['param2'],
                              minRadius=params['min_radius'], maxRadius=params['max_radius'])
    print(circles)
    if circles is  None:
        return image

    ret = np.dstack((image, image, image))
    circle_color = (0, 255, 0)
    for (cx, cy, r) in circles[0, :]:
        cv.circle(ret, (cx, cy), r, circle_color, 3)

    return ret


TRANSFORMS = {
    'grayscale': grayscale,
    'blur': gaussian_blur,
    'binarize': threshold,
    'norm_histogram': norm_histogram,
    'circles': hough_circles}


def apply_transformations(input_image, transformations):
    def pick_transformation(stage_image, tr):
        t_name, t_param = tr
        if t_name not in TRANSFORMS:
            print(f"ignoring unknown transform {tr}")
            return stage_image

        tr_fn = TRANSFORMS[t_name]
        im = tr_fn(stage_image, t_param)
        return im
            
    return reduce(pick_transformation, transformations.items(), input_image)
    

def update_image(source_image, transformations):
    transformed_image = apply_transformations(source_image, transformations)
    cv.imshow(WINDOW_NAME, transformed_image)


def is_subattribute(param_name):
    return '/' in param_name

def split_subattribute(param_name):
    return param_name.split('/')
    
def control_callback(image, tr_kind, tr_param):
    global g_Transformations
    if is_subattribute(tr_kind):
        tr_kind, tr_subkind = split_subattribute(tr_kind)
        if tr_kind not in g_Transformations:
            print(f"Control {tr_kind}/{tr_subkind} has no meaning")
            return
        g_Transformations[tr_kind][tr_subkind] = tr_param
    else:
        if tr_kind not in g_Transformations:
            print(f"Control {tr_kind}/{tr_subkind} has no meaning")
            return
        g_Transformations[tr_kind] = tr_param

    update_image(image, g_Transformations)
    

def build_ui(image, controls):

    for control in controls:
        name = control['name']
        transformation = control['attr']

        min_v = control.get('min', 0)
        max_v = control.get('max', 255)
        print(control, name, transformation, min_v, max_v)
        
        cv.createTrackbar(name, WINDOW_NAME, min_v, max_v, partial(control_callback, image, transformation))

    update_image(sample_image, g_Transformations)

    
sample_image = sys.argv[1]
sample_image = cv.imread(sample_image)

cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)


controls = [
    {'name': 'blur', 'min': 1, 'max': 33, 'attr': 'blur'},
    {'name': 'eq hist', 'max': 1, 'attr': 'norm_histogram'},
    {'name': 'binarize', 'min': 1, 'attr': 'binarize'},

    {'name': 'circles scale', 'min': 1, 'max': 10, 'attr': 'circles/scale'},
    {'name': 'circles param1', 'min': 1, 'max': 20, 'attr': 'circles/param1'},
    {'name': 'circles param2', 'min': 1, 'max': 1000, 'attr': 'circles/param2'},
    {'name': 'circles min radius', 'min': 200, 'max': 3000, 'attr': 'circles/min_radius'},
    {'name': 'circles max radius', 'min': 200, 'max': 3000, 'attr': 'circles/max_radius'},
]

build_ui(sample_image, controls)

while True:
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break


cv.destroyAllWindows()
