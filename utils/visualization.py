import cv2
import numpy as np

def calculate_angle(im, re):
    """
    param: im(float): imaginary parts of the plural
    param: re(float): real parts of the plural
    return: The angle at which the objects rotate
    around the Z axis in the lidar coordinate system
    """
    if re > 0:
        return np.arctan(im / re)
    elif im < 0:
        return -np.pi + np.arctan(im / re)
    else:
        return np.pi + np.arctan(im / re)

def draw_rotated_box(img, cy, cx, w, h, angle, color):
    """
    param: img(array): RGB image
    param: cy(int, float):  Here cy is cx in the image coordinate system
    param: cx(int, float):  Here cx is cy in the image coordinate system
    param: w(int, float):   box's width
    param: h(int, float):   box's height
    param: angle(float): rz
    param: color(tuple, list): the color of box, (R, G, B)
    """
    left = int(cy - w / 2)
    top = int(cx - h / 2)
    right = int(cx + h / 2)
    bottom = int(cy + h / 2)
    ro = np.sqrt(pow(left - cy, 2) + pow(top - cx, 2))
    a1 = np.arctan((w / 2) / (h / 2))
    a2 = -np.arctan((w / 2) / (h / 2))
    a3 = -np.pi + a1
    a4 = np.pi - a1
    rotated_p1_y = cy + int(ro * np.sin(angle + a1))
    rotated_p1_x = cx + int(ro * np.cos(angle + a1))
    rotated_p2_y = cy + int(ro * np.sin(angle + a2))
    rotated_p2_x = cx + int(ro * np.cos(angle + a2))
    rotated_p3_y = cy + int(ro * np.sin(angle + a3))
    rotated_p3_x = cx + int(ro * np.cos(angle + a3))
    rotated_p4_y = cy + int(ro * np.sin(angle + a4))
    rotated_p4_x = cx + int(ro * np.cos(angle + a4))
    center_p1p2y = int((rotated_p1_y + rotated_p2_y) * 0.5)
    center_p1p2x = int((rotated_p1_x + rotated_p2_x) * 0.5)
    cv2.line(img, (rotated_p1_y, rotated_p1_x), (rotated_p2_y, rotated_p2_x),
             color, 1)
    cv2.line(img, (rotated_p2_y, rotated_p2_x), (rotated_p3_y, rotated_p3_x),
             color, 1)
    cv2.line(img, (rotated_p3_y, rotated_p3_x), (rotated_p4_y, rotated_p4_x),
             color, 1)
    cv2.line(img, (rotated_p4_y, rotated_p4_x), (rotated_p1_y, rotated_p1_x),
             color, 1)
    cv2.line(img, (center_p1p2y, center_p1p2x), (cy, cx), color, 1)