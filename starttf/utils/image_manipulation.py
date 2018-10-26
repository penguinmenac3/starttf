# MIT License
# 
# Copyright (c) 2018 Michael Fuerst
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np
import scipy.ndimage.interpolation


def crop(img, start_y, start_x, h, w):
    """
    Crop an image given the top left corner.
    :param img: The image
    :param start_y: The top left corner y coord
    :param start_x: The top left corner x coord
    :param h: The result height
    :param w: The result width
    :return: The cropped image.
    """
    return img[start_y:start_y + h, start_x:start_x + w, :].copy()


def crop_center(img, cropy, cropx):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def resize_image_with_crop_or_pad(img, target_height, target_width):
    """
    Crops and/or pads an image to a target width and height.

    Resizes an image to a target width and height by either cropping the image or padding it with zeros.

    NO CENTER CROP. NO CENTER PAD. (Just fill bottom right or crop bottom right)

    :param img: Numpy array representing the image.
    :param target_height: Target height.
    :param target_width: Target width.
    :return: The cropped and padded image.
    """
    h, w = target_height, target_width
    max_h, max_w, c = img.shape

    # crop
    img = crop_center(img, min(max_h, h), min(max_w, w))

    # pad
    padded_img = np.zeros(shape=(h, w, c), dtype=img.dtype)
    padded_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img

    return padded_img


def _rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.

  Answer from: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr


def rotate_img_and_crop(img, angle):
    """
    Rotate an image and then crop it so that there is no black area.
    :param img: The image to rotate.
    :param angle: The rotation angle in degrees.
    :return: The rotated and cropped image.
    """
    h, w, _ = img.shape
    img = scipy.ndimage.interpolation.rotate(img, angle)
    w, h = _rotatedRectWithMaxArea(w, h, math.radians(angle))
    return crop_center(img, int(h), int(w))
