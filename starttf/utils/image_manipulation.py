import numpy as np


def resize_image_with_crop_or_pad(img, target_height, target_width, target_channels=3):
    """
    Crops and/or pads an image to a target width and height.

    Resizes an image to a target width and height by either cropping the image or padding it with zeros.

    NO CENTER CROP. NO CENTER PAD. (Just fill bottom right or crop bottom right)

    :param img: Numpy array representing the image.
    :param target_height: Target height.
    :param target_width: Target width.
    :param target_channels: Target channels (default 3).
    :return: The cropped and padded image.
    """
    h, w, c = target_height, target_width, target_channels
    max_h, max_w, max_c = img.shape

    # crop
    img = img[0:min(max_h, h), 0:min(max_w, w), 0:min(max_c, c)]

    # pad
    padded_img = np.zeros(shape=(h, w, c))
    padded_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img

    return padded_img
