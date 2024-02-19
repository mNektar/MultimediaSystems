import numpy as np
from scipy import ndimage

def crop_if_needed(imageRGB):
    # Get the dimensions of the image
    [width, height] = imageRGB.shape[:2]
    # Check if dimensions are multiples of 8
    if width % 8 != 0 or height % 8 != 0:
        # Calculate new dimensions that are multiples of 8
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        # Crop the image to the new dimensions
        imageRGB = imageRGB.crop((0, 0, new_width, new_height))
    return imageRGB

def convert2ycrcb(imageRGB, subimg):
    # Seperate each color channel
    R = imageRGB[:, :, 0]
    G = imageRGB[:, :, 1]
    B = imageRGB[:, :, 2]
    # Calculate channels for YCrCb
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128
    # Subsample chromaticity fields based on the given subsampling format
    sub_y, sub_cr, sub_cb = subimg
    if sub_y == 4 and sub_cr == 4 and sub_cb == 4:
        return Y, Cr, Cb
    elif sub_y == 4 and sub_cr == 2 and sub_cb == 2:
        return Y, ndimage.zoom(Cr, 0.5), ndimage.zoom(Cb, 0.5)
    elif sub_y == 4 and sub_cr == 2 and sub_cb == 0:
        return Y, ndimage.zoom(Cr, 0.25), ndimage.zoom(Cb, 0.25)
    else:
        raise ValueError("Unsupported subsampling format.")

def convert2rgb(imageY, imageCr, imageCb, subimg):
    # Oversample chromaticity fields based on the given subsampling format
    sub_y, sub_cr, sub_cb = subimg
    Y, Cr, Cb = imageY, imageCr, imageCb
    if sub_y == 4 and sub_cr == 2 and sub_cb == 2:
        Cr = ndimage.zoom(imageCr, 2)
        Cb = ndimage.zoom(imageCb,2)
    elif sub_y == 4 and sub_cr == 2 and sub_cb == 0:
        Cr = ndimage.zoom(imageCr, 4)
        Cb = ndimage.zoom(imageCb, 4)
    else:
        if not (sub_y == 4 and sub_cr == 4 and sub_cb == 4):
            raise ValueError("Unsupported subsampling format.")
    # Calculate channels for RGB
    R = Y + 1.402 * (Cr - 128) - 0.001 * (Cb - 128)
    G = Y - 0.714 * (Cr - 128) - 0.344 * (Cb - 128)
    B = Y + 0.001 * (Cr - 128) + 1.772 * (Cb - 128)
    imageRGB = np.stack([R, G, B], axis=-1)
    # Find the minimum and maximum values in the original data
    min_value = np.min(imageRGB)
    max_value = np.max(imageRGB)
    # Normalize the image to the float range [0, 1]
    normalized_image = (imageRGB - min_value) / (max_value - min_value)
    return normalized_image
