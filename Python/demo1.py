import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PreProcessing
import TransformDCT
import Quantization
# Adjust image directory
imageRGB_baboon = mpimg.imread('D:/Documents/University/9/Multimedia/Project/baboon.png')
imageRGB_lena = mpimg.imread('D:/Documents/University/9/Multimedia/Project/lena_color_512.png')
# Change the range of the images from [0, 1] to [0, 255]
# If range is not changed quantization cannot work since it will round everything at either 0 or 1
imageRGB_baboon = (imageRGB_baboon * 255).astype(int)
imageRGB_lena = (imageRGB_lena * 255).astype(int)

def part1(image, subimg):
    imageRGB = PreProcessing.crop_if_needed(image)
    imageY, imageCr, imageCb = PreProcessing.convert2ycrcb(imageRGB, subimg)
    reconstructed_imageRGB = PreProcessing.convert2rgb(imageY, imageCr, imageCb, subimg)
    # Plotting both images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imageRGB)
    plt.title('RGB')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_imageRGB)
    plt.title('Reconstructed RGB')
    plt.show()
    return

def part2(image, subimg, qScale):
    qTableLuminance = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])
    qTableChrominance = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                [18, 21, 26, 66, 99, 99, 99, 99],
                                [24, 26, 56, 99, 99, 99, 99, 99],
                                [47, 66, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99]])
    imageRGB = PreProcessing.crop_if_needed(image)
    imageY, imageCr, imageCb = PreProcessing.convert2ycrcb(imageRGB, subimg)
    dctBlockY = np.zeros_like(imageY)
    dctBlockCr = np.zeros_like(imageCr)
    dctBlockCb = np.zeros_like(imageCb)
    QuantizedY = np.zeros_like(imageY)
    QuantizedCr = np.zeros_like(imageCr)
    QuantizedCb = np.zeros_like(imageCb)
    deQuantizedY = np.zeros_like(imageY)
    deQuantizedCr = np.zeros_like(imageCr)
    deQuantizedCb = np.zeros_like(imageCb)
    for i in range(0, imageY.shape[0], 8):
        for j in range(0, imageY.shape[1], 8):
            blockY = imageY[i:i + 8, j:j + 8]
            dctBlockY[i:i + 8, j:j + 8] = TransformDCT.blockDCT(blockY)
            QuantizedY[i:i + 8, j:j + 8] = Quantization.quantizeJPEG(dctBlockY[i:i + 8, j:j + 8], qTableLuminance, qScale)
            deQuantizedY[i:i + 8, j:j + 8] = Quantization.dequantizeJPEG(QuantizedY[i:i + 8, j:j + 8], qTableLuminance, qScale)
            imageY[i:i + 8, j:j + 8] = TransformDCT.iBlockDCT(deQuantizedY[i:i + 8, j:j + 8])
    for i in range(0, imageCr.shape[0], 8):
        for j in range(0, imageCr.shape[1], 8):
            blockCr = imageCr[i:i + 8, j:j + 8]
            dctBlockCr[i:i + 8, j:j + 8] = TransformDCT.blockDCT(blockCr)
            QuantizedCr[i:i + 8, j:j + 8] = Quantization.quantizeJPEG(dctBlockCr[i:i + 8, j:j + 8], qTableChrominance, qScale)
            deQuantizedCr[i:i + 8, j:j + 8] = Quantization.dequantizeJPEG(QuantizedCr[i:i + 8, j:j + 8], qTableLuminance, qScale)
            imageCr[i:i + 8, j:j + 8] = TransformDCT.iBlockDCT(deQuantizedCr[i:i + 8, j:j + 8])
    for i in range(0, imageCb.shape[0], 8):
        for j in range(0, imageCb.shape[1], 8):
            blockCb = imageCb[i:i + 8, j:j + 8]
            dctBlockCb[i:i + 8, j:j + 8] = TransformDCT.blockDCT(blockCb)
            QuantizedCb[i:i + 8, j:j + 8] = Quantization.quantizeJPEG(dctBlockCb[i:i + 8, j:j + 8], qTableChrominance, qScale)
            deQuantizedCb[i:i + 8, j:j + 8] = Quantization.dequantizeJPEG(QuantizedCb[i:i + 8, j:j + 8], qTableLuminance, qScale)
            imageCb[i:i + 8, j:j + 8] = TransformDCT.iBlockDCT(deQuantizedCb[i:i + 8, j:j + 8])
    reconstructed_imageRGB = PreProcessing.convert2rgb(imageY, imageCr, imageCb, subimg)
    # Plotting both images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imageRGB)
    plt.title('RGB')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_imageRGB)
    plt.title('Reconstructed RGB')
    plt.show()
    return

part1(imageRGB_baboon, [4, 2, 0])
part1(imageRGB_lena, [4, 4, 4])
part2(imageRGB_baboon, [4, 2, 2], 0.6)
part2(imageRGB_lena, [4, 4, 4], 5)
