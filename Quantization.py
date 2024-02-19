import numpy as np

def quantizeJPEG(dctBlock, qTable, qScale):
    # Apply quantization to each DCT coefficient in the block
    qBlock = np.round(dctBlock / (qTable * qScale))
    return qBlock

def dequantizeJPEG(qBlock, qTable, qScale):
    # Apply dequantization to each quantized DCT coefficient in the block
    dctBlock = qBlock * (qTable * qScale)
    return dctBlock

