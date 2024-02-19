import numpy as np
from scipy.fftpack import dct, idct
precision = 8

def blockDCT(block):
    """Apply DCT transform to the input block with level shifting"""
    shift_value = 2**(precision - 1)
    shifted_block = block - (shift_value - 1)
    dct_block = dct(dct(shifted_block, norm='ortho'), norm='ortho')
    return dct_block

def iBlockDCT(dctBlock):
    """Apply inverse DCT transform to the input DCT coefficients with inverse level shifting"""
    inverse_shift_value = 2**(precision - 1)
    shifted_block = idct(idct(dctBlock, norm='ortho'), norm='ortho')
    reconstructed_block = shifted_block + (inverse_shift_value - 1)
    reconstructed_block = np.clip(reconstructed_block, 0, 2**(precision - 1))
    return reconstructed_block