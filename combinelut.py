#!/usr/bin/env python

import glob
import numpy as np

def combinelut():
    filenames = glob.glob("*_lut.npy")
    shape = (16, 13, 17)
    luts = np.zeros((10, np.prod(shape)))

    for i, f in enumerate(filenames):
        luts[i,:] = np.ravel(np.load(f))

    nocean = np.sum(luts == 0, axis=0)
    ncloud = np.sum(luts == 1, axis=0)

    lut = -1 + np.zeros(np.prod(shape))
    lut[nocean > ncloud] = 0
    lut[ncloud > nocean] = 1
    lut.shape = shape
    return luts, lut
