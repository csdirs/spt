#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')

import glob
import matplotlib.pyplot as plt
import numpy as np

def combinelut():
    filenames = sorted(glob.glob("*_lut.npy"))
    shape = (16, 13, 17)    # sst, delta, omega
    luts = np.zeros((10, np.prod(shape)))

    for i, f in enumerate(filenames):
        print "file:", f
        luts[i,:] = np.ravel(np.load(f))

    nocean = np.sum(luts == 0, axis=0)
    ncloud = np.sum(luts == 1, axis=0)

    lut = -1 + np.zeros(np.prod(shape))
    lut[nocean > ncloud] = 0
    lut[ncloud > nocean] = 1
    lut.shape = shape
    return luts, lut

def main():
    luts, lut = combinelut()
    shape = lut.shape
    for i in xrange(16):
        print "fig", i
        plt.figure()
        plt.imshow(lut[i,:,:], extent=(-3, 5, 3, -3))
        plt.colorbar()
        plt.savefig("lut_delta_omega_%02d.png" % (i,))
        plt.close()

    for k in xrange(luts.shape[0]):
        print "granule", k
        e = luts[k,:].reshape(shape)
        for i in xrange(16):
            plt.figure()
            plt.imshow(e[i,:,:], extent=(-3, 5, 3, -3))
            plt.colorbar()
            plt.savefig("granule%02d_lut_delta_omega_%02d.png" % (k, i))
            plt.close()

if __name__ == '__main__':
    main()
