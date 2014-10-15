#!/usr/bin/env python

import numpy as  np

shape = (5392, 3200)
sst = np.fromfile("sst.bin", dtype='f8').reshape(shape)
avgsst = np.fromfile("avgsst.bin", dtype='f8').reshape(shape)
gradmag = np.fromfile("gradmag.bin", dtype='f8').reshape(shape)
lam2 = np.fromfile("lam2.bin", dtype='f8').reshape(shape)
D = np.fromfile("D.bin", dtype='f8').reshape(shape)
easyclouds = np.fromfile("easyclouds.bin", dtype='u1').reshape(shape)
easyfronts = np.fromfile("easyfronts.bin", dtype='u1').reshape(shape)
maskf = np.fromfile("maskf.bin", dtype='u1').reshape(shape)
labels = np.fromfile("labels.bin", dtype='i4').reshape(shape)
