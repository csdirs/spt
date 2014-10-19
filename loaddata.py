#!/usr/bin/env python

import numpy as  np

shape = (5392, 3200)
sst = np.fromfile("sst.bin", dtype='f8').reshape(shape)
gradmag = np.fromfile("gradmag.bin", dtype='f8').reshape(shape)
m15 = np.fromfile("m15.bin", dtype='f4').reshape(shape)
m16 = np.fromfile("m16.bin", dtype='f4').reshape(shape)
delta = np.fromfile("delta.bin", dtype='f8').reshape(shape)
TQ = np.fromfile("TQ.bin", dtype='i2').reshape(shape)
DQ = np.fromfile("DQ.bin", dtype='i2').reshape(shape)
