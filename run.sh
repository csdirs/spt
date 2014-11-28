#!/bin/sh

make || exit 1

#for f in /cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-11/*.nc
#do
#	./spt $f
#done


# "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-10/ACSPO_V2.30_NPP_VIIRS_2014-07-10_1230-1240_20140713.061812.nc"
# "/cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/osb/ykihai/VIIRS_Samples_for_Irina/Select/ACSPO_V2.30_NPP_VIIRS_2014-06-20_1710-1719_20140623.071032.nc"

cat > loaddata.py <<EOF
#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

def intimage(img, **kwargs):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0, bottom=0.20)
    im = ax.imshow(img, **kwargs)
    cbar = fig.colorbar(im)

    rax = plt.axes([0.025, 0.025, 0.13, 0.13])
    cmap_names = ['jet', 'gray', 'binary']
    c = im.get_cmap().name
    active = 0
    for i, name in enumerate(cmap_names):
        if c == name:
            active = i
            break
    radio = RadioButtons(rax, cmap_names, active=active)
    def cmapfunc(label):
        im.set_cmap(label)
        fig.canvas.draw_idle()
    radio.on_clicked(cmapfunc)

    low, high = im.get_clim()
    bot = min(low, np.nanmin(img))
    top = max(high, np.nanmax(img))
    axmin = plt.axes([0.25, 0.025, 0.60, 0.03])
    axmax  = plt.axes([0.25, 0.07, 0.60, 0.03])
    smin = Slider(axmin, 'Min', bot, top, valinit=low)
    smax = Slider(axmax, 'Max', bot, top, valinit=high)

    def update(val):
        im.set_clim(smin.val, smax.val)
        fig.canvas.draw_idle()
    smin.on_changed(update)
    smax.on_changed(update)

    flipxbut = Button(plt.axes([0.25, 0.12, 0.1, 0.04]), 'Flip X')
    def flipx(event):
        img = im.get_array()
        im.set_data(img[:,::-1])
        fig.canvas.draw_idle()
    flipxbut.on_clicked(flipx)

    flipybut = Button(plt.axes([0.36, 0.12, 0.1, 0.04]), 'Flip Y')
    def flipx(event):
        img = im.get_array()
        im.set_data(img[::-1,:])
        fig.canvas.draw_idle()
    flipybut.on_clicked(flipx)

    # return these so we keep a reference to them.
    # otherwise the widget will no longer be responsive
    return im, radio, smin, smax, flipxbut, flipybut

EOF

#./spt /cephfs/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/sst/micros_data/acspo_nc/npp/2014-07-11/ACSPO_V2.30_NPP_VIIRS_2014-07-11_0000-0010_20140714.005638.nc

#DATA_PATH=/home/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/osb/ykihai/VIIRS_Samples_for_Irina/Select/ACSPO_V2.30_NPP_VIIRS_2014-06-20_1710-1719_20140623.071032.nc

DATA_DIR=/home/fhs/data/in/acspo/www.star.nesdis.noaa.gov/pub/sod/osb/petrenko/VIIRS/VIIRS/day
#DATA_PATH=$DATA_DIR/ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1410-1419_20141111.212954.nc
DATA_PATH=$DATA_DIR/ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1740-1750_20141111.225405.nc

OMP_NUM_THREADS=24 ./spt $DATA_PATH | tee -a loaddata.py

#for f in ACSPO_V2.31b02_NPP_VIIRS_2014-10-30_0020-0030_20141110.232954.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-10-30_0400-0409_20141111.005748.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-10-30_1910-1919_20141111.065956.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_0040-0049_20141111.163507.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_0410-0419_20141111.174756.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_0710-0719_20141111.185006.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1030-1039_20141111.200521.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1410-1419_20141111.212954.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1710-1719_20141111.224328.nc \
#	ACSPO_V2.31b02_NPP_VIIRS_2014-11-04_1740-1750_20141111.225405.nc; do
#	
#	OMP_NUM_THREADS=24 ./spt $DATA_DIR/$f
#done
