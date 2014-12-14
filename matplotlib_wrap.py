#!/usr/bin/env python2

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

def intimage(img, **kwargs):
    """Interactive imshow with widgets.
    """
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


def cbar():
    """Alternative to plt.colorbar() function because
    it seems like that function only work with plt.imshow
    and not with ax.imshow (used by Fig class below).
    """
    ax = plt.gca()
    # get the mappable, the 1st and the 2nd are the x and y axes
    # XXX: maybe use ax.get_images() instead of ax.get_children()?
    im = ax.get_children()[2]
    plt.colorbar(im, ax=ax)

# Remove colorbar.
# Doesn't work well. leaves empty space where the colorbar was.
#def nocbar():
#    fig = plt.gcf()
#    fig.delaxes(fig.axes[1])

def cmap(name):
    """Set colormap to name
    """
    plt.gca().get_children()[2].set_cmap(name)
    plt.draw()

def clim(low, high):
    """Set colorbar min/max to low/high.
    Overwrite the standard plt.clim because gci() returns None
    for Fig class figures.
    """
    plt.gca().get_children()[2].set_clim(low, high)
    plt.draw()

def flipx():
    """Flip x-axis (rows) of image.
    """
    im = plt.gca().get_children()[2]
    img = im.get_array()
    im.set_data(img[:,::-1])
    plt.draw()

def flipy():
    """Flip y-axis (columns) of image.
    """
    im = plt.gca().get_children()[2]
    img = im.get_array()
    im.set_data(img[::-1,:])
    plt.draw()

class Fig(object):
    """Wrapper around imshow with synchronized pan/zoom.
    """
    def __init__(self, img, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(img, **kwargs)
        #ax.autoscale(False)
        ax.set_adjustable('box-forced')
        plt.colorbar(im, ax=ax)
        self.ax = ax

    def imshow(self, img, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, sharex=self.ax, sharey=self.ax)
        im = ax.imshow(img, **kwargs)
        #ax.autoscale(False)
        ax.set_adjustable('box-forced')
        plt.colorbar(im, ax=ax)
        return ax


_SHAREDFIG = None

def imagesh(img, **kwargs):
    global _SHAREDFIG

    if _SHAREDFIG is None:
        _SHAREDFIG = Fig(img, **kwargs)
    else:
        _SHAREDFIG.imshow(img, **kwargs)


def loadnc(filename):
    return np.array(netCDF4.Dataset(filename).variables["data"])
