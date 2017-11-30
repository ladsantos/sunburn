#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of HST/COS data.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from blastoise import tools
from scipy.integrate import simps

__all__ = ["COSSpectrum"]


# COS spectra class
class COSSpectrum(object):
    """

    """
    def __init__(self, x1d_file, corrtag_file,
                 good_pixel_limits=((1250, 15180), (1015, 15030))):
        self.path = x1d_file
        self.gpl = good_pixel_limits

        # Read data from x1d file
        with fits.open(self.path) as self.orbit:
            self.header = self.orbit[0].header
            self.data = self.orbit['SCI'].data

        # Read some metadata from the corrtag file
        with fits.open(corrtag_file) as self.meta:
            self.start_JD = self.meta[3].header['EXPSTRTJ']
            self.end_JD = self.meta[3].header['EXPENDJ']

        # Extract the most important information from the data
        self.wavelength = np.array([self.data['WAVELENGTH'][0],
                                    self.data['WAVELENGTH'][1]])
        self.flux = np.array([self.data['FLUX'][0], self.data['FLUX'][1]])
        self.error = np.array([self.data['ERROR'][0], self.data['ERROR'][1]])
        self.gross_counts = np.array([self.data['GCOUNTS'][0],
                                      self.data['GCOUNTS'][1]])
        self.background = np.array([self.data['BACKGROUND'][0],
                                    self.data['BACKGROUND'][1]])
        self.net = np.array([self.data['NET'][0], self.data['NET'][1]])
        self.exp_time = np.array([self.data['EXPTIME'][0],
                                  self.data['EXPTIME'][1]])
