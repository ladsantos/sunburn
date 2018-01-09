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
                 good_pixel_limits=((1260, 15170), (1025, 15020))):
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

        # If ``good_pixel_limits`` is set to ``None``, then the data will be
        # retrieved from the file in its entirety. Otherwise, it will be
        # retrieved using the limits established by ``good_pixel_limits``
        if self.gpl is None:
            self.gpl = ((0, -1), (0, -1))
        else:
            pass

        # Extract the most important information from the data
        i00 = self.gpl[0][0]
        i01 = self.gpl[0][1]
        i10 = self.gpl[1][0]
        i11 = self.gpl[1][1]
        self.wavelength = np.array([self.data['WAVELENGTH'][0][i00:i01],
                                    self.data['WAVELENGTH'][1][i10:i11]])
        self.flux = np.array([self.data['FLUX'][0][i00:i01],
                              self.data['FLUX'][1][i10:i11]])
        self.error = np.array([self.data['ERROR'][0][i00:i01],
                               self.data['ERROR'][1][i10:i11]])
        self.gross_counts = np.array([self.data['GCOUNTS'][0][i00:i01],
                                      self.data['GCOUNTS'][1][i10:i11]])
        self.background = np.array([self.data['BACKGROUND'][0][i00:i01],
                                    self.data['BACKGROUND'][1][i10:i11]])
        self.net = np.array([self.data['NET'][0][i00:i01],
                             self.data['NET'][1][i10:i11]])
        self.exp_time = np.array([self.data['EXPTIME'][0],
                                  self.data['EXPTIME'][1]])

        # Instantiating useful global variables
        self.sensitivity = None

    # Compute the correct errors for the HST/COS observation
    def compute_proper_error(self):
        """

        """
        self.sensitivity = self.flux / self.net / self.exp_time
        self.error = (self.gross_counts + 1.0) ** 0.5 * self.sensitivity

    # Compute the integrated flux in a given wavelength range
    def integrated_flux(self, wavelength_range,
                        uncertainty_method='quadratic_sum'):
        """

        Args:
            wavelength_range:
            uncertainty_method:

        Returns:

        """
        if wavelength_range[0] > np.min(self.wavelength[0]) and \
                wavelength_range[1] < np.max(self.wavelength[0]):
            ind = 0
        elif wavelength_range[0] > np.min(self.wavelength[1]) and \
                wavelength_range[1] < np.max(self.wavelength[1]):
            ind = 1
        else:
            raise ValueError('The requested wavelength range is not available'
                             'in this spectrum.')

        min_wl = tools.nearest_index(self.wavelength[ind], wavelength_range[0])
        max_wl = tools.nearest_index(self.wavelength[ind], wavelength_range[1])
        # The following line is hacky, but it works
        delta_wl = self.wavelength[ind][1:] - self.wavelength[ind][:-1]
        int_flux = simps(self.flux[ind][min_wl:max_wl],
                         x=self.wavelength[ind][min_wl:max_wl])

        # Compute the uncertainty of the integrated flux
        if uncertainty_method == 'quadratic_sum':
            uncertainty = np.sqrt(np.sum((delta_wl[min_wl:max_wl] *
                                          self.error[ind][min_wl:max_wl]) ** 2))
        elif uncertainty_method == 'bootstrap':
            n_samples = 10000
            # Draw a sample of spectra and compute the fluxes for each
            samples = np.random.normal(loc=self.flux[ind][min_wl:max_wl],
                                       scale=self.error[ind][min_wl:max_wl],
                                       size=[n_samples, max_wl - min_wl])
            fluxes = []
            for i in range(n_samples):
                fluxes.append(simps(samples[i],
                                    x=self.wavelength[ind][min_wl:max_wl]))
            fluxes = np.array(fluxes)
            uncertainty = np.std(fluxes)
        else:
            raise ValueError('This value of ``uncertainty_method`` is not '
                             'accepted.')

        return int_flux, uncertainty


# STIS spectrum class
class STISSpectrum(object):
    """

    """
    def __init__(self):
        pass
