#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data reduction of raw HST/STIS data.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import warnings
import stistools
from astropy.time import Time
from astropy.io import fits


__all__ = []


# The basic class for STIS/MAMA fits files
class MAMAFits(object):
    """

    """
    def __init__(self, dataset_name, type, prefix=None):
        self.dataset = dataset_name

        if prefix is None:
            prefix = ''
        else:
            pass

        # Read the information from the raw data file
        with fits.open(prefix + '%s_raw.fits' % self.dataset) as f:
            self.primary_header = f[0].header
            self.science_extension = f[1]
            self.error_extension = f[2]
            self.quality_extension = f[3]


# The raw STIS/MAMA class
class MAMARaw(MAMAFits):
    """

    """
    def __init__(self, dataset_name, prefix=None):
        super(MAMAFits, self).__init__(dataset_name, prefix)

    # Perform the basic data reduction using ``stistools``
    def basic_2d(self):
        pass

    # Plot the raw 2d image
    def plot_2d(self, extension='science'):
        """

        Args:
            extension:

        Returns:

        """
        raise NotImplementedError('This feature has not been implemented yet.')
