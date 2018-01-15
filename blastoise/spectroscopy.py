#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spectroscopy classes and methods to use in the analysis of HST/COS data.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np


__all__ = []


# Spectral line object
class Line(object):
    """

    """
    def __init__(self, central_wavelength, wavelength_range=None,
                 formation_temperature=None):
        self.central_wavelength = central_wavelength
        self.wavelength_range = wavelength_range
        try:
            self.log_tmax = np.log10(formation_temperature)
        except AttributeError:
            self.log_tmax = None


# The line list object
class LineList(object):
    """

    """
    def __init__(self, lines):
        self.lines = lines


# COS/FUV spectral line list class
class COSFUVLineList(object):
    """

    """
    def __init__(self):
        self.lines = {
            'C III': [Line(1174.93, formation_temperature=10 ** 4.8),
                      Line(1175.26),
                      Line(1175.59),
                      Line(1175.71),
                      Line(1175.99),
                      Line(1176.37)],

            'Si III': [Line(1206.5, formation_temperature=10 ** 4.7)],

            'O V': [Line(1218.344, formation_temperature=10 ** 5.3)],

            'N V': [Line(1238.821, formation_temperature=10 ** 5.2),
                    Line(1242.8040)],

            'Si II': [Line(1264.738, formation_temperature=10 ** 4.2),
                      Line(1265.002)],

            'O I': [Line(1302.168, formation_temperature=10 ** 3.9),
                    Line(1304.858),
                    Line(1306.029)],

            'C II': [Line(1334.532, formation_temperature=10 ** 4.5),
                     Line(1335.708)],

            'Si IV': [Line(1393.755, formation_temperature=10 ** 4.9),
                      Line(1402.770)]
        }
