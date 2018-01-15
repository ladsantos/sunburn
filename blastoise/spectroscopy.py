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
                 line_width=None, formation_temperature=None):
        self.central_wavelength = central_wavelength
        try:
            self.log_tmax = np.log10(formation_temperature)
        except AttributeError:
            self.log_tmax = None

        standard_line_width = 0.25
        if wavelength_range is not None:
            self.wavelength_range = wavelength_range
        elif line_width is not None:
            self.wavelength_range = [central_wavelength - line_width,
                                     central_wavelength + line_width]
        else:
            self.wavelength_range = [central_wavelength - standard_line_width,
                                     central_wavelength + standard_line_width]


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
    def __init__(self, wavelength_shift=0.0, range_factor=1.0):
        self.lines = {
            'C III': [Line(1174.93, formation_temperature=10 ** 4.8),
                      Line(1175.26),
                      Line(1175.59),
                      Line(1175.71),
                      Line(1175.99),
                      Line(1176.37)],

            'Si III': [Line(1206.5, formation_temperature=10 ** 4.7,
                            line_width=0.15)],

            'O V': [Line(1218.344, formation_temperature=10 ** 5.3,
                         line_width=0.15)],

            'N V': [Line(1238.821, formation_temperature=10 ** 5.2),
                    Line(1242.8040)],

            'Si II': [Line(1264.738, formation_temperature=10 ** 4.2),
                      Line(1265.002)],

            'O I': [Line(1302.168, formation_temperature=10 ** 3.9),
                    Line(1304.858),
                    Line(1306.029)],

            'C II': [Line(1334.532, formation_temperature=10 ** 4.5,
                          line_width=0.10),
                     Line(1335.708, line_width=0.20)],

            'Si IV': [Line(1393.755, formation_temperature=10 ** 4.9,
                           line_width=0.15),
                      Line(1402.770, line_width=0.05)]
        }

        # Correct the lines by wavelength shift and line widths by the range
        # factor
        for transition in self.lines:
            for line in self.lines[transition]:
                delta = line.central_wavelength - line.wavelength_range[0]
                delta *= range_factor
                line.central_wavelength += wavelength_shift
                line.wavelength_range = [line.central_wavelength - delta,
                                         line.central_wavelength + delta]

