#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Various general tools used by the code.
"""

import numpy as np


def nearest_index(array, target_value):
    """
    Finds the index of a value in ``array`` that is closest to ``target_value``.

    Args:
        array (``numpy.array``): Target array.
        target_value (``float``): Target value.

    Returns:
        index (``int``): Index of the value in ``array`` that is closest to
            ``target_value``.
    """
    index = array.searchsorted(target_value)
    index = np.clip(index, 1, len(array) - 1)
    left = array[index - 1]
    right = array[index]
    index -= target_value - left < right - target_value
    return index


def pick_side(wavelength_array, wavelength_range):
    """
    Finds which side (or index) of the chip corresponds to the requested
    wavelength range.

    Args:
        wavelength_array (``numpy.array``): The wavelength array read from a
            ``UVSpectrum`` object.
        wavelength_range (array-like): Upper and lower limit of wavelength.

    Returns:
        index (``int``): Index of the side (or chip) where the requested
            wavelength falls into.
    """
    if wavelength_range[0] > np.min(wavelength_array[0]) and \
            wavelength_range[1] < np.max(wavelength_array[0]):
        index = 0
    elif wavelength_range[0] > np.min(wavelength_array[1]) and \
            wavelength_range[1] < np.max(wavelength_array[1]):
        index = 1
    else:
        raise ValueError('The requested wavelength range is not available'
                         'in this spectrum.')

    return index
