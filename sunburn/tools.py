#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Various general tools used by the code.
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.signal import correlate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic


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
        raise ValueError('The requested wavelength range (%i-%i) is not '
                         'available in this spectrum.' % (wavelength_range[0],
                                                          wavelength_range[1]))

    return index


def make_bins(array):
    """
    Transform an array (e.g., wavelengths) into a bin-array (bins of
    wavelengths, useful to make a barplot).

    Args:
        array:

    Returns:

    """
    bin_array = (array[:-1] + array[1:]) / 2
    spacing = bin_array[1] - bin_array[0]
    bin_array -= spacing
    bin_array = np.append(bin_array, bin_array[-1] + spacing)
    bin_array = np.append(bin_array, bin_array[-1] + spacing)
    return bin_array


def cross_correlate(line, spectrum, wavelength_span=1 * u.angstrom,
                    mask_width_factor=5):
    """
    Computes the cross-correlation function of the spectrum in with a
    square-function mask.

    Args:
        line (`spectroscopy.Line`):
        spectrum (`hst_observation.UVSpectrum`):
        wavelength_span:
        mask_width_factor:

    Returns:

    """
    if isinstance(wavelength_span, u.Quantity):
        wavelength_span = wavelength_span.to(u.angstrom).value
    else:
        pass

    def square(x, x0, width):
        if (x0 - width / 2) < x <= (x0 + width / 2):
            y = 1 / width
        else:
            y = 0
        return y

    w0 = line.central_wavelength
    wl_width = (line.wavelength_range[1] -line.wavelength_range[0]) / 2
    dw = wavelength_span / 2

    # Find the interval where to compute the ccf
    ind = pick_side(spectrum.wavelength, [w0 - dw, w0 + dw])
    min_wl = nearest_index(spectrum.wavelength[ind], w0 - dw)
    max_wl = nearest_index(spectrum.wavelength[ind], w0 + dw)
    mask = np.array([square(xk, w0, wl_width / mask_width_factor)
                     for xk in spectrum.wavelength[ind][min_wl:max_wl]])

    ccf = correlate(spectrum.flux[ind][min_wl:max_wl], mask, mode='same')
    d_shift = (spectrum.wavelength[ind][min_wl:max_wl] - w0) / w0 * \
        c.c.to(u.km / u.s).value

    # Setting the initial guesses to fit a Gaussian to the CCF
    mult_factor = 1E14
    ds_0 = 0
    fwhm_0 = wl_width / w0 * c.c.to(u.km / u.s).value
    ampl_0 = np.max(ccf) * mult_factor
    coeff = fit_gaussian(d_shift, ccf * mult_factor, ds_0, fwhm_0, ampl_0)
    return d_shift, ccf, coeff


def fit_gaussian(x, y, x_0, fwhm_0, amplitude_0):
    """
    Fit a Gaussian to the (x, y) curve, using as a first guess the values of the
    Gaussian center `x_0`, the `fwhm_0` and `amplitude_0`.

    Args:
        x:
        y:
        x_0:
        fwhm_0:
        amplitude_0:

    Returns:
        x_f:
        fwhm_f:
        amplitude_f:

    """
    # The function that defines a Gaussian
    def gaussian(xs, *p):
        mu, sigma, ampl = p
        return ampl * np.exp(-(xs - mu) ** 2 / (2 * sigma ** 2))

    # The initial guess
    p0 = [x_0, fwhm_0, amplitude_0]

    # Perform the fit
    coeff, var = curve_fit(gaussian, x, y, p0=p0)
    return coeff


# Apply Doppler shift to a spectrum
def doppler_shift(velocity, ref_wl, wavelength, flux, uncertainty,
                  interp_type='linear', fill_value='extrapolate'):
    """

    Args:
        velocity:
        ref_wl:
        wavelength:
        flux:
        uncertainty:
        interp_type:

    Returns:

    """
    l_speed = c.c.to(u.km / u.s).value
    try:
        dv = velocity.to(u.km / u.s).value
    except AttributeError:
        dv = velocity

    shift = dv / l_speed * ref_wl
    old_wavelength = np.copy(wavelength)
    old_flux = np.copy(flux)
    old_error = np.copy(uncertainty)
    new_wv = np.copy(wavelength) + shift
    func0 = interp1d(new_wv, old_flux, kind=interp_type,
                     fill_value=fill_value, bounds_error=False)
    func1 = interp1d(new_wv, old_error, kind=interp_type,
                     fill_value=fill_value, bounds_error=False)
    new_flux = func0(old_wavelength)
    new_uncertainty = func1(old_wavelength)
    return new_flux, new_uncertainty


# Bin a spectrum to a specific Doppler shift width
def bin_spectrum(bin_width, wavelength, doppler_shift, flux, flux_uncertainty):
    """

    Args:
        wavelength:
        doppler_shift:
        flux:
        flux_uncertainty:

    Returns:

    """
    bw = bin_width
    wv = wavelength
    ds = doppler_shift
    f = flux
    u = flux_uncertainty
    v_bins = np.arange(min(ds), max(ds) + bw, bw)

    binned_data, edges, inds = binned_statistic(ds, [wv, ds, f], bins=v_bins,
                                                statistic='mean')
    wv_bin = binned_data[0]
    v_bin = binned_data[1]
    f_bin = binned_data[2]
    u_bin, edges, inds = binned_statistic(ds, u ** 2, bins=v_bins,
                                          statistic='sum')
    u_count, edges, inds = binned_statistic(ds, u ** 2, bins=v_bins,
                                            statistic='count')
    u_bin = u_bin ** 0.5 / u_count
    return wv_bin, v_bin, f_bin, u_bin