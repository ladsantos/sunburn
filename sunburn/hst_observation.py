#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
HST observation module.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from . import tools
from scipy.integrate import simps

__all__ = ["Visit", "UVSpectrum", "COSSpectrum", "STISSpectrum"]


# HST visit
class Visit(object):
    """
    HST visit object. It is used as a container for a collection of HST
    observational data from a single visit.

    Args:

        dataset_name (``str``): Name of the dataset, as downloaded from MAST.
            For example, if the 1-d extracted spectrum file is named
            ``'foo_x1d.fits'``, then the dataset name is ``'foo'``.

        instrument (``str``): Instrument name. Currently, the only options
            available are ``'cos'`` and ``'stis'``.

        good_pixel_limits (``tuple``, optional): Tuple containing the good pixel
            limits of the detector, with shape (2, 2), where the first line is
            the limits for the red chip, and the second line is for the blue
            chip. If ``None``, use all pixels. Default is ``None``.
    """
    def __init__(self, dataset_name, instrument, good_pixel_limits=None):

        self.orbit = {}

        for i in range(len(dataset_name)):
            if instrument == 'cos':
                self.orbit[dataset_name[i]] = \
                    COSSpectrum(dataset_name[i], good_pixel_limits)
                self.orbit[dataset_name[i]].compute_proper_error()
            elif instrument == 'stis':
                raise NotImplementedError('STIS instrument not implemented '
                                          'yet.')

    # Plot all the spectra in a wavelength range
    def plot_spectra(self, wavelength_range, uncertainties=False,
                     figure_sizes=(9.0, 6.5), axes_font_size=18,
                     legend_font_size=13):
        """
        Method used to plot all the spectra in the visit. It is necessary to use
        ``matplotlib.pyplot.plot()`` after running this method to visualize the
        plot.

        Args:

            wavelength_range (array-like): Wavelength limits to be plotted,
                with shape (2, ).

            uncertainties (``bool``, optional): If ``True``, then plot the
                spectra with their respective uncertainties. Default is
                ``False``.

            figure_sizes (array-like, optional): Sizes of the x- and y-axes of
                the plot. Default values are 9.0 for the x-axis and 6.5 for the
                y-axis.

            axes_font_size (``int``, optional): Font size of the axes marks.
                Default value is 18.

            legend_font_size (``int``, optional): Font size of the legend.
                Default value is 13.
        """
        pylab.rcParams['figure.figsize'] = figure_sizes[0], figure_sizes[1]
        pylab.rcParams['font.size'] = axes_font_size

        for i in self.orbit:
            # Use the start time of observation as label
            label = self.orbit[i].start_JD.iso
            # Find which side of the chip corresponds to the wavelength range
            ind = tools.pick_side(self.orbit[i].wavelength, wavelength_range)
            # Now find which spectrum indexes correspond to the requested
            # wavelength
            min_wl = tools.nearest_index(self.orbit[i].wavelength[ind],
                                         wavelength_range[0])
            max_wl = tools.nearest_index(self.orbit[i].wavelength[ind],
                                         wavelength_range[1])
            if uncertainties is False:
                plt.plot(self.orbit[i].wavelength[ind][min_wl:max_wl],
                         self.orbit[i].flux[ind][min_wl:max_wl],
                         label=label)
            else:
                plt.errorbar(self.orbit[i].wavelength[ind][min_wl:max_wl],
                             self.orbit[i].flux[ind][min_wl:max_wl],
                             yerr=self.orbit[i].error[ind][min_wl:max_wl],
                             fmt='.', label=label)
        plt.xlabel(r'Wavelength ($\mathrm{\AA}$)')
        plt.ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
        plt.legend(fontsize=legend_font_size)


# The general ultraviolet spectrum object
class UVSpectrum(object):
    """
    HST ultraviolet spectrum object, used as a container for the data obtained
    in one HST UV exposure.

    Args:

        dataset_name (``str``): Name of the dataset, as downloaded from MAST.
            For example, if the 1-d extracted spectrum file is named
            ``'foo_x1d.fits'``, then the dataset name is ``'foo'``.

        good_pixel_limits (``tuple``, optional): Tuple containing the good pixel
            limits of the detector, with shape (2, 2), where the first line is
            the limits for the red chip, and the second line is for the blue
            chip. If ``None``, use all pixels. Default is ``None``.

        units (``dict``, optional): Python dictionary containing the units of
            the spectrum. It must contain the units for the indexes
            ``'wavelength'``, ``'flux'`` and ``'exp_time'``. If ``None``, then
            the units will be set to angstrom, erg/s/cm**2/angstrom and s for
            the wavelength, flux and exposure time. Default is ``None``.
    """
    def __init__(self, dataset_name, good_pixel_limits=None, units=None):
        self.dataset_name = dataset_name
        self.x1d = dataset_name + '_x1d.fits'
        self.corrtag = dataset_name + '_corrtag_a.fits'
        self.gpl = good_pixel_limits

        if units is None:
            self.units = {'wavelength': u.angstrom,
                          'flux': u.erg / u.s / u.cm ** 2 / u.angstrom,
                          'exp_time': u.s}
        else:
            self.units = units

        # Read data from x1d file
        with fits.open(self.x1d) as f:
            self.header = f[0].header
            self.data = f['SCI'].data

        # Read some metadata from the corrtag file
        with fits.open(self.corrtag) as f:
            self.start_JD = Time(f[3].header['EXPSTRTJ'], format='jd')
            self.end_JD = Time(f[3].header['EXPENDJ'], format='jd')

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
        self.exp_time = self.data['EXPTIME'][0]


# COS spectrum class
class COSSpectrum(UVSpectrum):
    """
    HST/COS ultraviolet spectrum object, used as a container for the data
    obtained in one HST/COS UV exposure.

    Args:

        dataset_name (``str``): Name of the dataset, as downloaded from MAST.
            For example, if the 1-d extracted spectrum file is named
            ``'foo_x1d.fits'``, then the dataset name is ``'foo'``.

        good_pixel_limits (``tuple``, optional): Tuple containing the good pixel
            limits of the detector, with shape (2, 2), where the first line is
            the limits for the red chip, and the second line is for the blue
            chip. If ``None``, use all pixels. Default is
            ``((1260, 15170), (1025, 15020))``.
    """
    def __init__(self, dataset_name,
                 good_pixel_limits=((1260, 15170), (1025, 15020))):
        super(COSSpectrum, self).__init__(dataset_name, good_pixel_limits)

        # Instantiating useful global variables
        self.sensitivity = None

    # Compute the correct errors for the HST/COS observation
    def compute_proper_error(self):
        """
        Compute the proper uncertainties of the HST/COS spectrum, following the
        method proposed by Wilson+ 2017 (ADS code = 2017A&A...599A..75W).
        """
        self.sensitivity = self.flux / self.net / self.exp_time
        self.error = (self.gross_counts + 1.0) ** 0.5 * self.sensitivity

    # Compute the integrated flux in a given wavelength range
    # TODO: Offer the option to integrate between doppler shifts from line
    # center
    def integrated_flux(self, wavelength_range,
                        uncertainty_method='quadratic_sum'):
        """
        Compute the integrated flux of the COS spectrum in a user-defined
        wavelength range.

        Args:

            wavelength_range (array-like): Lower and upper bounds of the
                wavelength limits.

            uncertainty_method (``str``, optional): Method to compute the
                uncertainties of the integrated flux. The options currently
                available are  ``'quadratic_sum'`` and ``'bootstrap'``. Default
                is ``'quadratic_sum'``.

        Returns:

            int_flux (``float``): Value of the integrated flux.

            uncertainty (``float``): Value of the uncertainty of the integrated
                flux.
        """
        ind = tools.pick_side(self.wavelength, wavelength_range)

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

    # Plot the spectrum
    def plot_spectrum(self, wavelength_range=None, chip_index=None,
                      plot_uncertainties=False):
        """
        Plot the spectrum, with the option of selecting a specific wavelength
        range or the red or blue chips of the detector. In order to visualize
        the plot, it is necessary to run the command
        ``matplotlib.pyplot.plot()`` after running this method.

        Args:

            wavelength_range (array-like, optional): Lower and upper bounds of
                the wavelength limits. If ``None``, then ``chip_index`` must be
                provided. Default value is ``None``.

            chip_index (``str`` or ``int``, optional): Choose 0 for the red
                chip, 1 for the blue chip, or use the strings ``'red'`` or
                ``'blue'``. If ``None``, then ``wavelength_range`` must be
                provided. Default is ``None``.

            plot_uncertainties (``bool``, optional): If set to ``True``, than
                the spectrum is plotted with uncertainty bars. Default is
                ``False``.
        """

        if wavelength_range is not None:
            ind = tools.pick_side(self.wavelength, wavelength_range)
            min_wl = tools.nearest_index(self.wavelength[ind],
                                         wavelength_range[0])
            max_wl = tools.nearest_index(self.wavelength[ind],
                                         wavelength_range[1])

            # Finally plot it
            if plot_uncertainties is False:
                plt.plot(self.wavelength[ind][min_wl:max_wl],
                         self.flux[ind][min_wl:max_wl],
                         label=self.start_JD.value)
            else:
                plt.errorbar(self.wavelength[ind][min_wl:max_wl],
                             self.flux[ind][min_wl:max_wl],
                             yerr=self.error[ind][min_wl:max_wl],
                             fmt='.',
                             label=self.start_JD.value)
            plt.xlabel(r'Wavelength ($\mathrm{\AA}$)')
            plt.ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')

        elif chip_index is not None:
            if chip_index == 'red':
                chip_index = 0
            elif chip_index == 'blue':
                chip_index = 1
            if plot_uncertainties is False:
                plt.plot(self.wavelength[chip_index],
                         self.flux[chip_index],
                         label=self.start_JD.value)
            else:
                plt.errorbar(self.wavelength[chip_index],
                             self.flux[chip_index],
                             yerr=self.error[chip_index],
                             fmt='.',
                             label=self.start_JD.value)
            plt.xlabel(r'Wavelength ($\mathrm{\AA}$)')
            plt.ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')

        else:
            raise ValueError('Either the wavelength range or chip index must'
                             'be provided.')


# STIS spectrum class
class STISSpectrum(object):
    def __init__(self):
        raise NotImplementedError('This feature is not implemented yet.')
