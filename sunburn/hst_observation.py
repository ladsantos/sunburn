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
import astropy.constants as c
import os
import glob
from astropy.io import fits
from astropy.time import Time
from . import tools, spectroscopy
from scipy.integrate import simps
from costools import splittag, x1dcorr
from calcos.x1d import concatenateSegments

__all__ = ["Visit", "UVSpectrum", "COSSpectrum", "STISSpectrum"]


# HST visit
class Visit(object):
    """
    HST visit object. It is used as a container for a collection of HST
    observational data from a single visit.

    Args:

        dataset_name (``list``): List of names of the datasets, as downloaded
            from MAST. For example, if the 1-d extracted spectrum file is named
            ``'foo_x1d.fits'``, then the dataset name is ``'foo'``.

        instrument (``str``): Instrument name. Currently, the only options
            available are ``'cos'`` and ``'stis'``.

        good_pixel_limits (``tuple``, optional): Tuple containing the good pixel
            limits of the detector, with shape (2, 2), where the first line is
            the limits for the red chip, and the second line is for the blue
            chip. If ``None``, use all pixels. Default is ``None``.
    """
    def __init__(self, dataset_name, instrument, good_pixel_limits=None,
                 prefix=None):

        self.orbit = {}
        self.split = {}
        self.instrument = instrument
        self.coadd_flux = None
        self.coadd_f_unc = None
        self.coadd_time = None
        self.coadd_t_span = None
        self.n_orbit = len(dataset_name)

        for i in range(len(dataset_name)):
            if instrument == 'cos':
                self.orbit[dataset_name[i]] = \
                    COSSpectrum(dataset_name[i], good_pixel_limits,
                                prefix=prefix)
                self.orbit[dataset_name[i]].compute_proper_error()
            elif instrument == 'stis':
                raise NotImplementedError('STIS instrument not implemented '
                                          'yet.')

    # Plot all the spectra in a wavelength range
    def plot_spectra(self, wavelength_range=None, chip_index=None,
                     uncertainties=False, figure_sizes=(9.0, 6.5),
                     axes_font_size=18, legend_font_size=13, rotate_x_ticks=30):
        """
        Method used to plot all the spectra in the visit.

        Args:

            wavelength_range (array-like): Wavelength limits to be plotted,
                with shape (2, ).

            chip_index():

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

            # Use either the wavelength range or the chip_index
            if wavelength_range is None:
                k = chip_index
                try:
                    wavelength_range = [min(self.orbit[i].wavelength[k]) + 1,
                                        max(self.orbit[i].wavelength[k]) - 1]
                except TypeError:
                    raise ValueError('Either the wavelength range or the chip'
                                     'index have to be provided.')
            else:
                pass

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
        if rotate_x_ticks is not None:
            plt.xticks(rotation=rotate_x_ticks)
            plt.tight_layout()

    # Co-add spectra in a visit
    def coadd_spectra(self):
        """

        Returns:
        """

        self.coadd_flux = 0
        self.coadd_f_unc = 0
        #self.coadd_time = 0
        #self.coadd_t_span = 0
        for orbit in self.orbit:
            self.coadd_flux += self.orbit[orbit].flux
            self.coadd_f_unc += self.orbit[orbit].error ** 2
            #self.coadd_time.append(
            #    (self.orbit[orbit].start_JD.value +
            #     self.orbit[orbit].end_JD.value) / 2)
            #self.coadd_t_span.append(
            #    (self.orbit[orbit].end_JD - self.orbit[orbit].start_JD) / 2)
        self.coadd_flux /= self.n_orbit
        self.coadd_f_unc = np.sqrt(self.coadd_f_unc) / self.n_orbit
        #self.coadd_time = np.array(self.coadd_time)

        # Perform the co-addition
        #self.coadd_flux = np.mean(self.coadd_flux, axis=0)
        #self.coadd_f_unc = (np.sum(self.coadd_f_unc ** 2, axis=0)) ** 0.5 / \
        #    self.n_orbit
        #self.coadd_time = np.mean(self.coadd_time, axis=0)
        #self.coadd_time = Time(self.coadd_time, format='jd')


    # Time-tag split the observations in the visit
    def time_tag_split(self, n_splits, path_calibration_files, out_dir):
        """

        Args:
            n_splits:
            path_calibration_files:
            out_dir:

        Returns:

        """
        if self.instrument != 'cos':
            raise ValueError('Time-tag splitting is only available for COS.')
        else:
            pass

        for dataset in self.orbit:
            self.orbit[dataset].time_tag_split(
                n_splits, out_dir=out_dir,
                path_calibration_files=path_calibration_files)
            for i in range(n_splits):
                self.orbit[dataset].split[i].compute_proper_error()

    # Assign previously computed splits to the visit
    def assign_splits(self, path):
        """

        Args:
            path:
        """
        if self.instrument != 'cos':
            raise ValueError('Time-tag splitting is only available for COS.')
        else:
            pass

        for dataset in self.orbit:
            self.orbit[dataset].assign_splits(path)
            n_splits = len(self.orbit[dataset].split)
            for i in range(n_splits):
                self.orbit[dataset].split[i].compute_proper_error()


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
    def __init__(self, dataset_name, good_pixel_limits=None, units=None,
                 prefix=None):
        self.dataset_name = dataset_name
        self.x1d = dataset_name + '_x1d.fits'
        self.corrtag_a = dataset_name + '_corrtag_a.fits'
        self.corrtag_b = dataset_name + '_corrtag_b.fits'
        self.gpl = good_pixel_limits

        if units is None:
            self.units = {'wavelength': u.angstrom,
                          'flux': u.erg / u.s / u.cm ** 2 / u.angstrom,
                          'exp_time': u.s}
        else:
            self.units = units

        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = prefix

        # Read data from x1d file
        with fits.open(self.prefix + self.x1d) as f:
            self.header = f[0].header
            self.data = f['SCI'].data

        # Read some metadata from the corrtag file
        with fits.open(self.prefix + self.corrtag_a) as f:
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
                      plot_uncertainties=False, rotate_x_ticks=30):
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

            rotate_x_ticks ():
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

        if rotate_x_ticks is not None:
            plt.xticks(rotation=rotate_x_ticks)
            plt.tight_layout()

    # Apply wavelength shift
    def wavelength_shift(self, delta_lambda=None, delta_v=None):
        """

        Args:
            delta_lambda:
            delta_v:

        Returns:

        """
        raise NotImplementedError()

    # Plot specific spectral lines
    def plot_lines(self, line, plot_uncertainties=False, rotate_x_ticks=30,
                   coadd=False):
        """

        Args:
            line (`Line` object or list): A `Line` object or a list containing
                the lines to be plotted.

            plot_uncertainties (``bool``, optional): If set to ``True``, than
                the spectrum is plotted with uncertainty bars. Default is
                ``False``.

            rotate_x_ticks (`bool`, optional): Angle to rotate the ticks in the
                x-axis. Default value is 30 degrees.

            coadd (`bool`, optional): Co-add the lines before plotting. Default
                is False.

        Returns:

        """
        # Find the Doppler velocities from line center
        light_speed = c.c.to(u.km / u.s).value
        if isinstance(line, spectroscopy.Line):
            ind = tools.pick_side(self.wavelength, line.wavelength_range)
            min_wl = tools.nearest_index(self.wavelength[ind],
                                         line.wavelength_range[0])
            max_wl = tools.nearest_index(self.wavelength[ind],
                                         line.wavelength_range[1])
            doppler_v = \
                (self.wavelength[ind][min_wl:max_wl] - line.central_wavelength)\
                / line.central_wavelength * light_speed
            flux = self.flux[ind][min_wl:max_wl]
            unc = self.error[ind][min_wl:max_wl]
            print(doppler_v)

        elif isinstance(line, list):
            pass

        # Finally plot it
        if plot_uncertainties is False:
            plt.plot(doppler_v, flux, label=self.start_JD.value)
        else:
            plt.errorbar(doppler_v, flux, yerr=unc, fmt='.',
                         label=self.start_JD.value)
        plt.xlabel(r'Velocity (km s$^{-1}$)')
        plt.ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')


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
                 good_pixel_limits=((1260, 15170), (1025, 15020)), prefix=None):
        super(COSSpectrum, self).__init__(dataset_name, good_pixel_limits,
                                          prefix=prefix)

        # Instantiating useful global variables
        self.sensitivity = None
        self.split = None
        self._systematics = None

    # Compute the correct errors for the HST/COS observation
    def compute_proper_error(self):
        """
        Compute the proper uncertainties of the HST/COS spectrum, following the
        method proposed by Wilson+ 2017 (ADS code = 2017A&A...599A..75W).
        """
        self.sensitivity = self.flux / self.net / self.exp_time
        self.error = (self.gross_counts + 1.0) ** 0.5 * self.sensitivity

    # Time tag split the observation
    def time_tag_split(self, n_splits=None, time_bins=None, out_dir="",
                       auto_extract=True, path_calibration_files=None,
                       clean_intermediate_steps=True):
        """
        HST calibration files can be downloaded from here:
        https://hst-crds.stsci.edu

        Args:
            time_bins:
            n_splits:
            out_dir:
            auto_extract:
            path_calibration_files:
            clean_intermediate_steps:

        Returns:

        """
        # First check if out_dir exists; if not, create it
        if os.path.isdir(out_dir) is False:
            os.mkdir(out_dir)
        else:
            pass

        # Create the time_list string from time_bins if the user specified them,
        # or from the number of splits the user requested
        if isinstance(n_splits, int):
            time_bins = np.linspace(0, self.exp_time, n_splits + 1)
        else:
            pass

        if time_bins is not None:
            time_list = ""
            for time in time_bins:
                time_list += str(time) + ', '

            # Remove the last comma and space from the string
            time_list = time_list[:-2]

            # Add a forward slash to out_dir if it is not there
            if out_dir[-1] != '/':
                out_dir += '/'
            else:
                pass
        else:
            raise ValueError('Either `time_bins` or `n_splits` have to be '
                             'provided.')

        out_dir = self.prefix + out_dir

        # Split-tag the observation
        splittag.splittag(
            infiles=self.prefix + self.dataset_name + '_corrtag_a.fits',
            outroot=out_dir + self.dataset_name, time_list=time_list)
        splittag.splittag(
            infiles=self.prefix + self.dataset_name + '_corrtag_b.fits',
            outroot=out_dir + self.dataset_name, time_list=time_list)

        if auto_extract is True:

            assert isinstance(path_calibration_files, str), \
                'Calibration files path must be provided.'

            # Some hack necessary to avoid IO error when using x1dcorr
            split_list = glob.glob(out_dir + self.dataset_name +
                                   '_?_corrtag_?.fits')
            for item in split_list:
                char_list = list(item)
                char_list.insert(-13, char_list.pop(-6))
                char_list.insert(-12, char_list.pop(-6))
                link = ""
                new_item = link.join(char_list)
                os.rename(item, new_item)

            # Set lref environment variable
            if not 'lref' in os.environ:
                os.environ['lref'] = path_calibration_files

            # Extract the tag-split spectra
            split_list = glob.glob(out_dir + self.dataset_name +
                                   '_?_?_corrtag.fits')
            for item in split_list:
                x1dcorr.x1dcorr(input=item, outdir=out_dir)

            # Clean the intermediate steps files
            if clean_intermediate_steps is True:

                remove_list = glob.glob(out_dir + self.dataset_name +
                                        '*_flt.fits')
                for item in remove_list:
                    os.remove(item)

                remove_list = glob.glob(out_dir + self.dataset_name +
                                        '*_counts.fits')
                for item in remove_list:
                    os.remove(item)

            # Return the filenames back to normal
            split_list = glob.glob(out_dir + self.dataset_name +
                                   '*_corrtag.fits')
            for item in split_list:
                char_list = list(item)
                char_list.insert(-5, char_list.pop(-15))
                char_list.insert(-5, char_list.pop(-15))
                link = ""
                new_item = link.join(char_list)
                os.rename(item, new_item)
            split_list = glob.glob(out_dir + self.dataset_name + '*_x1d.fits')
            for item in split_list:
                char_list = list(item)
                char_list.insert(-5, char_list.pop(-9))
                char_list.insert(-5, char_list.pop(-10))
                link = ""
                new_item = link.join(char_list)
                os.rename(item, new_item)

            # Concatenate segments `a` and `b` of the detector
            for i in range(n_splits):
                x1d_list = glob.glob(out_dir + self.dataset_name +
                                     '_%i_x1d_?.fits' % (i + 1))
                concatenateSegments(x1d_list, out_dir + self.dataset_name +
                                    '_%i' % (i + 1) + '_x1d.fits')

            # Remove more intermediate steps
            if clean_intermediate_steps is True:
                remove_list = glob.glob(out_dir + self.dataset_name +
                                        '_?_x1d_?.fits')
                for item in remove_list:
                    os.remove(item)

            # Finally add each tag-split observation to the `self.split` object
            self.split = []
            time_step = ((self.exp_time / n_splits) * u.s).to(u.d)
            for i in range(n_splits):
                dataset_name = self.dataset_name + '_%i' % (i + 1)
                split_obs = COSSpectrum(dataset_name, prefix=out_dir)
                # Correct the start and end Julian Dates of the split data
                split_obs.start_JD += i * time_step
                split_obs.end_JD -= time_step * (n_splits - i - 1)
                self.split.append(split_obs)

    def assign_splits(self, path):
        """
        If time-tag splits were computed previously, you should use this method
        to assign the resulting split data to a ``COSSpeectrum`` object.

        Args:
            path:
        """
        # Add a trailing forward slash to path if it is not there
        if path[-1] != '/':
            path = path + '/'
        else:
            pass

        # Find the number of splits
        split_list = glob.glob(path + self.dataset_name + '_?_x1d.fits')
        n_splits = len(split_list)

        # Add each tag-split observation to the `self.split` object
        self.split = []
        time_step = ((self.exp_time / n_splits) * u.s).to(u.d)
        for i in range(n_splits):
            dataset_name = self.dataset_name + '_%i' % (i + 1)
            split_obs = COSSpectrum(dataset_name, prefix=path)
            split_obs.start_JD += i * time_step
            split_obs.end_JD -= time_step * (n_splits - i - 1)
            self.split.append(split_obs)

    # Plot the time-tag split spectra
    def plot_splits(self, wavelength_range=None, chip_index=None,
                    plot_uncertainties=False):
        """

        Args:
            wavelength_range:
            chip_index:
            plot_uncertainties:

        Returns:

        """
        for i in range(len(self.split)):
            self.split[i].plot_spectrum(wavelength_range, chip_index,
                                        plot_uncertainties)

    def verify_systematic(self, line_list, plot=True):
        """

        Args:
            line_list:

            plot (``bool``, optional)

        Returns:
            norm (``float``): Mean of the sum of integrated fluxes of the lines
                in the list over the time-tag split data. It is useful to set
                the baseline level of flux to be applied in systematics
                correction.
        """
        self._systematics = {}

        if self.split is None:
            raise ValueError('Can only compute systematics when time-tag '
                             'split data are available.')
        else:
            pass

        flux = []
        f_unc = []

        # For each species in the line list
        for species in line_list:
            n_lines = len(line_list[species])
            # For each spectral line of a species
            for i in range(n_lines):
                wl_range = line_list[species][i].wavelength_range
                # For each split in the observation
                for split in self.split:
                    f, unc = split.integrated_flux(wavelength_range=wl_range)
                    flux.append(f)
                    f_unc.append(unc)

        # Compute times of the observation (this is a repetition of code, should
        # be automated at some point.
        n_splits = len(self.split)
        time = []
        t_span = []
        for i in range(n_splits):
            time.append((self.split[i].start_JD.jd +
                         self.split[i].end_JD.jd) / 2)
            t_span.append((self.split[i].start_JD.jd -
                           self.split[i].end_JD.jd) / 2)
        time = np.array(time)
        t_span = np.array(t_span)
        self._systematics['time'] = time

        # Compute sum of integrated fluxes
        n_lines = len(flux) // n_splits
        flux = np.reshape(np.array(flux), (n_lines, n_splits))
        f_unc = np.reshape(np.array(f_unc), (n_lines, n_splits))
        total_flux = flux.sum(axis=0)
        self._systematics['flux'] = total_flux
        total_unc = ((f_unc ** 2).sum(axis=0)) ** 0.5

        # Plot the computed fluxes
        if plot is True:
            x_shift = (self.end_JD.jd + self.start_JD.jd) / 2
            norm = np.mean(total_flux)
            t_hour = ((time - x_shift) * u.d).to(u.h).value
            plt.errorbar(t_hour, total_flux / norm,
                         xerr=(t_span * u.d).to(u.h).value,
                         yerr=total_unc / norm, fmt='o')
            plt.xlabel('Time (h)')
            plt.ylabel('Normalized sum of integrated fluxes')
        else:
            norm = np.mean(total_flux)

        return norm

    # Systematic correction using a polynomial
    def correct_systematic(self, line_list, baseline_level, poly_deg=1,
                           temp_jd_shift=2.45E6, recompute_errors=False):
        """
        Correct the systematics of a HST/COS orbit by fitting a polynomial to
        the sum of the integrated fluxes of various spectral lines (these lines
        should preferably not have a transiting signal) for a series of time-tag
        split data.

        Args:

            line_list (`COSFUVLineList` object):

            baseline_level ():

            poly_deg (``int``, optional): Degree of the polynomial to be fit.
                Default value is 1.

            temp_jd_shift (``float``, optional): In order to perform a proper
                fit, it is necessary to temporarily modify the Julian Date to a
                smaller number, which is done by subtracting the value of this
                variable from the Julian Dates. Default value is 2.45E6.
        """
        if self._systematics is None:
            temp_norm = self.verify_systematic(line_list, plot=False)

        # Now fit a polynomial
        time = self._systematics['time']
        total_flux = self._systematics['flux']
        norm = baseline_level
        n_splits = len(self.split)
        mod_jd = time - temp_jd_shift
        coeff = np.polyfit(mod_jd, total_flux / norm, deg=poly_deg)
        func = np.poly1d(coeff)
        corr_factor = func(mod_jd)  # Array of correction factors

        # Now we change the spectral flux in each split of this ``COSSpectrum``
        # to take into account the systematics
        for i in range(n_splits):
            self.split[i].flux[0] /= corr_factor[i]
            self.split[i].flux[1] /= corr_factor[i]
            if recompute_errors is True:
                self.split[i].compute_proper_error()

        # Now correct the spectral flux of the ``COSSpectrum`` itself. The flux
        # will be given by the mean of the flux of all splits and the
        # uncertainties by the quadratic sum of those of the splits
        for k in range(2):
            sum_split_flux = []
            for split in self.split:
                sum_split_flux.append(split.flux[k])
            sum_split_flux = np.array(sum_split_flux)
            mean_flux = sum_split_flux.sum(axis=0) / n_splits
            self.flux[k] = mean_flux
            if recompute_errors is True:
                self.compute_proper_error()


# STIS spectrum class
class STISSpectrum(UVSpectrum):
    """
    HST/STIS ultraviolet spectrum object, used as a container for the data
    obtained in one HST/STIS UV exposure.

    Args:

        dataset_name (``str``): Name of the dataset, as downloaded from MAST.
            For example, if the 1-d extracted spectrum file is named
            ``'foo_x1d.fits'``, then the dataset name is ``'foo'``.
    """
    def __init__(self, dataset_name, data_folder=None):
        super(STISSpectrum, self).__init__(dataset_name,
                                           data_folder=data_folder)


# The combined visit class
class CombinedSpectra(object):
    """

    """
    def __init__(self, visit):
        self._orbits = []
        self._n_orbit = len(visit.orbit)
        self.flux = 0
        self.f_unc = 0
        self.start_JD = []
        self.end_JD = []

        for o in visit.orbit:
            self._orbits.append(visit.orbit[o])
            self.flux += visit.orbit[o].flux
            self.f_unc += visit.orbit[o].error ** 2
            self.start_JD.append(visit.orbit[o].start_JD)
            self.end_JD.append(visit.orbit[o].end_JD)

        self.wavelength = visit.orbit[o].wavelength
        self.flux /= self._n_orbit
        self.f_unc = self.f_unc ** 0.5 / self._n_orbit

    # Plot the combined spectrum
    def plot_spectrum(self, wavelength_range=None, chip_index=None,
                      uncertainties=False, figure_sizes=(9.0, 6.5),
                      axes_font_size=18, legend_font_size=13, rotate_x_ticks=30,
                      label=None):
        """

        Args:
            wavelength_range:
            chip_index:
            uncertainties:
            figure_sizes:
            axes_font_size:
            legend_font_size:
            rotate_x_ticks:
            label:

        Returns:

        """
        pylab.rcParams['figure.figsize'] = figure_sizes[0], figure_sizes[1]
        pylab.rcParams['font.size'] = axes_font_size

        # Figure out the wavelength range
        if wavelength_range is None:
            k = chip_index
            try:
                wavelength_range = [min(self.wavelength[k]) + 1,
                                    max(self.wavelength[k]) - 1]
            except TypeError:
                raise ValueError('Either the wavelength range or the chip'
                                 'index have to be provided.')
        else:
            pass

        # Find which side of the chip corresponds to the wavelength range
        ind = tools.pick_side(self.wavelength, wavelength_range)
        # Now find which spectrum indexes correspond to the requested
        # wavelength
        min_wl = tools.nearest_index(self.wavelength[ind], wavelength_range[0])
        max_wl = tools.nearest_index(self.wavelength[ind], wavelength_range[1])
        if uncertainties is False:
            plt.plot(self.wavelength[ind][min_wl:max_wl],
                     self.flux[ind][min_wl:max_wl], label=label)
        else:
            plt.errorbar(self.wavelength[ind][min_wl:max_wl],
                         self.flux[ind][min_wl:max_wl],
                         yerr=self.f_unc[ind][min_wl:max_wl], fmt='.',
                         label=label)
        plt.xlabel(r'Wavelength ($\mathrm{\AA}$)')
        plt.ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
        plt.legend(fontsize=legend_font_size)
        if rotate_x_ticks is not None:
            plt.xticks(rotation=rotate_x_ticks)
            plt.tight_layout()

    # Compute the integrated flux of the spectrum in a given wavelength range
    def integrate_flux(self, wavelength_range):
        """

        Args:
            wavelength_range:

        Returns:

        """
        ind = tools.pick_side(self.wavelength, wavelength_range)

        min_wl = tools.nearest_index(self.wavelength[ind], wavelength_range[0])
        max_wl = tools.nearest_index(self.wavelength[ind], wavelength_range[1])
        # The following line is hacky, but it works
        delta_wl = self.wavelength[ind][1:] - self.wavelength[ind][:-1]
        int_flux = simps(self.flux[ind][min_wl:max_wl],
                         x=self.wavelength[ind][min_wl:max_wl])
        uncertainty = np.sqrt(np.sum((delta_wl[min_wl:max_wl] *
                                      self.f_unc[ind][min_wl:max_wl]) ** 2))
        return int_flux, uncertainty
