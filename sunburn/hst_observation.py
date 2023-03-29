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
import astropy.uncertainty as a_unc
import os
import glob
import emcee


from warnings import warn
from astropy.io import fits
from astropy.time import Time
from astropy.stats import poisson_conf_interval
from . import tools, spectroscopy
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import binned_statistic
from costools import splittag, x1dcorr
from stistools import inttag
from calcos.x1d import concatenateSegments
from stissplice import splicer

__all__ = ["Visit", "UVSpectrum", "COSSpectrum", "STISSpectrum",
           "CombinedSpectrum", "SpectralLine", "ContaminatedLine",
           "AirglowTemplate"]


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
                 prefix=None, compute_proper_error=True, flux_debias=None,
                 echelle_mode=False):

        self.orbit = {}
        self.split = {}
        self.instrument = instrument
        self.coadd_flux = None
        self.coadd_f_unc = None
        self.coadd_time = None
        self.coadd_t_span = None
        self.n_orbit = len(dataset_name)
        self.prefix = prefix
        self.dataset_names = dataset_name

        for i in range(len(dataset_name)):
            # Instantiate the orbit classes
            if instrument == 'cos':
                self.orbit[dataset_name[i]] = \
                    COSSpectrum(dataset_name[i], good_pixel_limits,
                                prefix=prefix)
                if compute_proper_error is True:
                    self.orbit[dataset_name[i]].compute_proper_error()
                else:
                    pass
                if isinstance(flux_debias, float):
                    self.orbit[dataset_name[i]].flux *= flux_debias
                elif isinstance(flux_debias, np.ndarray):
                    self.orbit[dataset_name[i]].flux *= flux_debias[i]
                else:
                    pass
            elif instrument == 'stis':
                self.orbit[dataset_name[i]] = \
                    STISSpectrum(dataset_name[i], prefix=prefix,
                                 echelle_mode=echelle_mode)

    # Plot all the spectra in a wavelength range
    def plot_spectra(self, wavelength_range=None, velocity_range=None,
                     ref_wl=None, chip_index=None,  uncertainties=False,
                     figure_sizes=(9.0, 6.5), doppler_shift_corr=0.0,
                     velocity_space=False, axes_font_size=18, legend=False,
                     legend_font_size=13, rotate_x_ticks=30, labels=None,
                     return_spectrum=False, **mpl_kwargs):
        """
        Method used to plot all the spectra in the visit.

        Args:

            wavelength_range (array-like): Wavelength limits to be plotted,
                with shape (2, ).

            chip_index():

            ref_wl (``float``, optional): Reference wavelength used to plot the
                spectra in Doppler velocity space.

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

        x_axis_return = []
        y_axis_return = []
        y_err_return = []

        for i in self.orbit:
            if isinstance(labels, str):
                label = labels
            elif isinstance(labels, list):
                label = labels[i]
            else:
                # Use the start time of observation as label
                label = self.orbit[i].start_JD.iso

            # Compute the wavelength shift correction
            if ref_wl is not None:
                wl_shift = ref_wl * doppler_shift_corr / c.c.to(u.km / u.s).value
            else:
                wl_shift = 0.0

            # Use either the wavelength range or the chip_index
            if wavelength_range is None and velocity_range is None:
                k = chip_index
                try:
                    wavelength_range = [min(self.orbit[i].wavelength[k]) + 1,
                                        max(self.orbit[i].wavelength[k]) - 1]
                except TypeError:
                    raise ValueError('Either the wavelength range or the chip'
                                     'index have to be provided.')
            else:
                pass

            if velocity_range is not None:
                vi = velocity_range[0]
                vf = velocity_range[1]
                ls = c.c.to(u.km / u.s).value
                wavelength_range = (vi / ls * ref_wl + ref_wl,
                                    vf / ls * ref_wl + ref_wl)

            # Find which side of the chip corresponds to the wavelength range
            # (only for COS)
            if self.instrument == 'cos':
                ind = tools.pick_side(self.orbit[i].wavelength, wavelength_range)
                wavelength = self.orbit[i].wavelength[ind]
                flux = self.orbit[i].flux[ind]
                f_unc = self.orbit[i].error[ind]
            # In the case of STIS, there is no need to pick a chip
            else:
                wavelength = self.orbit[i].wavelength
                flux = self.orbit[i].flux
                f_unc = self.orbit[i].error

            # Now find which spectrum indexes correspond to the requested
            # wavelength
            min_wl = tools.nearest_index(wavelength,
                                         wavelength_range[0] - wl_shift)
            max_wl = tools.nearest_index(wavelength,
                                         wavelength_range[1] - wl_shift)

            if velocity_space is True:
                x_axis = \
                    (wavelength[min_wl:max_wl] + wl_shift -
                     ref_wl) / ref_wl * c.c.to(u.km / u.s).value
                x_label = r'Velocity (km s$^{-1}$)'
            else:
                x_axis = wavelength[min_wl:max_wl] + wl_shift
                x_label = r'Wavelength ($\mathrm{\AA}$)'

            # Store the data in a list to return if necessary
            x_axis_return.append(x_axis)
            y_axis_return.append(flux[min_wl:max_wl])
            y_err_return.append(f_unc[min_wl:max_wl])

            if uncertainties is False:
                plt.plot(x_axis, flux[min_wl:max_wl],  label=label,
                         **mpl_kwargs)
            else:
                plt.errorbar(x_axis, flux[min_wl:max_wl],
                             yerr=f_unc[min_wl:max_wl],
                             fmt='.', label=label, **mpl_kwargs)
        plt.xlabel(x_label)
        plt.ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')
        if legend is True:
            plt.legend(fontsize=legend_font_size)
        if rotate_x_ticks is not None:
            plt.xticks(rotation=rotate_x_ticks)
            plt.tight_layout()

        if return_spectrum is True:
            return x_axis_return, y_axis_return, y_err_return

    # Time-tag split the observations in the visit
    def time_tag_split(self, n_splits, out_dir,  path_calibration_files=None,
                       stis_high_res=None):
        """

        Args:
            n_splits:
            path_calibration_files:
            out_dir:

        Returns:

        """
        for dataset in self.orbit:
            self.orbit[dataset].time_tag_split(
                n_splits, out_dir=out_dir,
                path_calibration_files=path_calibration_files,
                highres=stis_high_res)
            if self.instrument == 'cos':
                for i in range(n_splits):
                    self.orbit[dataset].split[i].compute_proper_error()
            else:
                pass

    # Assign previously computed splits to the visit
    def assign_splits(self, path):
        """

        Args:
            path:
        """
        for dataset in self.orbit:
            self.orbit[dataset].assign_splits(path)
            n_splits = len(self.orbit[dataset].split)
            if self.instrument == 'cos':
                for i in range(n_splits):
                    self.orbit[dataset].split[i].compute_proper_error()
            else:
                pass


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

        # Instantiating global variables that are not instrument specific
        self.instrument = None
        self.dataset_name = dataset_name
        self.x1d = dataset_name + '_x1d.fits'
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
            self.data = f['SCI'].data

        # If ``good_pixel_limits`` is set to ``None``, then the data will be
        # retrieved from the file in its entirety. Otherwise, it will be
        # retrieved using the limits established by ``good_pixel_limits``
        if self.gpl is None:
            self.gpl = ((0, -1), (0, -1))
        else:
            pass

        # Instantiating the instrument-specific global variables
        self.jit = None
        self.header = None
        self.optical_element = None
        self.aperture = None
        self.start_JD = None
        self.end_JD = None
        self.exp_time = None
        self.wavelength = None
        self.flux = None
        self.net = None
        self.gross_counts = None
        self.error = None
        self.quality = None
        self.slit_orientation = None

    # Compute the integrated flux in a given wavelength range
    def integrated_flux(self, wavelength_range=None, velocity_range=None,
                        reference_wl=None, rv_correction=0.0,
                        uncertainty_method='quadratic_sum',
                        integrate_choice='flux'):
        """
        Compute the integrated flux of the spectrum in a user-defined
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
        ls = c.c.to(u.km / u.s)
        if wavelength_range is not None:
            if self.instrument == 'cos':
                ind = tools.pick_side(self.wavelength, wavelength_range)
                wavelength = self.wavelength[ind]
                flux = self.flux[ind]
                net = self.net[ind]
                gross = self.gross_counts[ind]
                f_unc = self.error[ind]
            else:
                wavelength = self.wavelength
                flux = self.flux
                net = self.net
                gross = self.gross_counts
                f_unc = self.error
        else:
            assert reference_wl is not None and velocity_range is not None, \
                   'Reference wavelength and RV range must be provided ' \
                   'if you did not pick a wavelength range.'
            velocity_range += rv_correction
            wavelength_range = velocity_range * reference_wl / ls + reference_wl
            if self.instrument == 'cos':
                ind = tools.pick_side(self.wavelength, wavelength_range)
                wavelength = self.wavelength[ind]
                flux = self.flux[ind]
                net = self.net[ind]
                gross = self.gross_counts[ind]
                f_unc = self.error[ind]
            else:
                wavelength = self.wavelength
                flux = self.flux
                net = self.net
                gross = self.gross_counts
                f_unc = self.error
        min_wl = tools.nearest_index(wavelength, wavelength_range[0])
        max_wl = tools.nearest_index(wavelength, wavelength_range[1]) + 1

        # The following line is hacky, but it works
        delta_wl = wavelength[1:] - wavelength[:-1]
        if integrate_choice == 'flux':
            int_flux = simps(flux[min_wl:max_wl], x=wavelength[min_wl:max_wl])
        elif integrate_choice == 'counts':
            int_flux = np.sum(gross[min_wl:max_wl])
            uncertainty_method = 'poisson'
        elif integrate_choice == 'net':
            int_flux = np.sum(net[min_wl:max_wl])
            uncertainty_method = 'poisson'
        else:
            raise ValueError('This integration choice is not implemented.')

        # Compute the uncertainty of the integrated flux
        if uncertainty_method == 'quadratic_sum':
            uncertainty = np.sqrt(np.sum((delta_wl[min_wl:max_wl] *
                                          f_unc[min_wl:max_wl]) ** 2))
        elif uncertainty_method == 'poisson':
            sensitivity = flux[min_wl:max_wl] / net[min_wl:max_wl]
            mean_sensitivity = np.nanmean(sensitivity * delta_wl[min_wl:max_wl])
            int_gross = np.sum(gross[min_wl:max_wl])
            gross_unc = poisson_conf_interval(int(int_gross),
                                              interval='root-n') - int_gross
            if integrate_choice == 'flux':
                uncertainty = (-gross_unc[0] + gross_unc[1]) / 2 * \
                    mean_sensitivity / self.exp_time
            elif integrate_choice == 'counts':
                uncertainty = (-gross_unc[0] + gross_unc[1]) / 2
            elif integrate_choice == 'net':
                uncertainty = (-gross_unc[0] + gross_unc[1]) / 2 / self.exp_time
            else:
                raise ValueError('This integration choice is not implemented.')
        elif uncertainty_method == 'bootstrap':
            n_samples = 10000
            # Draw a sample of spectra and compute the fluxes for each
            samples = np.random.normal(loc=flux[min_wl:max_wl],
                                       scale=f_unc[min_wl:max_wl],
                                       size=[n_samples, max_wl - min_wl])
            fluxes = []
            for i in range(n_samples):
                fluxes.append(simps(samples[i],
                                    x=wavelength[min_wl:max_wl]))
            fluxes = np.array(fluxes)
            uncertainty = np.std(fluxes)
        else:
            raise ValueError('This value of ``uncertainty_method`` is not '
                             'accepted.')

        return int_flux, uncertainty

    # Get the spectrum data in a specific range
    def get_spectrum(self, wavelength_range=None, velocity_range=None,
                     ref_wl=None):
        """

        Args:
            wavelength_range:
            velocity_range:
            ref_wl:

        Returns:

        """
        if wavelength_range is None and velocity_range is not None and ref_wl \
                is not None:
            velocity_range = np.array(velocity_range)
            wavelength_range = \
                velocity_range / c.c.to(u.km / u.s).value * ref_wl + ref_wl
        else:
            raise ValueError('Either wavelength range or velocity range and '
                             'ref_wl have to be provided.')

        if self.instrument == 'cos':
            ind = tools.pick_side(self.wavelength, wavelength_range)
            wavelength = self.wavelength[ind]
            flux = self.flux[ind]
            f_unc = self.error[ind]
        else:
            wavelength = self.wavelength
            flux = self.flux
            f_unc = self.error

        min_wl = tools.nearest_index(wavelength, wavelength_range[0])
        max_wl = tools.nearest_index(wavelength, wavelength_range[1])
        if ref_wl is not None:
            velocity = (wavelength[min_wl:max_wl] - ref_wl) / ref_wl * \
                       c.c.to(u.km / u.s).value
            return wavelength[min_wl:max_wl], velocity, flux[min_wl:max_wl], \
                f_unc[min_wl:max_wl]
        else:
            return wavelength[min_wl:max_wl], flux[min_wl:max_wl], \
                   f_unc[min_wl:max_wl]

    # Plot the spectrum
    def plot_spectrum(self, wavelength_range=None, velocity_range=None,
                      chip_index=None, plot_uncertainties=False,
                      rotate_x_ticks=30, ref_wl=None, flag=None, scale=None,
                      rv_shift=0.0, **kwargs):
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

            ref_wl (``float``, optional): Reference wavelength used to plot the
                spectra in Doppler velocity space.
        """
        ax = plt.subplot()

        if scale is None:
            scale = 1
            y_label = r'Flux density (erg s$^{-1}$ cm$^{-2}$ ' \
                      r'$\mathrm{\AA}^{-1}$)'
        elif isinstance(scale, float):
            pow_i = np.log10(scale)
            y_label = \
                r'Flux density (10$^{%i}$ erg s$^{-1}$ cm$^{-2}$ ' \
                r'$\mathrm{\AA}^{-1}$)' % pow_i
        else:
            raise ValueError('``scale`` must be ``float`` or ``None``.')

        if wavelength_range is None and velocity_range is not None:
            velocity_range = np.array(velocity_range)
            wavelength_range = \
                velocity_range / c.c.to(u.km / u.s).value * ref_wl + ref_wl
        else:
            pass

        if wavelength_range is not None:
            if self.instrument == 'cos':
                ind = tools.pick_side(self.wavelength, wavelength_range)
                wavelength = self.wavelength[ind]
                flux = self.flux[ind] / scale
                f_unc = self.error[ind] / scale
                quality = self.quality[ind]
            else:
                wavelength = self.wavelength
                flux = self.flux / scale
                f_unc = self.error / scale
                quality = self.quality

            min_wl = tools.nearest_index(wavelength, wavelength_range[0])
            max_wl = tools.nearest_index(wavelength, wavelength_range[1])

            if isinstance(ref_wl, float):
                x_axis = c.c.to(u.km / u.s).value * \
                    (wavelength[min_wl:max_wl] - ref_wl) / ref_wl
                x_axis += rv_shift
                x_label = r'Velocity (km s$^{-1}$)'
            else:
                x_axis = wavelength[min_wl:max_wl]
                wl_shift = rv_shift / c.c.to(u.km / u.s).value * x_axis
                x_axis += wl_shift
                x_label = r'Wavelength ($\mathrm{\AA}$)'

            # Finally plot it
            if plot_uncertainties is False:
                ax.plot(x_axis, flux[min_wl:max_wl], **kwargs)
            else:
                ax.errorbar(x_axis, flux[min_wl:max_wl],
                             yerr=f_unc[min_wl:max_wl],
                             fmt='.', **kwargs)

            # Overplot a span where there is a specific flag
            if flag is not None:
                dq_inds = np.where(quality[min_wl:max_wl] == flag)[0]
                for i in dq_inds:
                    ax.axvline(x=x_axis[i], color='k', alpha=0.1)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        elif chip_index is not None:
            if chip_index == 'red':
                chip_index = 0
            elif chip_index == 'blue':
                chip_index = 1
            if plot_uncertainties is False:
                ax.plot(self.wavelength[chip_index],
                         self.flux[chip_index], **kwargs)
            else:
                ax.errorbar(self.wavelength[chip_index],
                             self.flux[chip_index],
                             yerr=self.error[chip_index],
                             fmt='.', **kwargs)
            ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
            ax.set_ylabel(y_label)

        else:
            raise ValueError('Either the wavelength range or chip index must'
                             'be provided.')

        #if rotate_x_ticks is not None:
        #    ax.set_xticklabels(rotation=rotate_x_ticks)
            #plt.tight_layout()
        return ax

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
            if self.instrument == 'cos':
                ind = tools.pick_side(self.wavelength, line.wavelength_range)
                wavelength = self.wavelength[ind]
                flux = self.flux[ind]
                f_unc = self.error[ind]
            else:
                wavelength = self.wavelength
                flux = self.flux
                f_unc = self.error

            min_wl = tools.nearest_index(wavelength, line.wavelength_range[0])
            max_wl = tools.nearest_index(wavelength, line.wavelength_range[1])
            doppler_v = \
                (wavelength[min_wl:max_wl] - line.central_wavelength)\
                / line.central_wavelength * light_speed
            flux = flux[min_wl:max_wl]
            unc = f_unc[min_wl:max_wl]

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

    # Extract the wavelength array from a given range
    def extract_wl(self, wavelength_range):
        """

        Args:
            wavelength_range:

        Returns:

        """
        ind = tools.pick_side(self.wavelength, wavelength_range)
        min_wl = tools.nearest_index(self.wavelength[ind], wavelength_range[0])
        max_wl = tools.nearest_index(self.wavelength[ind], wavelength_range[1])
        wl_array = self.wavelength[ind][min_wl:max_wl + 1]
        return wl_array


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
                 good_pixel_limits=((1260, 15170), (1025, 15020)), prefix=None,
                 subexposure=False):
        super(COSSpectrum, self).__init__(dataset_name, good_pixel_limits,
                                          prefix=prefix)

        self.instrument = 'cos'
        # COS-specific variables
        self.corrtag_a = dataset_name + '_corrtag_a.fits'
        self.corrtag_b = dataset_name + '_corrtag_b.fits'
        with fits.open(self.prefix + self.x1d) as f:
            self.header = f[0].header
            self.optical_element = f[0].header['OPT_ELEM']
            self.aperture = f[0].header['APERTURE']

        # Read some metadata from the corrtag file
        with fits.open(self.prefix + self.corrtag_a) as f:
            self.start_JD = Time(f[3].header['EXPSTRTJ'], format='jd')
            self.end_JD = Time(f[3].header['EXPENDJ'], format='jd')

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
        self.quality = np.array([self.data['DQ'][0][i00:i01],
                                 self.data['DQ'][1][i10:i11]])
        self.visit_id = self.header['ASN_TAB'][:9]

        # Appending some important jitter information
        self.jit = self.visit_id + '_jit.fits'

        # Read the jitter information (only valid for a full exposure and not
        # subexposures)
        if subexposure is False:
            try:
                with fits.open(self.prefix + self.jit) as f:
                    # We need to figure out which index inside the jit file
                    # corresponds to this orbit
                    orbits_list = np.array([f[k + 1].header['EXPNAME'][:8]
                                            for k in range(len(f) - 1)])
                    ind = np.where(orbits_list == self.dataset_name[:-1])[0][0]
                    self.jitter_data = f[ind + 1].data
                    self.jitter_columns = self.jitter_data.columns
                    self._jp = []
                    for i in range(len(self.jitter_columns)):
                        self._jp.append(self.jitter_columns[i].name)
                    self._jp = np.array(self._jp)
            except OSError:
                warn('Could not find the jitter file (%s_jit.fits).'
                     % self.dataset_name)
                self.jitter_data = None
                self.jitter_columns = None

        if subexposure is False and self.jitter_data is not None:
            latitude = self.jitter_data['Latitude']
            longitude = self.jitter_data['Longitude']
            sin_lat = np.sin(latitude * u.deg)
            sin_long = np.sin(longitude * u.deg)
            sin_lat_col = fits.Column('sin_Latitude', format='E', array=sin_lat)
            sin_long_col = fits.Column('sin_Longitude', format='E',
                                       array=sin_long)
            self.jitter_columns = self.jitter_columns + \
                sin_lat_col + sin_long_col
            self.jitter_data = fits.FITS_rec.from_columns(self.jitter_columns)

        # Instantiating useful global variables
        self.sensitivity = None
        self.split = None
        self._systematics = None
        self.ccf = None

    # Compute the correct errors for the HST/COS observation
    def compute_proper_error(self, shift_net=1E-7):
        """
        Compute the proper uncertainties of the HST/COS spectrum, following the
        method proposed by Wilson+ 2017 (ADS code = 2017A&A...599A..75W).
        """
        self.sensitivity = self.flux / (self.net + shift_net) / self.exp_time
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
            out_dir = out_dir
            for item in split_list:
                x1dcorr.x1dcorr(item, outdir=out_dir)

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
                split_obs = COSSpectrum(dataset_name, prefix=out_dir,
                                        subexposure=True)
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
            offset = len(path)
            dataset_name = split_list[i][offset:offset + 11]
            split_obs = COSSpectrum(dataset_name, prefix=path, subexposure=True)
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

    def verify_systematic(self, line_list, plot=True, normalize=False,
                          return_norm=False, fold=True, rv_corr=None,
                          rv_range=None, **kwargs):
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
                if rv_corr is not None:
                    rv_shift = rv_corr[species][i] * u.km / u.s
                else:
                    rv_shift = 0.0
                if rv_range is not None:
                    ref_wl = line_list[species][i].central_wavelength
                    for split in self.split:
                        f, unc = split.integrated_flux(reference_wl=ref_wl,
                                                       rv_range=rv_range,
                                                       rv_correction=rv_shift)
                        flux.append(f)
                        f_unc.append(unc)
                else:
                    wl_range = line_list[species][i].wavelength_range
                    # For each split in the observation
                    for split in self.split:
                        f, unc = split.integrated_flux(
                            wavelength_range=wl_range)
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
        self._systematics['f_unc'] = total_unc

        # Plot the computed fluxes
        if plot is True:
            x_shift = (self.end_JD.jd + self.start_JD.jd) / 2
            norm = np.mean(total_flux)
            t_hour = ((time - x_shift) * u.d).to(u.min).value
            if fold is True:
                x_axis = t_hour
            else:
                x_axis = ((time - time[0]) * u.d).to(u.min).value
            if normalize is True:
                plt.errorbar(x_axis, total_flux / norm,
                             xerr=(t_span * u.d).to(u.min).value,
                             yerr=total_unc / norm, fmt='o', **kwargs)
                plt.ylabel('Normalized sum of integrated fluxes')
            else:
                plt.errorbar(x_axis, total_flux,
                             xerr=(t_span * u.d).to(u.min).value,
                             yerr=total_unc, fmt='o', **kwargs)
                plt.ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$)')
            plt.xlabel('Time (min)')

        else:
            norm = np.mean(total_flux)

        if return_norm is True:
            return norm
        else:
            pass

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
    def __init__(self, dataset_name, prefix=None, subexposure=False,
                 echelle_mode=False):
        super(STISSpectrum, self).__init__(dataset_name, prefix=prefix)
        self.instrument = 'stis'
        self.echelle_mode = echelle_mode

        # STIS-specific variables
        with fits.open(self.prefix + self.x1d) as f:
            self.header = f[1].header
            self.optical_element = f[0].header['OPT_ELEM']
            self.aperture = f[0].header['APERTURE']

        bjd_shift = 2400000.5
        self.start_JD = Time(self.header['EXPSTART'] + bjd_shift,
                             format='jd')
        self.end_JD = Time(self.header['EXPEND'] + bjd_shift,
                           format='jd')
        if self.echelle_mode is False:
            self.wavelength = self.data['WAVELENGTH'][0]
            self.flux = self.data['FLUX'][0]
            self.error = self.data['ERROR'][0]
            self.exp_time = self.header['EXPTIME']
            self.gross_counts = self.data['GROSS'][0] * self.exp_time
            self.background = self.data['BACKGROUND'][0]
            self.net = self.data['NET'][0]
        else:
            spec_table = splicer.splice_pipeline(dataset_name, prefix)
            self.wavelength = spec_table['WAVELENGTH'].data
            self.flux = spec_table['FLUX'].data
            self.error = spec_table['ERROR'].data
            self.exp_time = self.header['EXPTIME']
            self.gross_counts = self.data['GROSS'] * self.exp_time
            self.background = self.data['BACKGROUND']
            self.net = self.data['NET']
        self.slit_orientation = 45.35 * u.deg  # Angle between the slit and the
        # reference HST angle V3 (~U3)
        self.split = None

        # Appending some important jitter information
        self.jit = dataset_name + '_jit.fits'
        # Read the jitter information (only valid for a full exposure and not
        # subexposures)
        if subexposure is False:
            try:
                with fits.open(self.prefix + self.jit) as f:
                    self.jitter_data = f[1].data
                    self.jitter_columns = self.jitter_data.columns
                    self._jp = []
                    for i in range(len(self.jitter_columns)):
                        self._jp.append(self.jitter_columns[i].name)
                    self._jp = np.array(self._jp)
            except OSError:
                warn('Could not find the jitter file (%s_jit.fits).'
                     % self.dataset_name)
                self.jitter_data = None
                self.jitter_columns = None

        if subexposure is False and self.jitter_data is not None:
            theta = self.slit_orientation
            v3_roll = self.jitter_data['V3_roll']
            v2_roll = self.jitter_data['V2_roll']
            v3_dom = self.jitter_data['V3_dom']
            v2_dom = self.jitter_data['V2_dom']
            latitude = self.jitter_data['Latitude']
            longitude = self.jitter_data['Longitude']
            sin_lat = np.sin(latitude * u.deg)
            sin_long = np.sin(longitude * u.deg)
            vd_roll = v3_roll * np.cos(theta) + v2_roll * np.sin(theta)
            vs_roll = -v3_roll * np.sin(theta) + v2_roll * np.cos(theta)
            vd_dom = v3_dom * np.cos(theta) + v2_dom * np.sin(theta)
            vs_dom = -v3_dom * np.sin(theta) + v2_dom * np.cos(theta)
            vd_roll_col = fits.Column('Vd_roll', format='E', unit='degrees',
                                      array=vd_roll)
            vs_roll_col = fits.Column('Vs_roll', format='E', unit='degrees',
                                      array=vs_roll)
            vd_dom_col = fits.Column('Vd_dom', format='E', unit='degrees',
                                      array=vd_dom)
            vs_dom_col = fits.Column('Vs_dom', format='E', unit='degrees',
                                      array=vs_dom)
            sin_lat_col = fits.Column('sin_Latitude', format='E', array=sin_lat)
            sin_long_col = fits.Column('sin_Longitude', format='E',
                                       array=sin_long)
            self.jitter_columns = self.jitter_columns + vd_roll_col + \
                vs_roll_col + vd_dom_col + vs_dom_col + sin_lat_col + \
                sin_long_col
            self.jitter_data = fits.FITS_rec.from_columns(self.jitter_columns)

    # Time tag split the observation
    def time_tag_split(self, n_splits=None, time_bins=None, out_dir="",
                       highres=False, all_events=False,
                       process_raw=True,
                       path_calibration_files=None,
                       clean_intermediate_steps=True):
        """
        HST calibration files can be downloaded from here:
        https://hst-crds.stsci.edu

        Args:
            n_splits:
            out_dir:
            process_raw:
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
        for i in range(len(time_bins) - 1):
            start_time = time_bins[i]
            increment = time_bins[i + 1] - start_time
            inttag.inttag(
                tagfile=self.prefix + self.dataset_name + '_tag.fits',
                output=out_dir + self.dataset_name + '_%s_raw.fits' % str(i),
                starttime=start_time, increment=increment, highres=highres,
                allevents=all_events)

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
            offset = len(path)
            dataset_name = split_list[i][offset:offset + 11]
            split_obs = STISSpectrum(dataset_name, prefix=path,
                                     subexposure=True,
                                     echelle_mode=self.echelle_mode)
            self.split.append(split_obs)


# The combined visit class
class CombinedSpectrum(object):
    """

    """
    def __init__(self, orbit_list, reference_wavelength, instrument,
                 wavelength_range=None, velocity_range=None, doppler_corr=None,
                 velocity_grid=None, cleaned_spectra=False,
                 final_uncertainty='combine', final_flux='average'):
        self._orbits = orbit_list
        self._n_orbit = len(orbit_list)
        self._ref_wl = reference_wavelength
        self.wavelength = None
        self.velocity = None
        self.flux = None
        self.gross = None
        self.net = None
        self.f_unc = None
        self.test = None
        self.total_exp_time = 0.0
        self.binned_wavelength = None
        self.binned_velocity = None
        self.binned_flux = None
        self.binned_f_unc = None
        self.binned_net = None
        self.binned_gross = None

        if doppler_corr is None:
            doppler_corr = [0.0 for i in range(self._n_orbit)]

        ls = c.c.to(u.km / u.s).value
        # if self._ref_wl is not None:
        wl_shift = [self._ref_wl * dck / ls for dck in doppler_corr]
        # else:
        #     wl_shift = [self._ref_wl * dck / ls for dck in doppler_corr]

        wavelength_list = []
        velocity_list = []
        flux_list = []
        f_unc_list = []
        gross_list = []
        net_list = []

        # Use either the wavelength range or the velocity range
        if wavelength_range is None and velocity_range is None:
            raise ValueError(
                'Either the wavelength or velocity range has to be provided.')
        elif wavelength_range is None:
            vi = velocity_range[0]
            vf = velocity_range[1]
            wavelength_range = (vi / ls * self._ref_wl + self._ref_wl,
                                vf / ls * self._ref_wl + self._ref_wl)
        else:
            pass

        for i in range(self._n_orbit):
            # Find which side of the chip corresponds to the wavelength range
            # (only for COS)
            self.total_exp_time += self._orbits[i].exp_time
            if instrument == 'cos':
                ind = tools.pick_side(self._orbits[i].wavelength,
                                      wavelength_range)
                wavelength = self._orbits[i].wavelength[ind]
                if cleaned_spectra is True:
                    flux = self._orbits[i].clean_flux[ind]
                    f_unc = self._orbits[i].clean_f_unc[ind]
                    gross = None
                    net = None
                else:
                    flux = self._orbits[i].flux[ind]
                    f_unc = self._orbits[i].error[ind]
                    gross = self._orbits[i].gross_counts[ind]
                    net = self._orbits[i].net[ind]
            # In the case of STIS, there is no need to pick a chip
            else:
                wavelength = self._orbits[i].wavelength
                flux = self._orbits[i].flux
                f_unc = self._orbits[i].error
                gross = self._orbits[i].gross_counts
                net = self._orbits[i].net

            # Now find which spectrum indexes correspond to the requested
            # wavelength
            min_wl = tools.nearest_index(wavelength + wl_shift[i],
                                         wavelength_range[0])
            max_wl = tools.nearest_index(wavelength + wl_shift[i],
                                         wavelength_range[1])

            # Finally add the spectrum to the list
            velocity = (wavelength[min_wl:max_wl] - self._ref_wl) / \
                self._ref_wl * ls + doppler_corr[i]
            velocity_list.append(velocity)
            wavelength_list.append(wavelength[min_wl:max_wl] + wl_shift[i])
            flux_list.append(flux[min_wl:max_wl])
            f_unc_list.append(f_unc[min_wl:max_wl])
            gross_list.append(gross[min_wl:max_wl])
            net_list.append(net[min_wl:max_wl])

        # Finally combine the spectra
        # If the velocity grid was not specified, just use the one from the
        # first orbit provided
        if velocity_grid is None:
            velocity_grid = velocity_list[0]

        self.velocity = velocity_grid
        for i in range(self._n_orbit):
            fw = interp1d(velocity_list[i], wavelength_list[i], kind='linear',
                          bounds_error=False, fill_value='extrapolate')
            ff = interp1d(velocity_list[i], flux_list[i], kind='linear',
                          bounds_error=False, fill_value=1E-18)
            fu = interp1d(velocity_list[i], f_unc_list[i], kind='linear',
                          bounds_error=False, fill_value=1E-18)
            fg = interp1d(velocity_list[i], gross_list[i], kind='linear',
                          bounds_error=False, fill_value=1E-18)
            fn = interp1d(velocity_list[i], net_list[i], kind='linear',
                          bounds_error=False, fill_value=1E-18)
            velocity_list[i] = velocity_grid
            wavelength_list[i] = fw(velocity_grid)
            flux_list[i] = ff(velocity_grid)
            f_unc_list[i] = fu(velocity_grid)
            gross_list[i] = fg(velocity_grid)
            net_list[i] = fn(velocity_grid)
        self.wavelength = wavelength_list[0]
        self.test = np.array(flux_list)

        # Calculating the final flux. There are two options: 1) Regular average,
        # or 2) Weighted average.
        if final_flux == 'average' or final_flux == 'mean':
            self.flux = np.mean(np.array(flux_list), axis=0)
            self.net = np.mean(np.array(net_list), axis=0)
        elif final_flux == 'weighted':
            self.flux = np.sum(np.array(flux_list) * np.array(f_unc_list),
                               axis=0) / np.sum(np.array(f_unc_list), axis=0)
            self.net = np.sum(np.array(net_list) * np.array(f_unc_list),
                               axis=0) / np.sum(np.array(f_unc_list), axis=0)

        # Calculating the final gross counts and "sensitivity"
        self.gross = np.sum(np.array(gross_list), axis=0)

        # Calculating uncertainties. There are three options: 1) Simply combining
        # the tabulated uncertainties at face value, 2) Calculating the standard
        # deviation of the measured fluxes, or 3) Drawing samples and
        # calculating the percentiles of the sample.
        if final_uncertainty == 'combine':
            self.f_unc = (np.sum(np.array(f_unc_list) ** 2, axis=0) /
                          self._n_orbit) ** 0.5
        elif final_uncertainty == 'poisson':
            combined_gross = np.zeros_like(self.wavelength)
            sensitivity = np.nanmean(
                np.array([flux_list[i] / net_list[i]
                         for i in range(self._n_orbit)]), axis=0)
            for i in range(self._n_orbit):
                combined_gross += gross_list[i]
            combined_gross_unc = []
            for k in range(len(self.wavelength)):
                if combined_gross[k] < 0:
                    combined_gross[k] *= (-1)
                sample = a_unc.poisson(combined_gross[k], 1000)
                combined_gross_unc.append(sample.pdf_std())
            combined_gross_unc = np.array(combined_gross_unc)
            self.f_unc = combined_gross_unc / self.total_exp_time * sensitivity
        elif final_uncertainty == 'stdev':
            self.f_unc = np.std(np.array(flux_list), axis=0)
        elif final_uncertainty == 'sample':
            # For each wavelength bin and each exposure, we draw a random sample
            # of 500 measurements with mu = flux and sigma = uncertainty in a
            # particular exposure.
            self.f_unc = np.zeros_like(self.wavelength)
            for i in range(len(self.wavelength)):
                sample_all = []
                for k in range(len(f_unc_list)):
                    sample_all.append(np.random.normal(loc=self.flux[i],
                                      scale=f_unc_list[k][i], size=500))
                sample_all = np.array(sample_all)
                result = np.percentile(sample_all, [16, 50, 84])
                q = np.diff(result)
                self.f_unc[i] += (q[0] + q[1]) / 2
        else:
            raise ValueError('This options of `uncertainty` is not valid. Use '
                             'either `"combine"` or `"sample"`.')

    # Bin the spectrum
    def bin_spectrum(self, velocity_bin_width=5, uncertainty_regime='gaussian'):
        """

        Args:
            velocity_bin_width:
            uncertainty_regime:

        Returns:

        """
        gross = self.gross
        net = self.net
        ds = self.velocity
        wv = self.wavelength
        f = self.flux
        fu = self.f_unc
        bw = velocity_bin_width

        self.binned_wavelength, self.binned_velocity, \
            self.binned_flux, self.binned_f_unc = tools.bin_spectrum(
                bw, wv, ds, f, fu)

        if uncertainty_regime == 'poisson':
            v_bins = np.arange(min(ds), max(ds) + bw, bw)
            dwl = np.concatenate(
                (wv[1:] - wv[:-1], np.array([wv[-1] - wv[-2], ])))
            sensitivity = (self.flux / self.net) * dwl
            binned_sensitivity, edges, inds = binned_statistic(ds, sensitivity,
                                                            bins=v_bins,
                                                            statistic='mean')
            self.binned_gross, edges, inds = binned_statistic(ds, gross,
                                                              bins=v_bins,
                                                              statistic='sum')
            wv_bin = self.binned_wavelength
            dwl_bin = np.concatenate(
                (wv_bin[1:] - wv_bin[:-1], np.array([wv_bin[-1] -
                                                     wv_bin[-2], ])))
            limits = poisson_conf_interval(self.binned_gross)
            limits[0] -= self.binned_gross
            limits[1] -= self.binned_gross
            mean_sigma = (-limits[0] + limits[1]) / 2
            self.binned_f_unc = mean_sigma / self.total_exp_time * \
                binned_sensitivity / dwl_bin
        else:
            pass

    # Plot the combined spectrum
    def plot_spectrum(self, wavelength_range=None,
                      uncertainties=False, figure_sizes=(9.0, 6.5),
                      axes_font_size=18, legend_font_size=13, rotate_x_ticks=30,
                      label=None, barplot=False, velocity_space=False,
                      scale=None, **kwargs):
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
            barplot:
            velocity_space:
            scale:
            **kwargs:

        Returns:

        """
        pylab.rcParams['figure.figsize'] = figure_sizes[0], figure_sizes[1]
        pylab.rcParams['font.size'] = axes_font_size

        line_center = self._ref_wl

        if wavelength_range is None:
            min_wl = 0
            max_wl = -1
        else:
            # Now find which spectrum indexes correspond to the requested
            # wavelength
            min_wl = tools.nearest_index(self.wavelength, wavelength_range[0])
            max_wl = tools.nearest_index(self.wavelength, wavelength_range[1])

        # Figure out the x- and y-axes values
        if velocity_space is False:
            x_values = self.wavelength[min_wl:max_wl]
            x_label = r'Wavelength ($\mathrm{\AA}$)'
        else:
            ls = c.c.to(u.km / u.s).value
            x_values = (self.wavelength[min_wl:max_wl] - line_center) / \
                line_center * ls
            x_label = r'Velocity (km s$^{-1}$)'
        delta_x = x_values[1] - x_values[0]

        if scale is None:
            scale = 1
            y_label = r'Flux density (erg s$^{-1}$ cm$^{-2}$ ' \
                      r'$\mathrm{\AA}^{-1}$)'
        elif isinstance(scale, float):
            pow_i = np.log10(scale)
            y_label = \
                r'Flux density (10$^{%i}$ erg s$^{-1}$ cm$^{-2}$ ' \
                r'$\mathrm{\AA}^{-1}$)' % pow_i
        else:
            raise ValueError('``scale`` must be ``float``.')

        y_values = self.flux[min_wl:max_wl] / scale
        y_err = self.f_unc[min_wl:max_wl] / scale

        # Finally plot it
        if uncertainties is False:
            if barplot is False:
                plt.plot(x_values, y_values, label=label, **kwargs)
            else:
                plt.bar(x_values, y_values, label=label, width=delta_x,
                        **kwargs)
        else:
            if barplot is False:
                plt.errorbar(x_values, y_values, yerr=y_err, fmt='.',
                             label=label, **kwargs)
            else:
                plt.bar(x_values, y_values, yerr=y_err, label=label,
                        width=delta_x, **kwargs)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(fontsize=legend_font_size)
        if rotate_x_ticks is not None:
            plt.xticks(rotation=rotate_x_ticks)
            plt.tight_layout()

    # Compute the integrated flux of the spectrum in a given wavelength range
    def integrate_flux(self, wavelength_range,
                       uncertainty_method='quadratic_sum'):
        """

        Args:
            wavelength_range:

        Returns:

        """

        min_wl = tools.nearest_index(self.wavelength, wavelength_range[0])
        max_wl = tools.nearest_index(self.wavelength, wavelength_range[1])
        # The following line is hacky, but it works
        delta_wl = self.wavelength[1:] - self.wavelength[:-1]
        int_flux = simps(self.flux[min_wl:max_wl],
                         x=self.wavelength[min_wl:max_wl])

        # Calculating uncertainty
        if uncertainty_method == 'quadratic_sum':
            uncertainty = np.sqrt(np.sum((delta_wl[min_wl:max_wl] *
                                          self.f_unc[min_wl:max_wl]) ** 2))
        elif uncertainty_method == 'poisson':
            delta_wl = np.concatenate(
                (self.wavelength[1:] - self.wavelength[:-1],
                 np.array([self.wavelength[-1] - self.wavelength[-2], ])))
            sensitivity = self.flux[min_wl:max_wl] / self.net[min_wl:max_wl]
            mean_sensitivity = np.nanmean(sensitivity * delta_wl[min_wl:max_wl])
            int_gross = np.sum(self.gross[min_wl:max_wl])
            gross_unc = poisson_conf_interval(int(int_gross),
                                              interval='root-n') - int_gross
            uncertainty = (-gross_unc[0] + gross_unc[1]) / 2 * \
                mean_sensitivity / self.total_exp_time
        else:
            raise ValueError('This uncertainty method is not implemented.')

        return int_flux, uncertainty


# The airglow template class
class AirglowTemplate(object):
    """

    """
    def __init__(self, wavelength, flux, uncertainties=None,
                 reference_wavelength=None):
        self.wavelength = wavelength
        self.flux = flux
        self.f_unc = uncertainties
        self._ls = c.c.to(u.km / u.s).value
        if reference_wavelength is not None:
            self.ref_wl = reference_wavelength
        else:
            self.ref_wl = np.mean(self.wavelength)
        self.velocity = (self.wavelength - self.ref_wl) * self._ls/ self.ref_wl

        # Other useful global variables
        self._ls = c.c.to(u.km / u.s).value  # Light speed in km / s

    # Apply Doppler shift to the airglow spectrum
    def adjust_spectrum(self, doppler_shift=0.0 * u.km / u.s, scale_flux=1.0,
                        interpolation_type='linear', fill_value=0.0,
                        update_spectrum=False):
        """

        Args:
            doppler_shift:
            scale_flux:
            interpolation_type:
            fill_value:
            update_spectrum:

        Returns:

        """
        new_flux, new_f_unc = tools.doppler_shift(doppler_shift, self.ref_wl,
                                                  self.wavelength, self.flux,
                                                  self.f_unc,
                                                  interpolation_type,
                                                  fill_value)
        new_flux *= scale_flux
        new_f_unc *= scale_flux
        if update_spectrum is False:
            return new_flux, new_f_unc
        else:
            self.flux = np.copy(new_flux)
            self.f_unc = np.copy(new_f_unc)

    # Interpolate the template to a specific wavelengths array
    def interpolate_to(self, wavelength, interpolation_type='linear'):
        """

        Args:
            wavelength:
            interpolation_type:

        Returns:

        """
        x = self.wavelength
        y1 = self.flux
        y2 = self.f_unc
        f1 = interp1d(x, y1, kind=interpolation_type, fill_value='extrapolate')
        f2 = interp1d(x, y2, kind=interpolation_type, fill_value='extrapolate')
        new_flux = f1(wavelength)
        new_f_unc = f2(wavelength)
        return new_flux, new_f_unc

    # Plot the airglow template
    def plot(self, wavelength_range=None, velocity_range=None,
             uncertainties=False, figure_sizes=(9.0, 6.5), axes_font_size=18,
             rotate_x_ticks=0, **kwargs):
        """

        Args:
            wavelength_range:
            velocity_range:
            uncertainties:
            figure_sizes:
            axes_font_size:
            rotate_x_ticks:

        Returns:

        """
        pylab.rcParams['figure.figsize'] = figure_sizes[0], figure_sizes[1]
        pylab.rcParams['font.size'] = axes_font_size

        # Plot either in wavelength- or velocity-space
        if wavelength_range is not None:
            min_wl = tools.nearest_index(self.wavelength, wavelength_range[0])
            max_wl = tools.nearest_index(self.wavelength, wavelength_range[1])
            x_axis = self.wavelength[min_wl:max_wl]
            x_label = r'Wavelength ($\mathrm{\AA}$)'
        elif velocity_range is not None:
            vr = velocity_range
            wavelength_range = [vr[0] / self._ls * self.ref_wl + self.ref_wl,
                                vr[1] / self._ls * self.ref_wl + self.ref_wl]
            min_wl = tools.nearest_index(self.wavelength, wavelength_range[0])
            max_wl = tools.nearest_index(self.wavelength, wavelength_range[1])
            x_axis = (self.wavelength[min_wl:max_wl] - self.ref_wl) / \
                self.ref_wl * self._ls
            x_label = r'Velocity (km s$^{-1}$)'
        else:
            raise ValueError('Either wavelength range or velocity range has to '
                             'be provided.')

        if uncertainties is False:
            plt.plot(x_axis, self.flux[min_wl:max_wl], **kwargs)
        else:
            plt.errorbar(x_axis, self.flux[min_wl:max_wl],
                         yerr=self.f_unc[min_wl:max_wl], fmt='.', **kwargs)
        plt.xlabel(x_label)
        plt.ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$)')

        if rotate_x_ticks is not None:
            plt.xticks(rotation=rotate_x_ticks)
            plt.tight_layout()


# The general observed line profile class
class SpectralLine(object):
    """

    """
    def __init__(self, cos_observation, central_wavelength,
                 doppler_shift_range=(-100 * u.km / u.s, 100 * u.km / u.s)):

        # Check if the passed parameters are of correct type
        if isinstance(cos_observation, COSSpectrum) is True:
            cos_observation = [cos_observation]
        elif isinstance(cos_observation, list) is True:
            pass
        else:
            raise ValueError('`cos_observation` must be a `COSSpectrum` object '
                             'or a list of `COSSpectrum` objects.')

        self.n_spectra = len(cos_observation)

        if isinstance(central_wavelength, u.Quantity) is True:
            self.w0 = central_wavelength.to(u.angstrom).value
        else:
            self.w0 = central_wavelength

        # Figure out the wavelength range
        self.l_speed = c.c.to(u.km / u.s).value
        try:
            self.ds_range = (doppler_shift_range[0].to(u.km / u.s).value,
                             doppler_shift_range[1].to(u.km / u.s).value)
        except AttributeError:
            self.ds_range = (doppler_shift_range[0], doppler_shift_range[1])
        self.wl_range = (self.ds_range[0] / self.l_speed * self.w0 + self.w0,
                         self.ds_range[1] / self.l_speed * self.w0 + self.w0)

        # Extract the data from the spectrum, for each observation in the list
        # of COS spectra
        ind = tools.pick_side(cos_observation[0].wavelength, self.wl_range)
        min_wl = tools.nearest_index(cos_observation[0].wavelength[ind],
                                     self.wl_range[0])
        max_wl = tools.nearest_index(cos_observation[0].wavelength[ind],
                                     self.wl_range[1])

        self.wavelength = []
        self.flux = []
        self.f_unc = []
        self.velocity = []
        self.time = []
        self.start_JD = []
        self.end_JD = []
        for ck in cos_observation:
            self.wavelength.append(ck.wavelength[ind][min_wl:max_wl])
            self.flux.append(ck.flux[ind][min_wl:max_wl])
            self.f_unc.append(ck.error[ind][min_wl:max_wl])
            self.velocity.append((self.wavelength[-1] - self.w0) *
                                 self.l_speed / self.w0)
            # Obtain other info that can be useful
            self.start_JD.append(ck.start_JD.jd)
            self.end_JD.append(ck.end_JD.jd)
            self.time.append((self.start_JD[-1] + self.end_JD[-1]) / 2)

    # Apply a Doppler shift to the spectra
    def doppler_shift(self, velocity, interpolation_type='linear',
                      fill_value=0.0):
        """

        Args:
            velocity:
            interpolation_type:
            fill_value:

        Returns:

        """
        for i in range(self.n_spectra):
            new_flux, new_f_unc = tools.doppler_shift(velocity, self.w0,
                                                      self.wavelength[i],
                                                      self.flux[i],
                                                      self.f_unc[i],
                                                      interpolation_type,
                                                      fill_value)
            self.flux[i] = np.copy(new_flux)
            self.f_unc[i] = np.copy(new_f_unc)

    # Plot the lines
    def plot(self, velocity_space=True, x_range=None, select_exposures=None,
             uncertainties=False, **kwargs):
        """

        Returns:

        """
        if select_exposures is None:
            select_exposures = range(self.n_spectra)
        else:
            pass

        if velocity_space is True:
            for i in select_exposures:
                if uncertainties is False:
                    plt.plot(self.velocity[i], self.flux[i], **kwargs)
                else:
                    plt.errorbar(self.velocity[i], self.flux[i],
                                 yerr=self.f_unc[i], fmt='.', **kwargs)
            plt.xlabel(r'Velocity (km s$^{-1}$)')
            if x_range is None:
                x_range = self.ds_range
            else:
                pass
        else:
            for i in select_exposures:
                if uncertainties is False:
                    plt.plot(self.wavelength[i], self.flux[i], **kwargs)
                else:
                    plt.errorbar(self.wavelength[i], self.flux[i],
                                 yerr=self.f_unc[i], fmt='.', **kwargs)
            plt.xlabel(r'Wavelength ($\mathrm{\AA}$)')
            plt.xticks(rotation=30)
            if x_range is None:
                x_range = self.wl_range
            else:
                pass
        plt.xlim(x_range)
        plt.ylabel(r'Flux (erg s$^{-1}$ $\mathrm{\AA}^{-1}$ cm$^{-2}$)')

    # Integrate the flux of the lines between a range of velocities
    def integrated_flux(self, velocity_range=(-100, 100)):
        """

        Args:
            velocity_range:

        Returns:

        """
        int_flux = []
        uncertainty = []

        for i in range(self.n_spectra):
            min_v = tools.nearest_index(self.velocity[i], velocity_range[0])
            max_v = tools.nearest_index(self.velocity[i], velocity_range[1])
            delta_wl = self.wavelength[i][1:] - self.wavelength[i][:-1]
            int_flux.append(simps(self.flux[i][min_v:max_v],
                             self.wavelength[i][min_v:max_v]))
            uncertainty.append(np.sqrt(np.sum((delta_wl[min_v:max_v] *
                                          self.f_unc[i][min_v:max_v]) ** 2)))

        return int_flux, uncertainty


# The Lyman-alpha profile class
class ContaminatedLine(SpectralLine):
    """

    """
    def __init__(self, cos_observation, airglow_template, central_wavelength,
                 doppler_shift_range=(-300 * u.km / u.s, 300 * u.km / u.s)):
        super(ContaminatedLine,
              self).__init__(cos_observation, central_wavelength,
                             doppler_shift_range=doppler_shift_range)

        # Check if the passed parameters are of correct type
        if isinstance(airglow_template, AirglowTemplate) is False:
            raise ValueError('`airglow_template` must be an `AirglowTemplate` '
                             'object.')
        else:
            self.ag_template = airglow_template

        # Start some useful global variables
        self.fit_result = None
        self.mcmc_sample = None
        self.clean_flux = []
        self.clean_f_unc = []
        self.clean_flux_sample = None
        self.clean_f_unc_sample = None

    # Fit airglow template to observed line within a specific range of the
    # spectrum
    def fit_template(self, velocity_range, shift_guess, scales_guess,
                     fixed_shift=False, shift_bounds=(None, None),
                     scale_bounds=(None, None), fill_value=1E-18,
                     perform_mcmc=False, n_walkers=10, n_steps=500,
                     maxiter_minimize=500, neg_flux_severity=100):
        """

        Args:
            velocity_range:
            shift_guess:
            scales_guess:
            fixed_shift:
            shift_bounds:
            scale_bounds:
            fill_value:
            perform_mcmc:
            n_walkers:
            n_steps:
            maxiter_minimize:
            neg_flux_severity:

        Returns:

        """
        # Find the indexes of the velocity_range
        min_v = []
        max_v = []
        if isinstance(velocity_range, list):
            n_pass = 1
            min_v.append(
                tools.nearest_index(self.velocity[0], velocity_range[0]))
            max_v.append(
                tools.nearest_index(self.velocity[0], velocity_range[1]))
        elif isinstance(velocity_range, np.ndarray):
            n_pass = np.shape(velocity_range)[0]
            for i in range(n_pass):
                min_v.append(
                    tools.nearest_index(self.velocity[0], velocity_range[i, 0]))
                max_v.append(
                    tools.nearest_index(self.velocity[0], velocity_range[i, 1]))
        else:
            raise TypeError('Wrong format for ``velocity_range``.')

        # The badness of the fit function
        def _lnlike(params):
            if fixed_shift is True:
                params[0] = shift_guess
            else:
                pass
            badness = []
            # For each observation...
            for i in range(self.n_spectra):

                # Compute the fluxes based on shift and scale
                templ_flux, templ_error = \
                    self.ag_template.adjust_spectrum(params[0],
                                                     params[i + 1],
                                                     fill_value=fill_value)

                # Create temporary template
                temporary_ag = AirglowTemplate(self.ag_template.wavelength,
                                               templ_flux, templ_error, self.w0)
                # Interpolate to the wavelengths of the observation
                interp_flux, interp_error = \
                    temporary_ag.interpolate_to(self.wavelength[i])
                # Compute the difference between template and observed spectrum
                diff = (self.flux[i] - interp_flux) * 1E13
                u_diff = (self.f_unc[i] ** 2 + templ_error ** 2) ** 0.5
                weight = 1 / (1E13 * u_diff) ** 2

                temp_badness = 0
                # Add the badness of the core fit
                for k in range(n_pass):
                    lnlike = -0.5 * (np.sum(diff[min_v[k]:max_v[k]] ** 2 *
                                            weight[min_v[k]:max_v[k]] -
                                            np.log(weight[min_v[k]:max_v[k]])))
                    temp_badness += lnlike
                    # Also punish the airglow model if the resulting cleaned
                    # spectra has negative flux
                    f_clip = np.clip(diff, a_min=None, a_max=0)
                    int_f_clip = simps(f_clip, self.wavelength[i])
                    int_u_clean = simps(u_diff * 1E13, self.wavelength[i])
                    punish = neg_flux_severity * int_f_clip / int_u_clean
                    badness.append(temp_badness + punish)

            badness = np.array(badness)
            return np.sum(badness)

        # Flat prior
        def _lnprior(params):
            if shift_bounds[0] < params[0] < shift_bounds[1]:
                badness = 0
                for i in range(self.n_spectra):
                    if scale_bounds[0] < params[i + 1] < scale_bounds[1]:
                        pass
                    else:
                        badness = -np.inf
                        break
                return badness
            else:
                return -np.inf

        # Probability function
        def _lnprob(params):
            lp = _lnprior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + _lnlike(params)

        # Perform the minimization of the badness of fit function
        guess = np.array([shift_guess] + scales_guess)
        bounds = [list(shift_bounds)]
        # Need to add bounds for each of the scale parameters, but all of them
        # are `None`
        for i in range(self.n_spectra):
            bounds.append(list(scale_bounds))
        nll = lambda *args: -_lnlike(*args)
        self.fit_result = minimize(nll, x0=guess, method='TNC', bounds=bounds,
                                   options={'maxiter': maxiter_minimize})

        if perform_mcmc is False:
            return self.fit_result
        else:
            ndim = 1 + self.n_spectra
            pos = [self.fit_result["x"] + 1e-4*np.random.randn(ndim)
                   for i in range(n_walkers)]
            sampler = emcee.EnsembleSampler(n_walkers, ndim, _lnprob)
            sampler.run_mcmc(pos, n_steps, progress=True)
            self.mcmc_sample = sampler.chain[:, int(n_steps / 10):, :].\
                reshape((-1, ndim))
            return self.fit_result, self.mcmc_sample

    # Base method to correct the contaminated spectrum
    def _clean_spectrum(self, params, fill_value=1E-18):
        """

        Args:
            params:
            fill_value:

        Returns:

        """
        clean_flux = []
        clean_f_unc = []

        # Compute the flux and uncertainties of the best fit clean line
        for i in range(self.n_spectra):
            bf_flux, bf_error = \
                self.ag_template.adjust_spectrum(params[0],
                                                 params[i + 1],
                                                 fill_value=fill_value)
            bf_templ = AirglowTemplate(self.ag_template.wavelength, bf_flux,
                                       bf_error, self.w0)
            bf_flux, bf_error = bf_templ.interpolate_to(self.wavelength[i])
            clean_flux.append(self.flux[i] - bf_flux)
            clean_f_unc.append((self.f_unc[i] ** 2 + bf_error ** 2) ** 0.5)

        return np.array(clean_flux), np.array(clean_f_unc)

    # Clean the contaminated spectra with the best solution
    def clean(self, fit_result=None, fill_value=1E-18):
        """

        Args:
            fit_result:
            fill_value:

        Returns:

        """
        if fit_result is None:
            params = self.fit_result['x']
        else:
            params = fit_result['x']

        self.clean_flux, self.clean_f_unc = self._clean_spectrum(params,
                                                                 fill_value)

    # Correct the contaminated spectra using the MCMC results
    def clean_mcmc_sample(self, mcmc_sample=None, fill_value=1E-18,
                          n_limit=100):
        """

        Args:
            mcmc_sample:
            fill_value:
            n_limit: 

        Returns:

        """
        if mcmc_sample is None:
            samples = self.mcmc_sample
        else:
            samples = mcmc_sample

        # Setup the number of samples to correct
        if len(samples) > n_limit:
            n_samples = n_limit
        else:
            n_samples = len(samples)

        clean_sample = np.array([self._clean_spectrum(samples[i], fill_value)
                                 for i in range(n_samples)])
        self.clean_flux_sample = clean_sample[:, 0, :, :]
        self.clean_f_unc_sample = clean_sample[:, 1, :, :]

    # Plot the clean spectrum
    def plot_clean(self, velocity_space=True, x_range=None,
                   select_exposures=None, uncertainties=False,
                   scale_flux=1E-13, velocity_bin_width=None, y_shift=0.0,
                   **kwargs):
        """

        Args:
            velocity_space:
            x_range:
            select_exposures:
            uncertainties:
            scale_flux:
            velocity_bin_width:
            y_shift:
            **kwargs:

        Returns:

        """
        if scale_flux is not None:
            f_scale = 1.0 / scale_flux
            log_scale = int(np.log10(scale_flux))
            ylabel = r'Flux density (10$^{%i}$ erg s$^{-1}$ $\mathrm{\AA}^{-1}$ cm$^{-2}$)' % log_scale
        else:
            f_scale = 1.0
            ylabel = r'Flux density (erg s$^{-1}$ $\mathrm{\AA}^{-1}$ cm$^{-2}$)'

        if select_exposures is None:
            select_exposures = range(self.n_spectra)
        else:
            pass

        # No binning
        if velocity_bin_width is None:
            v = self.velocity
            wv = self.wavelength
            cf = [cfk + y_shift / f_scale for cfk in self.clean_flux]
            cferr = self.clean_f_unc
        # Bin the spectrum
        else:
            vbw = velocity_bin_width
            binned_data = np.array(
                [tools.bin_spectrum(vbw, self.wavelength[i], self.velocity[i],
                                    self.clean_flux[i], self.clean_f_unc[i])
                 for i in range(self.n_spectra)])
            wv = binned_data[:, 0]
            v = binned_data[:, 1]
            cf = binned_data[:, 2] + y_shift / f_scale
            cferr = binned_data[:, 3]

        if velocity_space is True:
            for i in select_exposures:
                if uncertainties is False:
                    plt.plot(v[i], cf[i] * f_scale, **kwargs)
                else:
                    plt.errorbar(v[i], cf[i] * f_scale,
                                 yerr=cferr[i] * f_scale, **kwargs)
            plt.xlabel(r'Velocity (km s$^{-1}$)')
            if x_range is None:
                x_range = self.ds_range
            else:
                pass
        else:
            for i in select_exposures:
                if uncertainties is False:
                    plt.plot(wv[i], cf[i] * f_scale, **kwargs)
                else:
                    plt.errorbar(wv[i], cf[i] * f_scale,
                                 yerr=cferr[i] * f_scale, **kwargs)
            plt.xlabel(r'Wavelength ($\mathrm{\AA}$)')
            plt.xticks(rotation=30)
            if x_range is None:
                x_range = self.wl_range
            else:
                pass
        plt.xlim(x_range)
        plt.ylabel(ylabel)

    # Compute the integrated flux in the clean spectrum
    def integrated_clean_flux(self, velocity_range=(-100, 100)):
        """

        Args:
            velocity_range:

        Returns:

        """
        min_v = tools.nearest_index(self.velocity[0], velocity_range[0])
        max_v = tools.nearest_index(self.velocity[0], velocity_range[1])

        int_flux = []
        uncertainty = []

        for i in range(self.n_spectra):
            delta_wl = self.wavelength[i][1:] - self.wavelength[i][:-1]
            int_flux.append(simps(self.clean_flux[i][min_v:max_v],
                                  self.wavelength[i][min_v:max_v]))
            uncertainty.append(
                np.sqrt(np.sum((delta_wl[min_v:max_v] *
                                self.clean_f_unc[i][min_v:max_v]) ** 2)))

            # If clean spectra from the MCMC were calculated, then incorporate
            # them in the uncertainties
            if self.clean_flux_sample is not None:
                n_sample = len(self.clean_flux_sample)
                int_flux_sample = np.array([simps(
                    self.clean_flux_sample[k, i, min_v:max_v],
                    self.wavelength[i][min_v:max_v]) for k in range(n_sample)])
                add_unc = np.std(int_flux_sample)
                uncertainty[-1] = (uncertainty[-1] ** 2 + add_unc ** 2) ** 0.5

        return np.array(int_flux), np.array(uncertainty)
