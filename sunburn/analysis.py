#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of HST/COS data.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import warnings
from astropy.time import Time
from scipy.stats import binned_statistic, pearsonr
from scipy.optimize import minimize
from astroplan import EclipsingSystem
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.exoplanet_orbit_database import ExoplanetOrbitDatabase


__all__ = ["Transit", "LightCurve"]


class Transit(object):
    """
    The transiting exoplanet object. It is used to predict transit events.

    Args:

        planet_name (``str``, optional): Name of the planet. Default is
            ``None``.

        period (scalar, optional): Orbital period of the planet. If set to
            ``None``, then the value is going to be retrieved from an online
            database or a local table. Default is ``None``.

        transit_midpoint (scalar, optional): Value of a known transit midpoint.
            If set to ``None``, then the value is going to be retrieved from an
            online database or a local table. Default is ``None``.

        eccentricity (``float``, optional): Value of the orbital eccentricity.
            If set to ``None``, then the value is going to be retrieved from an
            online database or a local table. Default is ``None``.

        duration14 (scalar, optional): Value of the transit duration from the
            first to fourth points of contact. If set to ``None``, then the
            value is going to be retrieved from an online database or a local
            table. Default is ``None``.

        duration23 (scalar, optional): Value of the transit duration from the
            second to third points of contact. If set to ``None``, then the
            value  is going to be retrieved from an online database or a local
            table. Default is ``None``.

        database (``str``, optional): Database choice to look up the exoplanet
            parameters. The current options available are ``'nasa'`` and
            ``'orbit'``, which correspond to the NASA Exoplanet Archive and the
            Exoplanet Orbit Database, respectively. Default is ``'nasa'``.
    """
    def __init__(self, planet_name=None, period=None, transit_midpoint=None,
                 eccentricity=None, duration14=None, duration23=None,
                 semi_a=None, inclination=None, longitude_periastron=None,
                 planet_radius=None, stellar_radius=None, database='nasa'):
        self.name = planet_name
        self.period = period
        self.transit_midpoint = transit_midpoint
        self.eccentricity = eccentricity
        self.semi_a = semi_a
        self.stellar_radius = stellar_radius
        self.planet_radius = planet_radius
        self.inclination = inclination
        self.long_periastron = longitude_periastron
        self.duration14 = duration14
        self.duration23 = duration23
        self.database = database

        # If period or transit_center_time are not provided, look up the data
        # from a database
        if self.period is None or self.transit_midpoint is None:
            assert isinstance(self.name, str), '``name`` is required to find ' \
                                               'transit ephemeris.'

            # Retrieve info from the NASA Exoplanet Archive
            if self.database == 'nasa':
                # For now use a local table instead of looking up online
                # because I need to figure out how to add more than one column
                # to the online query
                self.planet_properties = \
                    NasaExoplanetArchive.query_planet(
                        self.name, table_path='../data/planets.csv')
                self.period = self.planet_properties['pl_orbper']
                self.transit_midpoint = \
                    Time(self.planet_properties['pl_tranmid'], format='jd')
                self.duration14 = self.planet_properties['pl_trandur']

            # Retrieve info from the Exoplanet Orbit Database
            elif self.database == 'orbit':
                self.planet_properties = \
                    ExoplanetOrbitDatabase.query_planet(self.name)
                self.period = self.planet_properties['PER']
                self.transit_midpoint = \
                    Time(self.planet_properties['TT'], format='jd')
                self.duration14 = self.planet_properties['T14']

        else:
            pass

        self.system = EclipsingSystem(
            primary_eclipse_time=self.transit_midpoint,
            orbital_period=self.period, duration=self.duration14,
            eccentricity=self.eccentricity, name=self.name)

    # Find the next transit(s).
    def next_transit(self, jd_range):
        """
        Method to look for transit events inside a Julian Date range.

        Args:

            jd_range (array-like): The Julian Date interval where to look up
                for transit events.

        Returns:

            midtransit_times (``list``): List of Julian Dates of transit events.
        """
        jd_range = np.array(jd_range)
        jd0 = Time(jd_range[0], format='jd')
        jd1 = Time(jd_range[1], format='jd')
        jd_center = (jd1.jd + jd0.jd) / 2

        n_transits = int((jd1 - self.transit_midpoint).value /
                         self.period.to(u.d).value) - \
            int((jd0 - self.transit_midpoint).value /
                self.period.to(u.d).value)

        if n_transits > 0:
            midtransit_times = \
                self.system.next_primary_eclipse_time(jd0,
                                                      n_eclipses=n_transits)
        # If no transit was found, simply figure out the closest event
        else:
            next_event = self.system.next_primary_eclipse_time(jd1,
                n_eclipses=1)[0]
            next_event = Time(next_event, format='jd')
            previous_event = next_event - self.period
            if next_event.jd - jd_center < jd_center - previous_event.jd:
                midtransit_times = Time([next_event.value], format='jd')
            else:
                midtransit_times = Time([previous_event.value],
                                        format='jd')

        return midtransit_times


# The light curve object
class LightCurve(object):
    """
    Light curve object, used to compute the light curves (duh!) of the
    integrated flux inside a specific wavelength range or spectral line.

    Args:

        visit (``sunburn.hst_observation.Visit`` object): HST visit object.

        transit (``sunburn.analysis.Transit`` object): Exoplanet transit object.

        line_list (``dict``, optional): Spectral line list.
    """
    def __init__(self, visit, transit=None, line_list=None,
                 transit_search_expansion=(0.0 * u.d, 0.0 * u.d)):
        self.visit = visit
        self.transit = transit
        self.line_list = line_list

        # Instantiating useful global variables
        self._expand = transit_search_expansion
        self.integrated_flux = None
        self.time = []
        self.t_span = []
        self.phase = []
        self.f_unc = None

        # Instantiate variables for tag-split data
        self.tt_integrated_flux = None
        self.tt_time = None
        self.tt_phase = None
        self.tt_t_span = None
        self.tt_f_unc = None

        # Reading time information
        jd_start = []
        jd_end = []
        for i in self.visit.orbit:
            orbit = self.visit.orbit[i]
            self.time.append((orbit.start_JD.jd + orbit.end_JD.jd) / 2)
            self.t_span.append((orbit.end_JD.jd - orbit.start_JD.jd) / 2)
            jd_start.append(orbit.start_JD)
            jd_end.append(orbit.end_JD)

            # If time-tag split data are available, compute information from
            # them
            if orbit.split is not None:
                if self.tt_time is None or self.tt_t_span is None:
                    self.tt_time = []
                    self.tt_t_span = []
                    self.tt_phase = []
                else:
                    pass
                n_splits = len(orbit.split)
                for j in range(n_splits):
                    self.tt_time.append((orbit.split[j].start_JD.jd +
                                         orbit.split[j].end_JD.jd) / 2)
                    self.tt_t_span.append((orbit.split[j].end_JD.jd -
                                           orbit.split[j].start_JD.jd) / 2)

        # Figure out the range of julian dates spanned by the visit
        jd_start = np.array(jd_start)
        jd_end = np.array(jd_end)
        self.jd_range = (np.min(jd_start) - self._expand[0],
                         np.max(jd_end) + self._expand[1])

        # Transform lists into numpy arrays
        self.time = np.array(self.time)
        self.t_span = np.array(self.t_span)
        if self.tt_time is not None or self.tt_t_span is not None:
            self.tt_time = np.array(self.tt_time)
            self.tt_t_span = np.array(self.tt_t_span)
        else:
            pass

        # Now look if there is a transit happening during the visit
        if self.transit is not None:
            self.transit_midpoint = self.transit.next_transit(self.jd_range)
            if self.transit_midpoint is None:
                warnings.warn("No transit was found during this visit.")

        # Compute the phases in relation to transit midpoint
        if self.transit is not None and self.transit_midpoint is not None:
            # If there is only one transit inside the visit
            if len(self.transit_midpoint) == 1:
                self.phase = ((self.time - self.transit_midpoint.value) *
                              u.d).to(u.h).value
                if self.tt_time is not None:
                    self.tt_phase = ((self.tt_time -
                                      self.transit_midpoint.value) *
                                     u.d).to(u.h).value
                    # Check if the phases are the best possible
                    for p in self.tt_phase:
                        if p < -self.transit.period.to(u.h).value / 2:
                            p += self.transit.period.to(u.h).value
                        elif p > self.transit.period.to(u.h).value / 2:
                            p -= self.transit.period.to(u.h).value
                        else:
                            pass

            # If there are two or more transits inside the visit, figure out
            # the phases
            else:
                self.phase = np.zeros_like(self.time)
                for i in range(len(self.time)):
                    time_diff = abs(self.time[i] - self.transit_midpoint.value)
                    ind = np.where(time_diff == np.min(time_diff))
                    self.phase[i] = \
                        ((self.time[i] - self.transit_midpoint[ind].value) *
                         u.d).to(u.h).value
                if self.tt_time is not None:
                    for i in range(len(self.tt_time)):
                        time_diff = abs(
                            self.tt_time[i] - self.transit_midpoint.value)
                        ind = np.where(time_diff == np.min(time_diff))
                        self.tt_phase.append(
                            ((self.tt_time[i] - self.transit_midpoint[ind].value) *
                             u.d).to(u.h).value)
                    self.tt_phase = np.array(self.tt_phase)
                    # Check if the phases are the best possible
                    for p in self.tt_phase:
                        if p < -self.transit.period.to(u.h).value / 2:
                            p += self.transit.period.to(u.h).value
                        elif p > self.transit.period.to(u.h).value / 2:
                            p -= self.transit.period.to(u.h).value
                        else:
                            pass

            # Check if phases are good
            for i in range(len(self.phase)):
                if self.phase[i] < -self.transit.period.to(u.h).value / 2:
                    self.phase[i] += self.transit.period.to(u.h).value
                elif self.phase[i] > self.transit.period.to(u.h).value / 2:
                    self.phase[i] -= self.transit.period.to(u.h).value
                else:
                    pass

    # Compute the integrated flux
    def compute_flux(self, velocity_range=None, transition=None,
                     line_index=None, doppler_shift_corr=0.0,
                     wavelength_range=None, recompute_from_splits=False,
                     detrend_factor_splits=None):
        """
        Compute the flux in a given wavelength range or for a line from the line
        list.

        Args:

            wavelength_range (array-like, optional): Lower and upper limits of
                wavelengths where to compute the integrated flux, with shape
                (2, ). If ``None``, than the transition and line index must be
                provided. Default is ``None``.

            transition (``str``, optional): Transition of the line to compute
                the integrated flux. Example: ``'C II'``. If ``None``, than
                the wavelength range must be provided. Default is ``None``.

            line_index (``int``, optional): Index of the line to compute
                the integrated flux. If ``None``, than the wavelength range must
                be provided. Default is ``None``.
        """
        self.integrated_flux = []
        self.f_unc = []
        self.tt_integrated_flux = []
        self.tt_f_unc = []
        n_orbits = len(self.visit.orbit)
        light_speed = c.c.to(u.km / u.s).value

        # The local function to perform flux integration
        def _integrate(wl_range, split_detrend_factor=None):
            """

            Args:
                wl_range:
                split_detrend_factor:

            Returns:

            """
            # For each orbit in visit, compute the integrated flux
            i = 0
            for o in self.visit.orbit:
                orbit = self.visit.orbit[o]
                int_f, unc = orbit.integrated_flux(wl_range)
                self.integrated_flux.append(int_f)
                self.f_unc.append(unc)

                # In addition, for each split, compute the integrate flux, if
                # there are time-tag split data available
                if orbit.split is not None:
                    n_splits = len(orbit.split)
                    temp_int_f = 0
                    temp_f_unc = 0
                    if split_detrend_factor is None:
                        scale = np.ones(n_splits)
                    else:
                        scale = split_detrend_factor[i]
                    for sk in range(n_splits):
                        int_f, unc = orbit.split[sk].integrated_flux(
                            wl_range)
                        int_f *= scale[sk]
                        temp_int_f += int_f
                        temp_f_unc += unc ** 2
                        self.tt_integrated_flux.append(int_f)
                        self.tt_f_unc.append(unc)

                    # If user asks to recompute fluxes from splits
                    if recompute_from_splits is True:
                        self.integrated_flux[-1] = temp_int_f / n_splits
                        self.f_unc[-1] = (temp_f_unc ) ** 0.5 / n_splits
                i += 1

        # If the wavelength range is provided, just straight out compute the LC
        if wavelength_range is not None and \
                isinstance(wavelength_range, np.ndarray) is False:
            n_bands = 1
            _integrate(wavelength_range, detrend_factor_splits)
        # If the user provide a ``numpy.array`` with various bandpasses, combine
        # their integrated fluxes
        elif isinstance(wavelength_range, np.ndarray) is True:
            n_bands = np.shape(wavelength_range)[0]
            for band in wavelength_range:
                _integrate(band, detrend_factor_splits)
        # If not, then the wavelength range has to be figured out from the given
        # transition, Doppler shift correction and velocity range
        else:
            assert isinstance(self.line_list, dict), 'Either a wavelength ' \
                                                     'range or a transition ' \
                                                     'has to be provided.'

            # If the line_index us just a scalar, only one wavelength range will
            # be used
            if isinstance(line_index, int):
                n_bands = 1
                # Find the wavelength range
                central_wl = \
                    self.line_list[transition][line_index].central_wavelength
                ds_range = np.array(velocity_range) + doppler_shift_corr
                wavelength_range = ds_range / light_speed * central_wl + \
                    central_wl
                # Compute the integrated flux for the line
                _integrate(wavelength_range, detrend_factor_splits)

            # If more than one line is requested, then the integrated fluxes
            # will be co-added line by line
            elif isinstance(line_index, list):
                n_bands = len(line_index)
                count = 0
                for k in line_index:
                    # Find the wavelength range for the line
                    central_wl = \
                        self.line_list[transition][k].central_wavelength
                    try:
                        ds_range = np.array(velocity_range) + \
                            doppler_shift_corr[count]
                    except IndexError:
                        ds_range = np.array(velocity_range) + doppler_shift_corr
                    wavelength_range = ds_range / light_speed * central_wl + \
                        central_wl
                    # Compute the integrated flux for the line
                    _integrate(wavelength_range, detrend_factor_splits)
                    count += 1

            else:
                raise ValueError('`line_index` must be `int` or `list`.')

        # Transform the lists into numpy arrays
        self.integrated_flux = np.array(self.integrated_flux)
        self.f_unc = np.array(self.f_unc)
        self.tt_integrated_flux = np.array(self.tt_integrated_flux)
        self.tt_f_unc = np.array(self.tt_f_unc)

        # Co-add the integrated line fluxes if more than one was requested
        if n_bands > 1:
            temp = self.integrated_flux.reshape((n_bands, n_orbits))
            self.integrated_flux = np.sum(temp, axis=0)
            temp = self.f_unc.reshape((n_bands, n_orbits))
            self.f_unc = np.sqrt(np.sum(temp ** 2, axis=0))
            temp = self.tt_integrated_flux.reshape(
                (n_bands, len(self.tt_integrated_flux) // n_bands))
            self.tt_integrated_flux = np.sum(temp, axis=0)
            temp = \
                self.tt_f_unc.reshape((n_bands, len(self.tt_f_unc) // n_bands))
            self.tt_f_unc = np.sqrt(np.sum(temp ** 2, axis=0))

    # Plot the light curve
    def plot(self, figure_sizes=(9.0, 6.5), axes_font_size=18,
             label_choice='iso_date', symbol='o', symbol_color=None, fold=False,
             plot_splits=True, norm_factor=None, norm_uncertainty=0.0,
             transit_lines=True, log_flux_scale=None, relative_phase=False,
             **kwargs):
        """
        Plot the light curve. It is necessary to use
        ``matplotlib.pyplot.plot()`` after running this method to visualize the
        plot.

        Args:
            figure_sizes (array-like, optional): Sizes of the x- and y-axes of
                the plot. Default values are 9.0 for the x-axis and 6.5 for the
                y-axis.

            axes_font_size (``int``, optional): Font size of the axes marks.
                Default value is 18.

            label_choice (``str``, optional): Choice of label to be used in the
                light curve of the specified visit. Defaults to the ISO date
                of the start of the first orbit.

            symbol_color (``str``, optional):

            fold (``bool``, optional): If ``True``, then fold the light curve
                in the period of the exoplanet. Default is
                ``False``.

            plot_splits (``bool``, optional): If available, plot the
                split-tagged data in addition. Default value is ``True``.

            norm_factor (``float`` or ``astropy.Quantity``, optional):
                Normalization factor to apply to light curve. If ``float``, then
                assume the unit of the factor is erg / s / (cm ** 2). If
                ``None``, then do not normalize. Default is ``None``.

            transit_lines (``bool``, optional):
                If ``True``, then plot vertical lines corresponding to the
                ingress, egress and mid-transit times.
        """
        pylab.rcParams['figure.figsize'] = figure_sizes[0], figure_sizes[1]
        pylab.rcParams['font.size'] = axes_font_size

        # Choose label type for each visit
        if label_choice == 'iso_date':
            label = (Time(self.jd_range[0], format='jd')).iso
        elif label_choice == 'jd' or label_choice == 'julian_date':
            label = (Time(self.jd_range[0], format='jd')).jd
        elif label_choice is None:
            label = None
        else:
            label = str(label_choice)

        if isinstance(norm_factor, u.Quantity):
            norm = norm_factor.to(u.erg / u.s / u.cm ** 2).value
        elif isinstance(norm_factor, float):
            norm = norm_factor
        else:
            norm = 1.0

        if isinstance(log_flux_scale, int):
            scale = 10 ** log_flux_scale
            ylabel = r'Flux (10$^{%i}$ erg s$^{-1}$ cm$^{-2}$)' % log_flux_scale
        else:
            scale = 1.0
            ylabel = r'Flux (erg s$^{-1}$ cm$^{-2}$)'

        # Compute the correct values of the flux uncertainties taking into
        # account the uncertainty of the normalization
        f_plot = self.integrated_flux / norm / scale
        unc_plot = f_plot * ((self.f_unc / self.integrated_flux) ** 2 +
                             (norm_uncertainty / norm) ** 2) ** 0.5

        ax = plt.subplot()

        # Plot the integrated fluxes
        if fold is False:
            ax.errorbar(self.time, f_plot, xerr=self.t_span, yerr=unc_plot,
                         fmt=symbol, label=label, color=symbol_color, **kwargs)
            ax.set_xlabel('Julian date')
        else:
            t_span_mod = (self.t_span * u.d).to(u.h).value
            if relative_phase is True:
                t_plot = self.phase / self.transit.period.to(u.h).value
                xlabel = 'Phase'
            else:
                t_plot = self.phase
                xlabel = 'Time (h)'
            ax.errorbar(t_plot, f_plot, xerr=t_span_mod, yerr=unc_plot,
                         fmt=symbol, label=label, color=symbol_color, **kwargs)
            ax.set_xlabel(xlabel)

        if norm_factor is None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(r'Normalized flux')

        # Plot the time-tag split data, if they are available
        if len(self.tt_integrated_flux) > 0 and plot_splits is True:
            f_plot = self.tt_integrated_flux / norm / scale
            unc_plot = f_plot * ((self.tt_f_unc / self.tt_integrated_flux) ** 2
                                 + (norm_uncertainty / norm) ** 2) ** 0.5
            if fold is False:
                ax.errorbar(self.tt_time, f_plot, xerr=self.tt_t_span,
                             yerr=unc_plot, fmt='.', color=symbol_color,
                             alpha=0.2)
            else:
                if relative_phase is True:
                    t_plot = self.tt_phase / self.transit.period.to(u.h).value
                else:
                    t_plot = self.tt_phase
                t_span_mod = (self.tt_t_span * u.d).to(u.h).value
                ax.errorbar(t_plot, f_plot, xerr=t_span_mod,
                             yerr=unc_plot, fmt='.', color=symbol_color,
                             alpha=0.2)

        # Plot the transit times
        if self.transit is not None and self.transit_midpoint is not None and \
                transit_lines is True:
            if fold is False:
                for jd in self.transit_midpoint:
                    #plt.axvline(x=jd.jd, ls='--', color='k')
                    ax.axvline(
                        x=jd.jd - self.transit.duration14.to(u.d).value / 2,
                        color='r')
                    ax.axvline(
                        x=jd.jd + self.transit.duration14.to(u.d).value / 2,
                        color='r')
                    if self.transit.duration23 is not None:
                        ax.axvline(
                            x=jd.jd - self.transit.duration23.to(u.d).value / 2,
                            ls='-.', color='r')
                        ax.axvline(
                            x=jd.jd + self.transit.duration23.to(u.d).value / 2,
                            ls='-.', color='r')
            else:
                #plt.axvline(x=0.0, ls='--', color='k')
                ax.axvline(x=-self.transit.duration14.to(u.h).value / 2,
                            color='r')
                ax.axvline(x=self.transit.duration14.to(u.h).value / 2,
                            color='r')
                if self.transit.duration23 is not None:
                    ax.axvline(
                        x=-self.transit.duration23.to(u.h).value / 2,
                        ls='-.', color='r')
                    ax.axvline(
                        x=self.transit.duration23.to(u.h).value / 2,
                        ls='-.', color='r')

        return ax


# The co-added light curve object
class CombinedLightCurve(object):
    """

    """
    def __init__(self, light_curves, phase_bins, norm_factors=None):
        self.n_lc = len(light_curves)
        self.phase_bins = phase_bins
        self.n_p = len(phase_bins) - 1
        self.norm = norm_factors
        self.transit = light_curves[0].transit

        self.integrated_flux = np.zeros(self.n_p)
        self.f_unc = np.zeros(self.n_p)
        self._counts = np.zeros(self.n_p)
        self.phase = np.zeros(self.n_p)

        # For each light curve...
        for lc in light_curves:
            # For each orbit in the light curve...
            for i in range(len(lc.integrated_flux)):
                # Find if the current flux is inside the phase bin
                for j in range(self.n_p):
                    if phase_bins[j] < lc.phase[i] < phase_bins[j + 1]:
                        # If so, co-add it
                        self.integrated_flux[j] += lc.integrated_flux[i]
                        self.f_unc[j] += lc.f_unc[i] ** 2
                        self.phase[j] += lc.phase[i]
                        self._counts[j] += 1
                    else:
                        pass

        # Finally divide by number of counts in each bin
        self.integrated_flux /= self._counts
        self.f_unc = self.f_unc ** 0.5 / self._counts
        self.phase /= self._counts
        self.xspan = np.array([[self.phase[i] - self.phase_bins[i],
                                self.phase_bins[i + 1] - self.phase[i]]
                               for i in range(self.n_p)]).T

    # Plot the combined light curve
    def plot(self, norm=1, norm_unc=0.0, figure_sizes=(9.0, 6.5),
             axes_font_size=18, label='', symbol=None, symbol_color=None,
             symbol_size=None, transit_lines=True, plot_xerr=True, **kwargs):
        """

        Args:
            norm:
            norm_unc:
            figure_sizes:
            axes_font_size:
            label:
            symbol:
            symbol_color:
            symbol_size:
            transit_lines:
            **kwargs:

        Returns:

        """
        pylab.rcParams['figure.figsize'] = figure_sizes[0], figure_sizes[1]
        pylab.rcParams['font.size'] = axes_font_size

        if symbol is None:
            symbol = 'o'

        f_plot = self.integrated_flux / norm
        unc_plot = f_plot * ((self.f_unc / self.integrated_flux) ** 2 +
                             (norm_unc / norm) ** 2) ** 0.5
        if plot_xerr is True:
            xerr_plot = self.xspan
        else:
            xerr_plot = None

        plt.errorbar(self.phase, f_plot, yerr=unc_plot, xerr=xerr_plot,
                     fmt=symbol, label=label, color=symbol_color,
                     markersize=symbol_size, **kwargs)

        # Plot the transit lines
        if transit_lines is True:
            plt.axvline(x=0.0, ls='--', color='k')
            plt.axvline(x=-self.transit.duration14.to(u.h).value / 2,
                        color='r')
            plt.axvline(x=self.transit.duration14.to(u.h).value / 2,
                        color='r')
            if self.transit.duration23 is not None:
                plt.axvline(
                    x=-self.transit.duration23.to(u.h).value / 2,
                    ls='-.', color='r')
                plt.axvline(
                    x=self.transit.duration23.to(u.h).value / 2,
                    ls='-.', color='r')

        # Plot axes labels
        if norm == 1:
            plt.ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$)')
        else:
            plt.ylabel(r'Normalized flux')
        plt.xlabel('Time (h)')

'''
    # Fit the data to a polynomial trend of the jitter parameters
    def fit_trend(self, params, jitter_model=None, first_guess=None, norm=1E-16,
                  **kwargs):
        """

        Args:
            params:
            jitter_model:
            first_guess:
            norm:
            **kwargs:

        Returns:

        """
        if first_guess is None:
            first_guess = []
            for p in params:
                first_guess.append(1.0)
                first_guess.append(1.0)
            first_guess = np.array(first_guess)
        else:
            pass

        jitter_vector = []
        for p in params:
            jitter_vector.append(self.orbit.jitter_data[p])
        jitter_vector = np.array(jitter_vector)

        # The jitter model is a linear function of the jitter data
        def _model(jitter_matrix, p_matrix):
            a = p_matrix[:, 0]
            b = p_matrix[:, 1]
            f_matrix = (a * jitter_matrix.T + b).T
            f_value = np.sum(f_matrix, axis=0)
            return f_value

        # Evaluate the model at the flux time stamps and calculate the
        # log-likelihood
        def _fun(theta):
            p = np.reshape(theta, (len(theta) // 2, 2))
            if jitter_model is None:
                f_model = _model(jitter_vector, p)
            else:
                f_model = jitter_model(jitter_vector, p)
            f_value = np.interp(self.time, self.orbit.jitter_data['Seconds'],
                                f_model)
            diff = self.flux / norm - f_value
            return np.sum(diff ** 2 / (self.uncertainty / norm) ** 2)

        # Fit the observed time-tag fluxes to the jitter model
        result = minimize(_fun, x0=first_guess, **kwargs)
        self.fit_result = result
        if jitter_model is None:
            self.trend = _model(jitter_vector, np.reshape(
                result.x, (len(result.x) // 2, 2))) * norm
        else:
            self.trend = jitter_model(jitter_vector, np.reshape(
                result.x, (len(result.x) // 2, 2))) * norm
        return result

    # Plot the trend
    def plot_trend(self):
        plt.plot(self.orbit.jitter_data['Seconds'], self.trend)
        plt.errorbar(self.time, self.flux, xerr=self.t_span,
                     yerr=self.uncertainty, fmt='o')
'''
