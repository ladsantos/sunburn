#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of HST/COS data.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import warnings
from astropy.time import Time
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
                 database='nasa'):
        self.name = planet_name
        self.period = period
        self.transit_midpoint = transit_midpoint
        self.eccentricity = eccentricity
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

        n_transits = int((jd1 - self.transit_midpoint).value /
                         self.period.to(u.d).value) - \
            int((jd0 - self.transit_midpoint).value /
                self.period.to(u.d).value)

        if n_transits > 0:
            midtransit_times = \
                self.system.next_primary_eclipse_time(jd0,
                                                      n_eclipses=n_transits)
        else:
            midtransit_times = None

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
    def __init__(self, visit, transit=None, line_list=None):
        self.visit = visit
        self.transit = transit
        self.line_list = line_list

        # Instantiating useful global variables
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
                else:
                    pass
                n_splits = len(orbit.split)
                for j in range(n_splits):
                    self.tt_time.append((orbit.split[j].start_JD.jd +
                                         orbit.split[j].end_JD.jd) / 2)
                    self.tt_t_span.append((orbit.split[j].start_JD.jd -
                                           orbit.split[j].end_JD.jd) / 2)

        # Figure out the range of julian dates spanned by the visit
        jd_start = np.array(jd_start)
        jd_end = np.array(jd_end)
        self.jd_range = (np.min(jd_start), np.max(jd_end))

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
            self.phase = ((self.time - self.transit_midpoint.value) *
                          u.d).to(u.h).value
            if self.tt_time is not None:
                self.tt_phase = ((self.tt_time - self.transit_midpoint.value) *
                                 u.d).to(u.h).value

    # Systematic correction using a polynomial
    def correct_systematic(self, reference_line_list, baseline_level,
                           poly_deg=1, temp_jd_shift=2.45E6):
        """
        Correct the systematics of a HST/COS orbit by fitting a polynomial to
        the sum of the integrated fluxes of various spectral lines (these lines
        should preferably not have a transiting signal) for a series of time-tag
        split data.

        Args:

            reference_line_list (`COSFUVLineList` object):

            baseline_level ():

            poly_deg (``int``, optional): Degree of the polynomial to be fit.
                Default value is 1.

            temp_jd_shift (``float``, optional): In order to perform a proper
                fit, it is necessary to temporarily modify the Julian Date to a
                smaller number, which is done by subtracting the value of this
                variable from the Julian Dates. Default value is 2.45E6.
        """
        pass

    # Compute the integrated flux
    def compute_flux(self, wavelength_range=None, transition=None,
                     line_index=None, wing=None, correct_systematic=None):
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

            wing (``str``, optional): Choose to compute the integrated flux in
                the blue or red wing of the line. Not implemented yet. Default
                is ``None``.
        """
        self.integrated_flux = []
        self.f_unc = []
        self.tt_integrated_flux = []
        self.tt_f_unc = []
        n_orbits = len(self.visit.orbit)

        # The local function to perform flux integration
        def _integrate(wl_range):
            """

            Args:
                wl_range:

            Returns:

            """
            # For each orbit in visit, compute the integrated flux
            for o in self.visit.orbit:
                orbit = self.visit.orbit[o]
                int_f, unc = orbit.integrated_flux(wl_range)
                self.integrated_flux.append(int_f)
                self.f_unc.append(unc)

                # In addition, for each split, compute the integrate flux, if
                # there are time-tag split data available
                if orbit.split is not None:
                    n_splits = len(orbit.split)
                    for i in range(n_splits):
                        int_f, unc = orbit.split[i].integrated_flux(
                            wl_range)
                        self.tt_integrated_flux.append(int_f)
                        self.tt_f_unc.append(unc)

        # Figure out the wavelength ranges
        if wavelength_range is None:
            assert isinstance(self.line_list, dict), 'Either a wavelength ' \
                                                     'range or a transition ' \
                                                     'has to be provided.'

            # If the line_index us just a scalar, only one wavelength range will
            # be used
            if isinstance(line_index, int):
                n_lines = 1
                # Find the wavelength range
                wavelength_range = \
                    self.line_list[transition][line_index].wavelength_range
                if wing == 'blue':
                    wavelength_range[1] = \
                        self.line_list[transition][line_index].central_wavelength
                if wing == 'red':
                    wavelength_range[0] = \
                        self.line_list[transition][line_index].central_wavelength
                # Compute the integrated flux for the line
                _integrate(wavelength_range)

            # If more than one line is requested, then the integrated fluxes
            # will be co-added line by line
            elif isinstance(line_index, list):
                n_lines = len(line_index)
                for k in line_index:
                    # Find the wavelength range for the line
                    wavelength_range = \
                        self.line_list[transition][k].wavelength_range
                    if wing == 'blue':
                        wavelength_range[1] = \
                            self.line_list[transition][k].central_wavelength
                    if wing == 'red':
                        wavelength_range[0] = \
                            self.line_list[transition][k].central_wavelength
                    # Compute the integrated flux for the line
                    _integrate(wavelength_range)

            else:
                raise ValueError('`line_index` must be `int` or `list`.')

        else:
            n_lines = 1
            _integrate(wavelength_range)

        # Transform the lists into numpy arrays
        self.integrated_flux = np.array(self.integrated_flux)
        self.f_unc = np.array(self.f_unc)
        self.tt_integrated_flux = np.array(self.tt_integrated_flux)
        self.tt_f_unc = np.array(self.tt_f_unc)

        # Co-add the integrated line fluxes if more than one was requested
        if n_lines > 1:
            temp = self.integrated_flux.reshape((n_lines, n_orbits))
            self.integrated_flux = np.sum(temp, axis=0)
            temp = self.f_unc.reshape((n_lines, n_orbits))
            self.f_unc = np.sqrt(np.sum(temp ** 2, axis=0))
            temp = self.tt_integrated_flux.reshape(
                (n_lines, len(self.tt_integrated_flux) // n_lines))
            self.tt_integrated_flux = np.sum(temp, axis=0)
            temp = self.tt_f_unc.reshape((n_lines, len(self.tt_f_unc) // n_lines))
            self.tt_f_unc = np.sqrt(np.sum(temp ** 2, axis=0))

    # Plot the light curve
    def plot(self, figure_sizes=(9.0, 6.5), axes_font_size=18,
             label_choice='iso_date', symbol_color=None, fold=False,
             plot_splits=True, norm_factor=None):
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

        # Plot the integrated fluxes
        if fold is False:
            plt.errorbar(self.time, self.integrated_flux / norm,
                         xerr=self.t_span, yerr=self.f_unc / norm, fmt='o',
                         label=label, color=symbol_color)
            plt.xlabel('Julian date')
        else:
            time_mod = \
                ((self.time - self.transit_midpoint.value) * u.d).to(u.h).value
            t_span_mod = (self.t_span * u.d).to(u.h).value
            plt.errorbar(time_mod,
                         self.integrated_flux / norm, xerr=t_span_mod,
                         yerr=self.f_unc / norm, fmt='o', label=label,
                         color=symbol_color)
            plt.xlabel('Time (h)')

        if norm_factor is None:
            plt.ylabel(r'Integrated flux (erg s$^{-1}$ cm$^{-2}$)')
        else:
            plt.ylabel(r'Normalized integrated flux')

        # Plot the time-tag split data, if they are available
        if len(self.tt_integrated_flux) > 0 and plot_splits is True:
            if fold is False:
                plt.errorbar(self.tt_time, self.tt_integrated_flux / norm,
                             xerr=self.tt_t_span, yerr=self.tt_f_unc / norm,
                             fmt='.', color=symbol_color, alpha=0.2)
            else:
                # TODO: For now this only works if there's only one transit
                # per visit. Implement a solution for when there's more than
                # one transit event.
                time_mod = \
                    ((self.tt_time - self.transit_midpoint.value) * u.d).to(
                        u.h).value
                t_span_mod = (self.tt_t_span * u.d).to(u.h).value
                plt.errorbar(time_mod,
                             self.tt_integrated_flux / norm,
                             xerr=t_span_mod, yerr=self.tt_f_unc / norm,
                             fmt='.', color=symbol_color, alpha=0.2)

        # Plot the transit times
        if self.transit is not None and self.transit_midpoint is not None:
            if fold is False:
                for jd in self.transit_midpoint:
                    plt.axvline(x=jd.jd, ls='--', color='k')
                    plt.axvline(
                        x=jd.jd - self.transit.duration14.to(u.d).value / 2,
                        color='r')
                    plt.axvline(
                        x=jd.jd + self.transit.duration14.to(u.d).value / 2,
                        color='r')
                    if self.transit.duration23 is not None:
                        plt.axvline(
                            x=jd.jd - self.transit.duration23.to(u.d).value / 2,
                            ls='-.', color='r')
                        plt.axvline(
                            x=jd.jd + self.transit.duration23.to(u.d).value / 2,
                            ls='-.', color='r')
            else:
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

    # Plot the combined light curve
    def plot(self, norm=1, figure_sizes=(9.0, 6.5), axes_font_size=18,
             label='', symbol=None, symbol_color=None, symbol_size=None):
        """

        Args:
            norm:
            figure_sizes:
            axes_font_size:

        Returns:

        """
        pylab.rcParams['figure.figsize'] = figure_sizes[0], figure_sizes[1]
        pylab.rcParams['font.size'] = axes_font_size

        if symbol is None:
            symbol = 'o'

        plt.errorbar(self.phase, self.integrated_flux / norm,
                     yerr=self.f_unc / norm, fmt=symbol, label=label,
                     color=symbol_color, markersize=symbol_size)

        # Plot the transit lines
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
            plt.ylabel(r'Integrated flux (erg s$^{-1}$ cm$^{-2}$)')
        else:
            plt.ylabel(r'Normalized integrated flux')
        plt.xlabel('Time (h)')
