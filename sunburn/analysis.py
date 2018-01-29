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
                        self.name, select_columns=('pl_hostname', 'sky_coord', 'pl_letter',
                                                   'pl_orbper', 'pl_tranmid',
                                                   'pl_trandur'))
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
    def __init__(self, visit, transit, line_list=None):
        self.visit = visit
        self.transit = transit
        self.line_list = line_list

        # Instantiating useful global variables
        self.integrated_flux = []
        self.time = []
        self.t_span = []
        self.f_unc = []

        # Reading time information
        jd_start = []
        jd_end = []
        for i in self.visit.orbit:
            orbit = self.visit.orbit[i]
            self.time.append((orbit.start_JD.jd + orbit.end_JD.jd) / 2)
            self.t_span.append((orbit.end_JD.jd - orbit.start_JD.jd) / 2)
            jd_start.append(orbit.start_JD)
            jd_end.append(orbit.end_JD)

        # Figure out the range of julian dates spanned by the visit
        jd_start = np.array(jd_start)
        jd_end = np.array(jd_end)
        self.jd_range = (np.min(jd_start), np.max(jd_end))

        # Now look if there is a transit happening during the visit
        self.transit_midpoint = self.transit.next_transit(self.jd_range)
        if self.transit_midpoint is None:
            warnings.warn("No transit was found during this visit.")

    # Compute the integrated flux
    def compute_flux(self, wavelength_range=None, transition=None,
                     line_index=None, wing=None):
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

        # For each orbit in visit, compute the integrated flux
        if wavelength_range is None:
            assert isinstance(self.line_list, dict), 'Either a wavelength ' \
                                                     'range or the line list ' \
                                                     'has to be provided.'
            wavelength_range = \
                self.line_list[transition][line_index].wavelength_range
            if wing == 'blue':
                wavelength_range[1] = \
                    self.line_list[transition][line_index].central_wavelength
            if wing == 'red':
                wavelength_range[0] = \
                    self.line_list[transition][line_index].central_wavelength

        for i in self.visit.orbit:
            orbit = self.visit.orbit[i]
            int_f, unc = orbit.integrated_flux(wavelength_range)
            self.integrated_flux.append(int_f)
            self.f_unc.append(unc)

    # Plot the light curve
    def plot(self, figure_sizes=(9.0, 6.5), axes_font_size=18,
             label_choice='iso_date', fold=False):
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

            fold (``bool``, optional): If ``True``, then fold the light curve
                in the period of the exoplanet. Not implemented yet. Default is
                ``False``.
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

        # Plot the integrated fluxes
        plt.errorbar(self.time, self.integrated_flux, xerr=self.t_span,
                     yerr=self.f_unc, fmt='o', label=label)
        plt.xlabel('Julian date')
        plt.ylabel(r'Integrated flux (erg s$^{-1}$ cm$^{-2}$)')

        # Plot the transit times
        if self.transit_midpoint is not None:
            for jd in self.transit_midpoint:
                plt.axvline(x=jd.jd, ls='--', color='k')
                plt.axvline(x=jd.jd - self.transit.duration14.to(u.d).value / 2,
                            color='r')
                plt.axvline(x=jd.jd + self.transit.duration14.to(u.d).value / 2,
                            color='r')
                if self.transit.duration23 is not None:
                    plt.axvline(
                        x=jd.jd - self.transit.duration23.to(u.d).value / 2,
                        ls='-.', color='r')
                    plt.axvline(
                        x=jd.jd + self.transit.duration23.to(u.d).value / 2,
                        ls='-.', color='r')
