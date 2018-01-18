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


__all__ = []


class Transit(object):
    """

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

    def next_transit(self, jd_range):
        """

        Args:
            jd_range:

        Returns:

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

    Args:
        visit (``Visit`` object):
        transit (``Transit`` object):
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

    # Compute the flux in a given wavelength range or for a line from the line
    # list
    def compute_flux(self, wavelength_range=None, transition=None,
                     line_index=None, wing=None):
        """

        Args:
            wavelength_range:
            transition:
            line_index:
            wing:

        Returns:

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

        Args:
            figure_sizes:
            axes_font_size:
            label_choice:

        Returns:

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
