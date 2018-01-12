#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of HST/COS data.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.time import Time
from astroplan import EclipsingSystem
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.exoplanet_orbit_database import ExoplanetOrbitDatabase


__all__ = []


class Transit(object):
    """

    """
    def __init__(self, planet_name=None, period=None, transit_midpoint=None,
                 duration14=None, duration23=None, database='nasa'):
        self.name = planet_name
        self.period = period
        self.transit_midpoint = transit_midpoint
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
            name=self.name)

    def find_transit(self, jd_range):
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

        midtransit_times = \
            self.system.next_primary_eclipse_time(jd0, n_eclipses=n_transits)

        return midtransit_times


# The light curve object
class LightCurve(object):
    """

    Args:
        visit (``tuple``): Tuple containing the visit objects
    """
    def __init__(self, visit, transit, wavelength_range):
        self.visit = visit
        self.transit = transit
        self.wavelength_range = wavelength_range

        # Instantiating useful global variables
        self.flux = []
        self.time = []
        self.t_span = []
        self.f_unc = []

