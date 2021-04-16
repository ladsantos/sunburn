#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis of systematics of HST data.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)


import batman
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


from scipy.optimize import minimize
from scipy.stats import binned_statistic, spearmanr
from astropy.io import fits
from warnings import warn


class Jitter(object):
    """

    """
    def __init__(self, light_curve):
        self.lc = light_curve
        self.visit = self.lc.visit
        self.datasets = self.visit.dataset_names
        self.transit = self.lc.transit
        self.transit_midpoint = self.lc.transit_midpoint
        self._n_splits = []
        self._split_index = []

        # Instantiate variables for jitter data
        self.pearson_r = None
        self.p_value = None
        self.jitter_time = []
        self.jitter_phase = []
        self.jitter_data = None
        self.binned_jitter_data = None
        self.spread_jitter_data = None
        self.jitter_params = [col.name for col in
                                  self.visit.orbit[
                                      self.datasets[0]].jitter_columns]
        self._fudge_sample = {}

        # Instantiate global variables for the trend fitting
        self.fit_result = None
        self.log_likelihood = None
        self.best_fit_params = None
        self.fit_bic = None
        self.expected_flux = None
        self.detrend_factor = None
        self.detrended_flux = None

        # Begin populating the jitter metadata
        for i in self.visit.orbit:
            orbit = self.visit.orbit[i]
            n_splits = len(orbit.split)
            self._n_splits.extend(
                list(n_splits * np.ones(n_splits, dtype=int)))
            self._split_index.extend(list(np.arange(1, n_splits + 1)))
            if orbit.jitter_data is None:
                raise ValueError('There is no jitter information available for '
                                 'the dataset %s.' % i)
            else:
                pass

            # First add the time
            self.jitter_time.extend(
                orbit.start_JD.jd + (
                        orbit.jitter_data['Seconds'].astype(np.float128)
                        * u.s).to(u.d).value)

            # Add the phases in relation to the transit
            if self.transit is not None and self.transit_midpoint is not None:
                # If there is only one transit inside the visit
                if len(self.transit_midpoint) == 1:
                    self.jitter_phase = (
                                (self.jitter_time - self.transit_midpoint.value)
                                * u.d).to(u.h).value
                # If there are two or more transits inside the visit, figure out
                # the phases
                else:
                    raise ValueError('Currently only light curves with a '
                                     'single transit are supported.')
                # Check if phases are good
                for k in range(len(self.jitter_phase)):
                    if self.jitter_phase[k] < -self.transit.period.to(u.h).value / 2:
                        self.jitter_phase[k] += self.transit.period.to(u.h).value
                    elif self.jitter_phase[k] > self.transit.period.to(u.h).value / 2:
                        self.jitter_phase[k] -= self.transit.period.to(u.h).value
                    else:
                        pass

        # Make the jitter metadata as NumPy arrays
        self.jitter_time = np.array(self.jitter_time)
        self.jitter_phase = np.array(self.jitter_phase)

        # Now, we populate the jitter data
        list_of_cols = []
        for i in range(len(self.jitter_params)):
            temp = []
            name = self.jitter_params[i]
            unit = self.visit.orbit[self.datasets[0]].jitter_columns[name].unit
            fmt = self.visit.orbit[self.datasets[0]].jitter_columns[name].format
            for o in self.visit.orbit:
                orbit = self.visit.orbit[o]
                temp.extend(orbit.jitter_data[name])
            col = fits.Column(name, format=fmt, unit=unit, array=np.array(temp))
            list_of_cols.append(col)
        sum_of_cols = fits.ColDefs(list_of_cols)
        self.jitter_data = fits.FITS_rec.from_columns(sum_of_cols)

    # Plot the light curve and/or the jitter data
    def plot(self, param, include_light_curve=True, norm=None, y_shift=None,
             figure_sizes=(9.0, 6.5), axes_font_size=18, plot_splits=True,
             fold=True, lc_color=None, lc_log_flux_scale=None,
             transit_lines=False, **jit_kwargs):
        """

        Args:
            param:
            include_light_curve:
            norm:
            y_shift:
            figure_sizes:
            axes_font_size:
            plot_splits:
            fold:
            lc_color:
            lc_log_flux_scale:
            transit_lines:
            **jit_kwargs:

        Returns:

        """
        if fold is True:
            x = self.jitter_phase
        else:
            x = self.jitter_time

        # Shift the jitter data
        y = self.jitter_data[param]
        if y_shift == 'mean' or y_shift == 'average':
            y = y - np.mean(y)
        elif norm == 'median':
            y = y - np.median(y)
        elif isinstance(y_shift, float) is True:
            y -= y_shift
        else:
            pass

        # Normalize the jitter data
        if norm == 'mean' or norm == 'average':
            y = y / np.mean(y)
        elif norm == 'median':
            y = y / np.median(y)
        elif isinstance(norm, float) is True:
            y /= norm
        else:
            pass

        # First we plot the light curve
        if include_light_curve is True:
            ax1 = self.lc.plot(figure_sizes, axes_font_size=axes_font_size,
                               plot_splits=plot_splits, fold=fold,
                               symbol_color=lc_color,
                               log_flux_scale=lc_log_flux_scale,
                               transit_lines=transit_lines)
            ax2 = ax1.twinx()
            ax2.plot(x, y, **jit_kwargs)
        else:
            pylab.rcParams['figure.figsize'] = figure_sizes[0], figure_sizes[1]
            ax = plt.subplot()
            ax.plot(x, y, **jit_kwargs)
            ax.set_xlabel('Phase (h)')
            ax.set_ylabel(param)

    # Search for correlations between the light curve fluxes and the jitter
    # parameters
    def search_correlation(self, param_list, use_subexposures=True,
                           print_results=False, correlate_on_binned_data=True):
        """

        Args:
            param_list:
            use_subexposures:
            print_results:
            correlate_on_binned_data:

        Returns:

        """
        self.corr_r = []
        self.p_value = []

        if param_list is None:
            param_list = np.array(self.jitter_params)
            # Delete the first element of the list, because it's just the time
            param_list = np.delete(param_list, 0)
        else:
            pass

        # This function is used in the code
        def _stddev(any_sample):
            mean_any_sample = np.mean(any_sample)
            n = len(any_sample)
            sum2 = np.sum(np.array([(sk - mean_any_sample) ** 2 for sk in
                                    any_sample]))
            # Hacky hack
            if n == 1:
                n = 0
            else:
                pass
            # End of hacky hack

            stddev = (1 / (n - 1) * sum2) ** 0.5
            return stddev

        # Calculate the correlation by creating a fake sample of points for each
        # (sub-) exposure based on the uncertainties
        # Iterate through each jitter parameter
        self.binned_jitter_data = {}
        self.spread_jitter_data = {}
        for param in param_list:

            if use_subexposures is False:
                x = self.lc.time
                x_span = self.lc.t_span
                y = self.lc.integrated_flux
                y_u = self.lc.f_unc
            else:
                x = self.lc.tt_time
                x_span = self.lc.tt_t_span
                y = self.lc.tt_integrated_flux
                y_u = self.lc.tt_f_unc

            # Count the number of jitter data points inside the exposure bin
            sample = []
            binned_jitter_data = []
            spread_jitter_data = []
            for i in range(len(x)):
                count, bins, binnumber = binned_statistic(
                    self.jitter_time, self.jitter_data[param],
                    statistic='count',
                    bins=(x[i] - x_span[i], x[i] + x_span[i]))
                mean, bins, binnumber = binned_statistic(
                    self.jitter_time, self.jitter_data[param],
                    statistic='mean',
                    bins=(x[i] - x_span[i], x[i] + x_span[i]))
                stdev, bins, binnumber = binned_statistic(
                    self.jitter_time, self.jitter_data[param],
                    statistic=_stddev,
                    bins=(x[i] - x_span[i], x[i] + x_span[i]))
                # Create the fudge flux sample based on mean and uncertainties
                sample.extend(np.random.normal(loc=y[i], scale=y_u[i],
                                               size=int(count)))
                binned_jitter_data.append(mean[0])
                spread_jitter_data.append(stdev[0])
            sample = np.array(sample)
            # Hacky hack to deal with a sample with a different number of items
            # than the length of the jitter data
            if len(sample) < len(self.jitter_time):
                diff = len(self.jitter_time) - len(sample)
                sample = np.concatenate((sample, np.random.normal(loc=y[-1],
                                                                  scale=y_u[-1],
                                                                  size=int(diff)
                                                                  )))
            self.binned_jitter_data[param] = np.array(binned_jitter_data)
            self.spread_jitter_data[param] = np.array(spread_jitter_data)
            self._fudge_sample[param] = sample

            # Now that we have the fudge sample, we calculate the correlation
            # with the jitter data
            # Dirty hack to adjust the size of the fudge sample to match the
            # size
            if correlate_on_binned_data is False:
                corr_r, p_value = spearmanr(sample * 1E15,
                                            self.jitter_data[param])
            else:
                if use_subexposures is True:
                    corr_r, p_value = spearmanr(
                        self.lc.tt_integrated_flux * 1E15,
                        self.binned_jitter_data[param])
                elif use_subexposures is False:
                    corr_r, p_value = spearmanr(
                        self.lc.integrated_flux * 1E15,
                        self.binned_jitter_data[param])
            self.corr_r.append(corr_r)
            self.p_value.append(p_value)

            # Print the correlation results
            if print_results is True:
                print('%s: %.5f (p-value: %.5E)' % (param, corr_r, p_value))

    def plot_correlation(self, param, fudge=False, **kwargs):
        """

        Args:
            param:

        Returns:

        """
        if fudge is True:
            plt.plot(self.jitter_data[param], self._fudge_sample[param], '.',
                     **kwargs)
        else:
            try:
                plt.errorbar(self.binned_jitter_data[param],
                             self.lc.tt_integrated_flux, yerr=self.lc.tt_f_unc,
                             xerr=self.spread_jitter_data[param], fmt='o',
                             **kwargs)
            except ValueError:
                plt.errorbar(self.binned_jitter_data[param],
                             self.lc.integrated_flux, yerr=self.lc.f_unc,
                             xerr=self.spread_jitter_data[param], fmt='o',
                             **kwargs)

    # Fit a trend to fluxes in function of one or more jitter parameters
    def fit_trend(self, params, first_guess, jitter_model=None, norm=1E-16,
                  use_subexposures=True, fit_to_fudge=False,
                  independent_vars=None, poly_orders=None, **kwargs):
        """

        Args:
            params:
            first_guess:
            jitter_model:
            norm:
            use_subexposures:
            fit_to_fudge:
            extra_independent_vars:
            **kwargs:

        Returns:

        """
        jitter_matrix = []
        jitter_spread = []
        flux_vector = []
        flux_unc = []
        time = []
        n_params = len(params)

        for p in params:
            if fit_to_fudge is True:
                jitter_matrix.append(self.jitter_data[p])
                flux_vector = self._fudge_sample[p]
                time = self.jitter_phase
            else:
                jitter_matrix.append(self.binned_jitter_data[p])
                jitter_spread.append(self.spread_jitter_data[p])
                if use_subexposures is False:
                    flux_vector = self.lc.integrated_flux
                    flux_unc = self.lc.f_unc
                    time = self.lc.phase
                else:
                    flux_vector = self.lc.tt_integrated_flux
                    flux_unc = self.lc.tt_f_unc
                    time = self.lc.tt_phase
        jitter_matrix = np.array(jitter_matrix)
        jitter_spread = np.array(jitter_spread)

        if independent_vars is None:
            independent_vars = np.array([jitter_matrix, time])

        # The simplest model is the flux varying linearly with a jitter
        # parameter
        def _linear_model(x, theta):
            # A linear model has only two parameters, so we reshape the fit
            # parameter vector theta to have the shape (N, 2), where N is the
            # number jitter parameters
            theta_local = np.reshape(theta, (n_params, 2))
            jit, time = x
            a = theta_local[:, 0]
            b = theta_local[:, 1]
            y = (a * jit.T + b).T
            y = np.sum(y, axis=0)
            return y

        # Also a polynomial model
        def _poly_model(x, theta):
            # A polynomial has a number of parameters that depend on the degree
            # of the polynomial
            jit, time = x
            # If the user does not define the polynomial order, then assume all
            # jitter parameter fit follow polynomials with the same order and
            # figure them out from the theta length
            if poly_orders is None:
                order = (len(theta) // n_params) - 1
                theta_local = np.reshape(theta, (n_params, order + 1))
                y = np.array([np.polyval(tk, jit) for tk in theta_local])
            else:
                prev_order = 0
                y = []
                for i in range(n_params):
                    order = poly_orders[i]
                    ind0 = prev_order
                    ind1 = prev_order + order + 1
                    theta_current = theta[ind0:ind1]
                    prev_order += order + 1
                    y.append(np.polyval(theta_current, jit))
                y = np.array(y)
            y = np.sum(y, axis=0)[0]
            return y

        # A slightly more complicated model is the flux varying linearly with a
        # jitter parameter but also including a transit model (so there is a
        # dependence with time
        def _linear_with_transit_model(x, theta):
            jit, time = x
            r_p = theta[0]
            linear_theta = theta[1:]

            # Setup the batman model with a uniform limb darkening model
            transit_params = batman.TransitParams()
            transit_params.t0 = 0.
            transit_params.per = self.transit.period.to(u.h).value
            transit_params.rp = r_p
            transit_params.a = (
                    self.transit.semi_a /
                    self.transit.stellar_radius).decompose().value
            transit_params.inc = self.transit.inclination.to(u.deg).value
            transit_params.ecc = self.transit.eccentricity
            transit_params.w = self.transit.long_periastron.to(u.deg).value
            transit_params.limb_dark = "uniform"
            transit_params.u = []
            model = batman.TransitModel(transit_params, time)
            transit_f = model.light_curve(transit_params)
            y = transit_f * _linear_model(x, linear_theta)
            return y

        # Evaluate the model at the flux time stamps and calculate the negative
        # log-likelihood
        def _log_likelihood(theta):
            if jitter_model == 'linear' or jitter_model is None:
                f_model = _linear_model(independent_vars, theta)
            elif jitter_model == 'transit' or \
                    jitter_model == 'linear_and_transit':
                f_model = _linear_with_transit_model(independent_vars, theta)
            elif jitter_model == 'poly' or jitter_model == 'polynomial':
                f_model = _poly_model(independent_vars, theta)
            else:
                f_model = jitter_model(independent_vars, theta)

            # Finally compute the log-likelihood
            diff = flux_vector / norm - f_model
            if fit_to_fudge is True:
                return np.sum(diff ** 2)
            else:
                weight = flux_unc / norm * jitter_spread
                return np.sum(diff ** 2 / weight ** 2)

        # Fit the observed fluxes to the jitter model
        result = minimize(_log_likelihood, x0=first_guess, **kwargs)
        self.fit_result = result
        self.log_likelihood = self.fit_result.fun
        if fit_to_fudge is False and use_subexposures is True:
            n_data = len(self.lc.tt_time)
        elif fit_to_fudge is False and use_subexposures is False:
            n_data = len(self.lc.time)
        else:
            n_data = len(self.jitter_time)
        self.fit_bic = \
            np.log(n_data) * len(first_guess) - 2 * self.log_likelihood
        self.best_fit_params = self.fit_result.x

        # Compute the expected flux
        if jitter_model == 'linear' or jitter_model is None:
            self.expected_flux = _linear_model(independent_vars,
                                               self.best_fit_params) * norm
        elif jitter_model == 'transit' or \
                jitter_model == 'linear_and_transit':
            self.expected_flux = _linear_with_transit_model(
                independent_vars, self.best_fit_params) * norm
        elif jitter_model == 'poly' or jitter_model == 'polynomial':
            self.expected_flux = _poly_model(
                independent_vars, self.best_fit_params) * norm
        else:
            self.expected_flux = jitter_model(independent_vars,
                                              self.best_fit_params) * norm

    # Finally we implement a method to de-trend the data
    def detrend(self):
        """

        Returns:

        """
        if self.expected_flux is not None:

            # First check if the trend fit was done to the actual data and not
            # the fudge data
            len_exp_f = len(self.expected_flux)
            len_tt_f = len(self.lc.tt_integrated_flux)
            len_f = len(self.lc.integrated_flux)
            if len_exp_f != len_tt_f and len_exp_f != len_f:
                raise ValueError(
                    'You should fit the trend to the data (and not the fudge '
                    'data) to use this feature.')
            else:
                pass

            mean_expected_flux = np.mean(self.expected_flux)
            detrend_factor = self.expected_flux / mean_expected_flux
            self.detrend_factor = detrend_factor

            try:
                self.detrended_flux = self.lc.integrated_flux / detrend_factor
            except ValueError:
                self.detrended_flux = self.lc.tt_integrated_flux / \
                                      detrend_factor
