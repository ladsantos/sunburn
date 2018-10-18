import numpy as np
import astropy.units as u

from . import hst_observation
from os.path import isdir, isfile
from shutil import rmtree

datasets = ['ld9m10ujq', 'ld9m10uyq']
visit1 = hst_observation.Visit(datasets, 'cos', prefix='data/')
visit1.plot_spectra([1300, 1400])

v_range = np.array([-50., 50.]) * u.km / u.s
ref_wl = 1206.5

int_flux, uncertainty = \
    visit1.orbit['ld9m10ujq'].integrated_flux(velocity_range=v_range,
                                              reference_wl=ref_wl,
                                              rv_correction=-20 * u.km / u.s)
visit1.orbit['ld9m10ujq'].plot_spectrum(chip_index='red')
