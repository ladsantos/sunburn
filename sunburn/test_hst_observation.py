import numpy as np
from . import hst_observation, spectroscopy

datasets = ['ld9m10ujq', 'ld9m10uyq']
visit1 = hst_observation.Visit(datasets, 'cos', prefix='data/')

line_list = spectroscopy.COSFUVLineList(wavelength_shift=.0,
                                        range_factor=1.0).lines

tr = 'Si III'
line = 0
ref_wl = line_list[tr][line].central_wavelength

shift = np.array([-23.623059845212502, -23.37932521889708])

orbit_list = [visit1.orbit['ld9m10ujq'], visit1.orbit['ld9m10uyq']]
test = hst_observation.CombinedSpectrum(
    orbit_list, ref_wl, 'cos', velocity_range=[-100, 100], doppler_corr=shift)
