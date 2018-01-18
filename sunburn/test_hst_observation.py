from . import hst_observation

datasets = ['data/ld9m10ujq', 'data/ld9m10uyq']
visit1 = hst_observation.Visit(datasets, 'cos')
visit1.plot_spectra([1300, 1400])
int_flux, uncertainty = \
    visit1.orbit['data/ld9m10ujq'].integrated_flux([1300, 1400])
visit1.orbit['data/ld9m10ujq'].plot_spectrum(chip_index='red')
