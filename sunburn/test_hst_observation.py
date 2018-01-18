from . import hst_observation

datasets = ['ld9m10ujq', 'ld9m10uyq']
visit1 = hst_observation.Visit(datasets, 'cos', data_folder='data/')
visit1.plot_spectra([1300, 1400])
int_flux, uncertainty = \
    visit1.orbit['ld9m10ujq'].integrated_flux([1300, 1400])
visit1.orbit['ld9m10ujq'].plot_spectrum(chip_index='red')
