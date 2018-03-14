from . import analysis, hst_observation, spectroscopy
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt


period = 2.64389803 * u.d
transit_midpoint = Time(2454865.084034, format='jd')
duration = 0.04189 * u.d

t = analysis.Transit(planet_name='GJ 436 b', period=period,
                     transit_midpoint=transit_midpoint, duration14=duration)

visit1 = hst_observation.Visit(['data/ld9m10ujq', 'data/ld9m10uyq'], 'cos')
#visit2 = hst_observation.Visit(['data/ld9mg2nlq', 'data/ld9mg2ntq'], 'cos')

line_list = spectroscopy.COSFUVLineList().lines

lc1 = analysis.LightCurve(visit1, t, line_list)
#lc2 = analysis.LightCurve(visit2, t, line_list)
lc1.compute_flux(transition='C III', line_index=0)
lc1.plot(symbol_color='C0', fold=True, label_choice='Visit 1 (19-Nov-2017)',
         norm_factor=1.0)
plt.show()
