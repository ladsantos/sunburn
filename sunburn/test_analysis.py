from . import analysis, hst_observation, spectroscopy
from astropy.time import Time
import astropy.units as u


period = 2.64389803 * u.d
transit_midpoint = Time(2454865.084034, format='jd')
duration = 0.04189 * u.d

t = analysis.Transit(planet_name='GJ 436 b', period=period,
                     transit_midpoint=transit_midpoint, duration14=duration)

visit = hst_observation.Visit(['data/ld9m10ujq', 'data/ld9m10uyq'], 'cos')

line_list = spectroscopy.COSFUVLineList().lines

lc = analysis.LightCurve(visit, t, line_list)
lc.compute_flux(transition='C III', line_index=0)
lc.plot()
