"""Derivatives of the neutron-star `observables` `M`,
        `R`, and `k_2` as functions of fixed central pressures `Pc` (chosen so
        that the masses are equally space) and the EoS parameters `param`.  The
        derivatives are scaled by the observable values and the parameter
        values so as to be dimensionless.

------------------------------------------------------
WARNING: This is a generated dataset - edit with care.

Generated on: 2018-03-13 15:00:36.462626

Binary data stored in the NPY format described here:

https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
Mercural Id (hg id): 736756d971b5+
"""
import numpy as np

from ._classes import _Array, _Coords, _Params


_dims = dict(
    observables=_Params(
        coords=['M', 'R', 'k2'],
        units=['M0', 'km', '1']),
    masses=_Coords(
        coords=np.linspace(0.9, 2.09, 120),
        unit='M0'),
    params=_Params(
        coords=[u'sigma_delta', u'C_C', u'd_C', u'n_0', u'e_0', u'K_0', u'S_2',
                u'L_2', u'K_2', u'a', u'alpha', u'b', u'beta', u'mu_p0', u'u_p',
                u'm_eff_m_p', u'E_c', u'E_max', u'C_max'],
        units=['MeV/fm**2', '1', '1', '1./fm**3', 'MeV', 'MeV', 'MeV', 'MeV',
               'MeV', 'MeV', '1', 'MeV', '1', 'MeV', '1', '1', 'MeV', 'MeV',
               '1'],
        values=[1.382, 0.8957, 3.0, 0.16, -16.0, 240.0, 32.0, 60.0, 30.0,
                14.4383200971, 0.56302798373, 1.86063464319, 2.71424708856,
                -104.5, 3.136, 0.8, 651.439985687, 1172.0729033599998, 1.0]))


k2 = _Array(
    'k2.npy',
    dims=['masses'],
    unit='1',
    _dims=_dims)


R = _Array(
    'R.npy',
    dims=['masses'],
    unit='km',
    _dims=_dims)


derivatives = _Array(
    'derivatives.npy',
    dims=['masses', 'observables', 'params'],
    unit='1',
    _dims=_dims)


Pc = _Array(
    'Pc.npy',
    dims=['masses'],
    unit='MeV/fm**3',
    _dims=_dims)


del _Array, _Coords, _Params, np
