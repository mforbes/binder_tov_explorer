"""Fisher Information Matrix for the ALF4 EoS with neutron star
binaries at at fudicial distance of 40MpC assuming aLIGO detector
sensitivity.

Derivatives of the neutron-star `observables` `M`, `R`, and `k2` as
functions of fixed central pressures `Pc` (chosen so that the masses
are equally space) and the EoS parameters `param`.  The derivatives
are scaled by the observable values and the parameter values so as to
be dimensionless.

------------------------------------------------------
WARNING: This is a generated dataset - edit with care.

Generated on: 2018-03-13 09:27:52.917596

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
        coords=np.linspace(0.9, 1.94, 105),
        unit='M0'),
    params=_Params(
        coords=['sigma_delta', 'C_C', 'n_0', 'e_0', 'K_0', 'S_2', 'L_2', 'K_2',
                'a', 'alpha', 'b', 'beta', 'mu_p0', 'u_p', 'm_eff_m_p', 'E_c',
                'E_max', 'C_max'],
        units=['MeV/fm**2', '1', '1./fm**3', 'MeV', 'MeV', 'MeV', 'MeV', 'MeV',
               'MeV', '1', 'MeV', '1', 'MeV', '1', '1', 'MeV', 'MeV', '1'],
        values=[1.382, 0.8957, 0.16, -16.0, 240.0, 32.0, 60.0, 30.0, 13.0, 0.5,
                3.31399183, 2.43134698, -104.5, 3.136, 0.8, 631.16621272,
                647.1383376, 1.0]))


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
