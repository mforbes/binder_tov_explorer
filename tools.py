from __future__ import division

from collections import namedtuple, OrderedDict
import numpy as np
import scipy.interpolate
import scipy.stats

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import ipywidgets as widgets

import ALF4
import APR4
import constants
from uncertainties import ufloat

sp = scipy

datasets = dict(
    ALF4=ALF4,
    APR4=APR4
)


def Lambda(M, R, k_2):
    """Return the (dimensionless) parameter Lambda computed from the MRks."""
    return (2./3.)*k_2*(R/M * constants.c**2/constants.G)**5


class Data(object):
    """Container for loading and manipulating the data."""
    def __init__(self, key='ALF4'):
        self.dataset = datasets[key]

        # This function interpolates from a given mass to the index of the
        # nearest tabulated mass.
        _args = dict(kind='nearest', bounds_error=False, fill_value='extrapolate')
        self._index = sp.interpolate.interp1d(self.M/constants.M0,
                                              np.arange(len(self.M)),
                                              **_args)

        def _Lambda(m1, R1, k1, m2=None, R2=None, k2=None):
            return Lambda(M=m1, R=R1, k_2=k1)

        self.Lambda = Lambda(self.M, self.R, self.k2)
        self.dLambda = np.array(
            [self.get_derivative(_Lambda, _i, _i) for _i in range(len(self.M))])

    def index(self, M_M0):
        """Return the index of the closes tabulated star with mass M_M0 in
        units of M0."""
        return int(self._index(M_M0))
        
    # Curated list of the most important parameters
    _significant_parameters = ['a', 'alpha', 'b', 'beta',
                               'mu_p0', 'u_p',
                               'E_max', 'E_c', 'C_max',
                               'n_0', 'e_0', 'K_0',
                               'S_2', 'L_2', 'K_2',
                               ]
    
    @property
    def param_names(self):
        return self.dataset._dims['params'].coords

    @property
    def params(self):
        """Return the parameters in numerical units defined in constants"""
        params_ = self.dataset._dims['params']
        return namedtuple('Params', self.param_names)(
            *self._evalu(params_.values, params_.units))

    @property
    def M(self):
        """Return the masses in numerical units defined in constants"""
        masses_ = self.dataset._dims['masses']
        return self._evalu(masses_.coords, masses_.unit)

    @property
    def R(self):
        """Return the radii in numerical units defined in constants"""
        Rs_ = self.dataset.R
        return self._evalu(Rs_.data, Rs_.unit)
    
    @property
    def k2(self):
        """Return the Love numbers"""
        k2s_ = self.dataset.k2
        return self._evalu(k2s_.data, k2s_.unit)

    @property
    def derivatives(self):
        """Return the derivatives.  Note: internally all derivatives are stored
        in a dimensionless format normalized by the observable value and the
        parameter values.
        """
        observable_names = self.dataset._dims['observables'].coords
        Derivatives = namedtuple('Derivatives', observable_names)
        return Derivatives(*[
            self.dataset.derivatives.data.real[:, _i, :]
            * getattr(self, _o)[:, None]
            / np.asarray(self.params)[None, :]
            for _i, _o in enumerate(observable_names)])

    def get_derivative(self, f, i1, i2):
        """Return the derivatives of f(m1, m2, r1, r2, k1, k2) as a function of
        the various parameters.
        """
        M, R, k2 = self.M, self.R, self.k2

        # Here we use the uncertainties package to automatically compute the
        # derivatives
        ms = ufloat(M[i1], 0), ufloat(M[i2], 0)
        Rs = ufloat(R[i1], 0), ufloat(R[i2], 0)
        ks = ufloat(k2[i1], 0), ufloat(k2[i2], 0)

        f_ = f(m1=ms[0], m2=ms[1], R1=Rs[0], R2=Rs[1], k1=ks[0], k2=ks[1])
        # Add zero so all derivatives exist even if f does not use a parameter
        f_ = f_ + 0*(ms[0] + ms[1] + Rs[0] + Rs[1] + ks[0] + ks[1])

        # derivatives with dimensions [m1, r1, k1, m2, r2, k2], params
        ders = np.asarray(self.derivatives)
        derivatives = np.concatenate([ders[:, i1], ders[:, i2]], axis=0)
        
        return sum(f_.derivatives[_p]*derivatives[_i, :]
                   for _i, _p
                   in enumerate((ms[0], Rs[0], ks[0], ms[1], Rs[1], ks[1])))

    @staticmethod
    @np.vectorize
    def _evalu(value, unit):
        """Evaluate the value with the specified unit."""
        return value * eval(unit, {}, constants.__dict__)

    def explore_parameters(self):
        # Get a list of names with significant names first.
        ps = OrderedDict()
        for _p in self._significant_parameters:
            if _p in self.param_names:
                ps[_p] = 1

        params = self.dataset._dims['params']
        
        for _p, _v, _u in zip(params.coords, params.values, params.units):
            if _u == '1':
                label = "{} ({:.4g})".format(_p, _v)
            else:
                label = "{} ({:.4g} {})".format(_p, _v, _u)
            ps[_p] = label

        ps = [ps[_p] for _p in ps]

        widgets.interact(
            p=widgets.Select(value=ps[0], options=ps),
            dp=widgets.IntSlider(10, 0, 100, description="dp (%)"),
        )(lambda p, dp: self._draw_MR(p, dp))

    def _draw_MR(self, p, dp=10, subplot_spec=None):
        """Draw the MR curve with dp % variation in the parameter p."""
        u = constants
        p = p.split()[0]
        parameter_variation = (p, dp/100.0)

        if subplot_spec:
            gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=subplot_spec)
        else:
            gs = GridSpec(1, 2)
            plt.figure(figsize=(10, 5))
            
        plt.subplot(gs[0])
        derivatives = self.derivatives._asdict()
        ip = self.param_names.index(parameter_variation[0])
        dp_ = parameter_variation[1]*self.params[ip]
        dR = derivatives['R'][:, ip] * dp_
        dM = derivatives['M'][:, ip] * dp_
        plt.plot(self.R/u.km, self.M/u.M0)
        error_ellipse(self.R/u.km, self.M/u.M0, xerr=dR/u.km, yerr=dM/u.M0,
                      alpha=0.2)
        plt.xlabel('R [km]')
        plt.ylabel('M [solar_mass]')

        plt.subplot(gs[1])
        dLambda = self.dLambda[:, ip] * dp
        plt.plot(self.Lambda, self.M/u.M0)
        error_ellipse(self.Lambda, self.M/u.M0, xerr=dLambda, yerr=dM/u.M0,
                      alpha=0.2)
        plt.xlabel('Lambda')
        plt.ylabel('M [solar_mass]')
        
    def _draw_bar(self, v, subplot_spec=None):
        """Vertical barplot of the eigen-vector"""
        v = v/np.linalg.norm(v)
        v /= np.sign(v[np.argmax(abs(v))])
        Np = len(self.params)
        ax = plt.subplot(subplot_spec)
        plt.sca(ax)
        ax.grid(True)
        ax.barh(np.arange(Np), v)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, Np-0.5)
        plt.xticks([-0.5, 0, 0.5], rotation='vertical')
        plt.axvline([0], c='k', lw=0.5, ls=':')
        # plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='y', direction='inout')
        plt.yticks(np.arange(Np), self.param_names)
        return ax


class PCA_Explorer(object):
    """Principle Component Analysis of the Fisher Information Matrix data.

    Attributes
    ----------
    FIM_GW : Array
      This should be a (Nm1, Nm2, Np, Np) dimensional array object containing
      the (Np, Np) Fisher Information Matrices for each of the Np parameters.
      Auxiliary data should be stored in the dimensions.

      FIM_GW.dims[0] == FIM_GW.dims[1] : Coords
         Nm mass dimension.
      FIM_GW.dims[2] == FIM_GW.dims[3] : Params
         Np parameters.
    distance_Mpc : float
       Fudicial distance in Mpc.
    """
    
    def __init__(self, data, distance_Mpc=40.0):
        self.data = data
        self.distance_Mpc = distance_Mpc

        self.masses = self.FIM_GW.dims[0]
        assert self.masses is self.FIM_GW.dims[1]
        self.params = self.FIM_GW.dims[2]
        assert self.params is self.FIM_GW.dims[3]

        # These functions interpolate from a given mass to the index of the
        # nearest tabulated mass.  In the future, we might like to interpolate
        # the actual FIM.
        _args = dict(kind='nearest', bounds_error=False, fill_value='extrapolate')
        self._index = sp.interpolate.interp1d(
            self.masses.coords, np.arange(len(self.masses.coords)), **_args)
        self.Np = len(self.params.coords)

    @property
    def FIM_GW(self):
        return self.data.dataset.F
    
    def get_FIM_GW(self, m1, m2, distance_Mpc=40.0):
        """Return the closes FIM for a binary system with the specified
        masses and distance.
        """
        return ((self.distance_Mpc/distance_Mpc)**2
                * self.FIM_GW.data[int(self._index(m1)),
                                   int(self._index(m2))])

    def get_PCA(self):
        """Return the full PCA for all tabulated mass pairs."""
        Nm = len(self.masses.coords)

        # Make some empty 2D "lists"
        ds = [[None]*Nm for _i in range(Nm)]
        Us = [[None]*Nm for _i in range(Nm)]
        for i1 in range(Nm):
            for i2 in range(i1, Nm):
                m1 = self.masses.coords[i1]
                m2 = self.masses.coords[i2]
                FIM = self.get_FIM_GW(m1=m1, m2=m2)
                assert np.allclose(FIM, FIM.T.conj())
                d, U = np.linalg.eigh(FIM)
                ds[i2][i1] = ds[i1][i2] = d
                Us[i2][i1] = Us[i1][i2] = U
        return np.array(ds), np.array(Us)

    def get_FIM(self, m1_m2_d):
        """Return the full Fisher information matrix for a given population.

        Arguments
        ---------
        m1_m2_d : [(m1, m2, d)]
           List of binaries in sample population.  Here m1, m2, and d should be
           the masses (M0) and distances (Mpc) of each pair.
        """
        FIM = 0
        for m1, m2, d in m1_m2_d:
            FIM = FIM + self.get_FIM_GW(m1=m1, m2=m2, distance_Mpc=d)
        return FIM

    def plot_PCA(self, m1_m2_d, significance=50.0, plot_samples=True):
        """Plot the principal component analysis.

        Arguments
        ---------
        m1_m2_d : [(m1, m2, d)]
           List of binaries in sample population.  Here m1, m2, and d should be
           the masses (M0) and distances (Mpc) of each pair.
        significance : float
           Only the components constrained to within this percent are shown.
           If this is 50.0, then the linear combination is constrained to
           +-50%.
        """
        
        m1_m2_d = np.asarray(m1_m2_d)
        if plot_samples:
            plt.scatter(m1_m2_d[:, 0], m1_m2_d[:, 1], c=m1_m2_d[:, 2])
            plt.colorbar(label='Distance [Mpc]')
            plt.gca().set_aspect(1)
            plt.xlabel('mass 1 [solar mass]')
            plt.ylabel('mass 2 [solar mass]')
            display(plt.gcf())
        
        params = self.params
        Np = len(params.coords)

        FIM = self.get_FIM(m1_m2_d)
        d, U = np.linalg.eigh(FIM)
        Npc = sum(100./np.sqrt(abs(d)+1e-32) < significance)
        print("{} principal component(s) better than {:.0f}%."
              .format(Npc, significance)
              +
              " (Next component constrained at {:.0f}%)"
              .format(100./np.sqrt(d[-1-Npc])))

        fig = plt.figure(figsize=(max(5, Npc), 5))
        plt.rc('grid', ls='-', lw=1.0, c='WhiteSmoke')

        gs = GridSpec(1, Npc, wspace=0)
        axs = []
        for n in range(Npc):
            ax = self.data._draw_bar(U[:, -n-1], subplot_spec=gs[n])
            axs.append(ax)
            plt.title(r"$\sigma$ {:.2g}%".format(100./np.sqrt(d[-n-1])))
            plt.setp(ax.get_yticklabels(), visible=False)

        axs[0].yaxis.tick_left()
        axs[-1].yaxis.tick_right()
        for ax in [axs[0], axs[-1]]:
            for label in ax.get_ymajorticklabels():
                label.set_visible(True)
        plt.tight_layout()
        plt.close('all')
        return fig


class PopulationModel(object):
    """Population model for binary neutron stars."""
    def __init__(self, m1, m2, distance, constant_distance=False):
        """
        Arguments
        ---------
        m1 : complex
           Mass 1: Gaussian with mean=real and std=imag (M0)
        m2 : complex
           Mass 2: Gaussian with mean=real and std=imag (M0)
        distance : float
           Uniform distribution within a band of specified radius.
        constant_distance:
           If True, then all events are at a fixed distance.
        """
        self.m1 = sp.stats.norm(loc=m1.real, scale=m1.imag).rvs
        self.m2 = sp.stats.norm(loc=m2.real, scale=m2.imag).rvs
        self._distance = distance
        self.constant_distance = constant_distance

    def distance(self, N):
        if self.constant_distance:
            return [self._distance] * N
        
        dist = []
        d_max = self._distance.real
        while len(dist) < N:
            x, y, z = (np.random.random(3)*2 - 1)*d_max/2.0
            d = np.sqrt(x**2 + y**2 + z**2)
            if (d <= d_max):
                dist.append(d)
        return dist
    
    def get_samples(self, N_observations=100):
        m1s = self.m1(N_observations)
        m2s = self.m2(N_observations)
        distances = self.distance(N_observations)
        return list(zip(m1s, m2s, distances))


def error_ellipse(x, y, xerr, yerr, alpha=0.2, **kw):
    """Like errorbar, but uses ellipses"""
    from matplotlib.collections import EllipseCollection
    ax = plt.gca()
    ec = EllipseCollection(xerr, yerr, 0, units='xy',
                           offsets=np.asarray([x, y]).T,
                           transOffset=ax.transData, alpha=alpha,
                           **kw)
    ax.add_collection(ec)
    return ec
