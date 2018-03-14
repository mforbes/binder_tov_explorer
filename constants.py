from __future__ import division

import numpy as np

# Globals are bad but convenient!
c = 1.0
GeV = 1000.0
MeV = GeV/1000.0
keV = MeV/1000.0
eV = keV/1000.0
fm = 1.0
km = 1e18*fm
m = km/1000.0
cm = m/100.0
s = 299792458.0 * m/c

kg = eV/c**2/1.7826619069409169633e-36
#kg = eV/c**2/1.78266181e-36
amu = 1.660539040e-27*kg
g = kg/1000.0
N = kg * m/s**2
dyne = 1e-5*N
hbarc = 197.326972 * MeV * fm
hbar = hbarc / c

J = 6.2415091258832579265e+12*MeV
erg = 1e-7*J

alpha = 1.0/137.03599913815450977  # Frink
alpha = 0.0072973525664  # PDG and Frink
e2 = alpha * hbarc
M0 = 1.9891e30*kg   # Frink
M0 = 1.98848e30*kg  # PDG
G = 6.67408e-11*m**3/kg/s**2  # PDG and Frink
Rs = 2*G*M0/c**2
m_n = 0.939565379*GeV/c**2
m_p = 0.938272046*GeV/c**2
m_e = 0.510998910*MeV/c**2
m_mu = 105.6583715*MeV/c**2

Fe56_binding = 492253.892*keV  # Binding energy of 56Fe

aleph = 4*np.pi * Rs**3 * GeV/M0/c**2/fm**3

r_nn = (2.75 + 0.11j) * fm
a_nn = (-18.9 + 0.4j) * fm
a_np = (-23.7153 + 0.0043j) * fm  # Singlet (from Hackenberg)
r_np = (2.754 + (0.018+0.056)*1j) * fm
