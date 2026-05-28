"""Physical constants and YIG material parameters."""
import numpy as np

# Fundamental constants
hbar     = 1.054571817e-34       # J·s
kB       = 1.380649e-23          # J/K
muB_g    = 1.4e10                # g·μB/h  [Hz/T]

# YIG parameters
Ms       = 1.4e5                 # Saturation magnetisation [A/m]
alpha_G  = 2.0e-4                # Gilbert damping
omega0   = 2.0 * np.pi * 3.75e9 # BEC frequency [rad/s]
q_BEC    = 1.8e7                 # BEC wave-vector [rad/m]
D_zz     = 3.0e-4               # Dispersion coefficient [m²/s]
T_self   = 1.0e-10              # GP self-interaction [m/s]
S_cross  = 0.7e-10              # GP cross-interaction [m/s]
Gamma0   = alpha_G * omega0     # Gilbert relaxation [rad/s]
T_K      = 300.0                # Temperature [K]
N_magnon = 1e10                 # Magnon number for thermal noise

# Derived
sig_th   = float(np.sqrt(kB * T_K / (hbar * omega0 * N_magnon)))
