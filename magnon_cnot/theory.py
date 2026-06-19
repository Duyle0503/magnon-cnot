"""Analytical theory: WKB coupling, fidelity bound, gate time."""
import numpy as np
from .constants import omega0, hbar, D_zz, muB_g, Gamma0


def K_WKB(d, dB):
    """Josephson coupling from WKB tunnelling.

    Parameters
    ----------
    d  : float or array  — barrier width [m]
    dB : float or array  — bias field [T]

    Returns
    -------
    K_max : float or array — coupling rate [rad/s]
    """
    V0  = 2.0 * np.pi * muB_g * dB * hbar
    kap = np.sqrt(np.maximum(V0, 1e-30) / (hbar * D_zz))
    return (omega0 / np.pi) * np.exp(-kap * d)


def F_bound(K):
    """Population fidelity bound  F >= 1 - pi*Gamma/K.

    Derived from exp(-2*Gamma*tau_g) where tau_g = pi/(2K).
    """
    return 1.0 - np.pi * Gamma0 / np.maximum(K, 1.0)


def tau_gate(K):
    """Half-Rabi gate time  tau = pi/(2K)  [seconds]."""
    return np.pi / (2.0 * np.maximum(K, 1.0))


# Key operating points
K_star = np.pi * Gamma0 / 0.01             # K for F>=0.99
K_demo = 2.0 * np.pi * 250e6              # demonstration operating point
T_demo = tau_gate(K_demo) * 1e9            # gate time [ns]
