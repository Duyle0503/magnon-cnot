"""Analytical helpers for the ideal benchmark and corrected WKB convention."""
import numpy as np
from .constants import omega0, hbar, D_zz, muB_g, Gamma0


def K_WKB(d, dB, *, omega_mode, omega_attempt):
    """Leading-order Hamiltonian coupling for a forbidden rectangular barrier.

    Parameters
    ----------
    d : float or array
        Barrier width [m].
    dB : float or array
        Longitudinal barrier-field increment [T].
    omega_mode : float
        Confined mode energy in angular-frequency units [rad/s].
    omega_attempt : float
        Attempt frequency extracted from the confinement level spacing [rad/s].

    Returns
    -------
    float or array
        Off-diagonal Hamiltonian coupling ``K`` [rad/s]. The associated
        symmetric/antisymmetric splitting is ``2*K``.

    Notes
    -----
    This follows the revised convention
    ``K = (omega_attempt / (2*pi)) * exp(-S)`` with
    ``S = d*sqrt((gamma*dB - omega_mode)/D_zz)``. It is valid only when the
    specified region is forbidden. It does not infer the attempt prefactor
    from the condensate carrier frequency.
    """
    d = np.asarray(d, dtype=float)
    dB = np.asarray(dB, dtype=float)
    forbidden_offset = 2.0 * np.pi * muB_g * dB - float(omega_mode)
    if np.any(forbidden_offset <= 0.0):
        raise ValueError("WKB expression requires a positive forbidden-region offset")
    action = d * np.sqrt(forbidden_offset / D_zz)
    return (float(omega_attempt) / (2.0 * np.pi)) * np.exp(-action)


def legacy_K_WKB_proxy(d, dB):
    """Historical carrier-prefactor proxy retained only for legacy figures.

    The revised manuscript does not use this map as geometry-resolved evidence.
    New calculations must use :func:`K_WKB` with confinement-derived
    ``omega_mode`` and ``omega_attempt``.
    """
    V0 = 2.0 * np.pi * muB_g * dB * hbar
    kap = np.sqrt(np.maximum(V0, 1e-30) / (hbar * D_zz))
    return (omega0 / np.pi) * np.exp(-kap * d)


def F_loss_exact(K):
    """Loss-only population ceiling for an otherwise perfect half-Rabi flip.

    This is ``exp(-pi*Gamma/K)`` for ``tau_g = pi/(2K)``.  It is not a lower
    bound on the total gate fidelity: coherent and implementation errors can
    reduce the target-state projection further.
    """
    return np.exp(-np.pi * Gamma0 / np.maximum(K, 1.0))


def F_loss_linear(K):
    """First-order large-``K`` approximation to :func:`F_loss_exact`."""
    return 1.0 - np.pi * Gamma0 / np.maximum(K, 1.0)


def F_bound(K):
    """Backward-compatible alias for the exact loss-only ceiling.

    The historical function name is retained for scripts that imported it,
    but the returned quantity is no longer described as a fidelity lower bound.
    """
    return F_loss_exact(K)


def tau_gate(K):
    """Half-Rabi gate time  tau = pi/(2K)  [seconds]."""
    return np.pi / (2.0 * np.maximum(K, 1.0))


# Ideal prescribed-coupling benchmark; not extracted from the reported geometry.
K_star = np.pi * Gamma0 / (-np.log(0.99))  # exact K for F_loss>=0.99
K_demo = 2.0 * np.pi * 250e6              # assigned ideal benchmark
T_demo = tau_gate(K_demo) * 1e9            # gate time [ns]
