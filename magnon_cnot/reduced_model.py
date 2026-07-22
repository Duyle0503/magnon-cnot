"""Reduced 4-mode ODE model and ensemble fidelity."""
import numpy as np
from .constants import T_self, S_cross, Gamma0, sig_th


def rhs_4m(s, K_max, Gam):
    """RHS of 4-mode ODE (single trajectory). s: (8,)"""
    Ap = s[0] + 1j * s[1]; Am = s[2] + 1j * s[3]
    Bp = s[4] + 1j * s[5]; Bm = s[6] + 1j * s[7]
    nA = np.abs(Ap)**2 + np.abs(Am)**2 + 1e-30
    Ke = K_max * np.clip(np.abs(Am)**2 / nA, 0, 1)
    dAp = (-1j * (T_self * np.abs(Ap)**2 + S_cross * np.abs(Am)**2) - Gam) * Ap
    dAm = (-1j * (T_self * np.abs(Am)**2 + S_cross * np.abs(Ap)**2) - Gam) * Am
    dBp = ((-1j * (T_self * np.abs(Bp)**2 + S_cross * np.abs(Bm)**2) - Gam) * Bp
           - 1j * Ke * Bm)
    dBm = ((-1j * (T_self * np.abs(Bm)**2 + S_cross * np.abs(Bp)**2) - Gam) * Bm
           - 1j * Ke * Bp)
    return np.array([dAp.real, dAp.imag, dAm.real, dAm.imag,
                     dBp.real, dBp.imag, dBm.real, dBm.imag])


def rk4_4m(s0, K_max, Gam, T_sim, Nt=300):
    """RK4 integrator for single trajectory."""
    dt = T_sim / Nt; s = s0.copy()
    for _ in range(Nt):
        k1 = rhs_4m(s, K_max, Gam)
        k2 = rhs_4m(s + .5 * dt * k1, K_max, Gam)
        k3 = rhs_4m(s + .5 * dt * k2, K_max, Gam)
        k4 = rhs_4m(s + dt * k3, K_max, Gam)
        s += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def fidelity_4m(K_max, Gam=Gamma0):
    """Average truth-table fidelity from 4-mode model (no renorm)."""
    cases = [('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')]
    pure  = {'0': (1 + 0j, 1e-3 + 0j), '1': (1e-3 + 0j, 1 + 0j)}
    T_s   = np.pi / (2 * max(K_max, 1.0))
    Fs = []
    for c, t, eB in cases:
        Ap0, Am0 = pure[c]; Bp0, Bm0 = pure[t]
        s0 = np.array([Ap0.real, Ap0.imag, Am0.real, Am0.imag,
                       Bp0.real, Bp0.imag, Bm0.real, Bm0.imag])
        sf = rk4_4m(s0, K_max, Gam, T_s)
        nBp = sf[4]**2 + sf[5]**2; nBm = sf[6]**2 + sf[7]**2
        Fs.append(nBp if eB == '0' else nBm)
    return float(np.mean(Fs))


# ── Batch (vectorised) ensemble ──────────────────────────────────────────────

def rhs_4m_batch(s, K_max, Gam):
    """Vectorised RHS over batch axis 0.  s: (N, 8)"""
    Ap = s[:, 0] + 1j * s[:, 1]; Am = s[:, 2] + 1j * s[:, 3]
    Bp = s[:, 4] + 1j * s[:, 5]; Bm = s[:, 6] + 1j * s[:, 7]
    nA = np.abs(Ap)**2 + np.abs(Am)**2 + 1e-30
    Ke = K_max * np.clip(np.abs(Am)**2 / nA, 0, 1)
    dAp = (-1j * (T_self * np.abs(Ap)**2 + S_cross * np.abs(Am)**2) - Gam) * Ap
    dAm = (-1j * (T_self * np.abs(Am)**2 + S_cross * np.abs(Ap)**2) - Gam) * Am
    dBp = ((-1j * (T_self * np.abs(Bp)**2 + S_cross * np.abs(Bm)**2) - Gam) * Bp
           - 1j * Ke * Bm)
    dBm = ((-1j * (T_self * np.abs(Bm)**2 + S_cross * np.abs(Bp)**2) - Gam) * Bm
           - 1j * Ke * Bp)
    return np.stack([dAp.real, dAp.imag, dAm.real, dAm.imag,
                     dBp.real, dBp.imag, dBm.real, dBm.imag], axis=1)


def propagate_batch(s0, K_max, Gam, T_sim, Nt=400):
    """RK4 integrator for batch of trajectories."""
    dt = T_sim / Nt; s = s0.copy()
    for _ in range(Nt):
        k1 = rhs_4m_batch(s, K_max, Gam)
        k2 = rhs_4m_batch(s + .5 * dt * k1, K_max, Gam)
        k3 = rhs_4m_batch(s + .5 * dt * k2, K_max, Gam)
        k4 = rhs_4m_batch(s + dt * k3, K_max, Gam)
        s += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def ensemble_F(K_max, Gam=Gamma0, N=500, noise=sig_th, Nt=400):
    """Ensemble-averaged truth-table fidelity (no renorm, absolute populations).

    Returns per-case means, per-case standard errors, the grand mean, and the
    mean of the four per-case standard errors.
    """
    pure  = {'0': (1 + 0j, 1e-3 + 0j), '1': (1e-3 + 0j, 1 + 0j)}
    cases = [('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')]
    T_s   = np.pi / (2 * K_max)
    rng   = np.random.default_rng(42)
    Fc, Fe = [], []
    for c, t, eB in cases:
        Ap0, Am0 = pure[c]; Bp0, Bm0 = pure[t]

        def noisy(a):
            r = rng.normal(0, noise, (N, 2))
            return complex(a) + r[:, 0] + 1j * r[:, 1]

        Ap = noisy(Ap0); Am = noisy(Am0); Bp = noisy(Bp0); Bm = noisy(Bm0)
        nA = np.sqrt(np.abs(Ap)**2 + np.abs(Am)**2 + 1e-30)
        nB = np.sqrt(np.abs(Bp)**2 + np.abs(Bm)**2 + 1e-30)
        Ap /= nA; Am /= nA; Bp /= nB; Bm /= nB
        s0 = np.stack([Ap.real, Ap.imag, Am.real, Am.imag,
                       Bp.real, Bp.imag, Bm.real, Bm.imag], axis=1)
        sf  = propagate_batch(s0, K_max, Gam, T_s, Nt)
        nBp = sf[:, 4]**2 + sf[:, 5]**2
        nBm = sf[:, 6]**2 + sf[:, 7]**2
        # absolute population (no renorm → damping penalises F)
        Ft = nBp if eB == '0' else nBm
        Fc.append(float(np.mean(Ft)))
        Fe.append(float(np.std(Ft) / np.sqrt(N)))
    return np.array(Fc), np.array(Fe), float(np.mean(Fc)), float(np.mean(Fe))
