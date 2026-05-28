"""Gross-Pitaevskii equation solver: split-step Fourier method."""
import math
import numpy as np

from .constants import D_zz, T_self, S_cross, Gamma0, kB, T_K, hbar, omega0
from .grid import xp, to_np, z_xp, kz_xp, z_um, z_np, dz, Nz, L_um, mask_A, mask_B


def make_potential(d_bar=1e-6, dB_bar=5e-3):
    """Double-well trapping potential V(z) [rad/s]."""
    omega_trap = 2.0 * np.pi * 50e6
    z_well     = (L_um * 1e-6) / 2.0
    delta2     = xp.minimum((z_xp - z_well)**2, (z_xp + z_well)**2)
    V          = (omega_trap**2 / (4.0 * D_zz)) * delta2
    muB_g      = 1.4e10
    Omega_bar  = 2.0 * np.pi * muB_g * dB_bar
    sigma      = dz * 2
    bar = 0.5 * (xp.tanh((z_xp + d_bar / 2) / sigma)
                 - xp.tanh((z_xp - d_bar / 2) / sigma))
    return V + Omega_bar * bar


def gauss(center_um, sigma_um=3.0):
    """Normalised Gaussian wavepacket on z_xp."""
    g = xp.exp(-((z_xp - center_um * 1e-6)**2) / (2 * (sigma_um * 1e-6)**2))
    return g / xp.sqrt(xp.sum(xp.abs(g)**2) * dz)


def init_state(ctrl, tgt):
    """Build (Ap, Am, Bp, Bm) initial wavefunctions from qubit labels '0'/'1'."""
    cA, cB = -L_um / 2, +L_um / 2
    eps = 1e-3
    mk = {'0': (gauss(cA), eps * gauss(cA)),
           '1': (eps * gauss(cA), gauss(cA))}
    mb = {'0': (gauss(cB), eps * gauss(cB)),
           '1': (eps * gauss(cB), gauss(cB))}
    Ap, Am = mk[ctrl]
    Bp, Bm = mb[tgt]
    return Ap, Am, Bp, Bm


def renorm(Pp, Pm, mask):
    """Renormalise qubit population (models continuous parametric pumping)."""
    n = xp.sqrt(xp.sum((xp.abs(Pp)**2 + xp.abs(Pm)**2) * mask) * dz + 1e-30)
    return Pp / n, Pm / n


def gpe_step(psi, V, K_max, dt, noise_amp=0.0, do_renorm=True):
    """One Strang-split step for the 4-component GPE.

    Parameters
    ----------
    psi      : tuple (Ap, Am, Bp, Bm)
    V        : potential array
    K_max    : maximum Josephson coupling [rad/s]
    dt       : time step [s]
    noise_amp: Langevin noise amplitude
    do_renorm: if True, renormalise after each step (pumped model)

    Returns
    -------
    (Ap, Am, Bp, Bm, K_eff)
    """
    Ap, Am, Bp, Bm = psi
    kin = xp.exp(-1j * D_zz * kz_xp**2 * dt * 0.5)

    # half kinetic
    Ap = xp.fft.ifft(kin * xp.fft.fft(Ap))
    Am = xp.fft.ifft(kin * xp.fft.fft(Am))
    Bp = xp.fft.ifft(kin * xp.fft.fft(Bp))
    Bm = xp.fft.ifft(kin * xp.fft.fft(Bm))

    # nonlinear + damping
    nAp = xp.abs(Ap)**2; nAm = xp.abs(Am)**2
    nBp = xp.abs(Bp)**2; nBm = xp.abs(Bm)**2
    Ap *= xp.exp((-1j * (V + T_self * nAp + S_cross * nAm) - Gamma0) * dt)
    Am *= xp.exp((-1j * (V + T_self * nAm + S_cross * nAp) - Gamma0) * dt)
    Bp *= xp.exp((-1j * (V + T_self * nBp + S_cross * nBm) - Gamma0) * dt)
    Bm *= xp.exp((-1j * (V + T_self * nBm + S_cross * nBp) - Gamma0) * dt)

    # conditional Josephson: K_eff from integrated control population
    NA_p = float(to_np(xp.sum(nAp * mask_A))) * dz
    NA_m = float(to_np(xp.sum(nAm * mask_A))) * dz
    K_eff = K_max * NA_m / (NA_p + NA_m + 1e-30)
    if K_eff > 0.0:
        cB = math.cos(K_eff * dt); sB = math.sin(K_eff * dt)
        Bp, Bm = cB * Bp - 1j * sB * Bm, cB * Bm - 1j * sB * Bp

    # half kinetic
    Ap = xp.fft.ifft(kin * xp.fft.fft(Ap))
    Am = xp.fft.ifft(kin * xp.fft.fft(Am))
    Bp = xp.fft.ifft(kin * xp.fft.fft(Bp))
    Bm = xp.fft.ifft(kin * xp.fft.fft(Bm))

    # Langevin noise
    if noise_amp > 0.0:
        sh = Ap.shape
        for arr in [Ap, Am, Bp, Bm]:
            arr += noise_amp * (xp.random.normal(0, 1, sh)
                                + 1j * xp.random.normal(0, 1, sh))

    # renormalise (continuous pumping)
    if do_renorm:
        Ap, Am = renorm(Ap, Am, mask_A)
        Bp, Bm = renorm(Bp, Bm, mask_B)

    return Ap, Am, Bp, Bm, K_eff


def run_sim(ctrl, tgt, K_max, T_ns, dt_ns=0.005, n_save=200,
            noise=True, do_renorm=True):
    """Run full GPE simulation for one truth-table case.

    Returns dict with keys: t_ns, z_um, N_Ap, N_Am, N_Bp, N_Bm, phi_B
    """
    dt     = dt_ns * 1e-9
    Nt     = int(T_ns / dt_ns)
    save_e = max(1, Nt // n_save)
    V      = make_potential()
    n_amp  = (float(np.sqrt(Gamma0 * kB * T_K / (hbar * omega0))) * 1e-7
              * np.sqrt(dt)) if noise else 0.0

    Ap, Am, Bp, Bm = init_state(ctrl, tgt)
    t_s = []; NAp_s = []; NAm_s = []; NBp_s = []; NBm_s = []; phi_s = []

    for it in range(Nt + 1):
        if it % save_e == 0:
            t_s.append(it * dt_ns)
            NAp_s.append(to_np(xp.abs(Ap)**2).copy())
            NAm_s.append(to_np(xp.abs(Am)**2).copy())
            NBp_s.append(to_np(xp.abs(Bp)**2).copy())
            NBm_s.append(to_np(xp.abs(Bm)**2).copy())
            jB = int(Nz / 2 + Nz / 4)
            phi_s.append(float(to_np(xp.angle(Bp[jB] * xp.conj(Bm[jB])))))
        if it == Nt:
            break
        Ap, Am, Bp, Bm, _ = gpe_step(
            (Ap, Am, Bp, Bm), V, K_max, dt, n_amp, do_renorm=do_renorm)

    return dict(t_ns=np.array(t_s), z_um=z_um,
                N_Ap=np.array(NAp_s), N_Am=np.array(NAm_s),
                N_Bp=np.array(NBp_s), N_Bm=np.array(NBm_s),
                phi_B=np.array(phi_s))


def integ(sim, region):
    """Spatially-integrated, normalised populations for qubit A or B."""
    mask = (sim['z_um'] < 0) if region == 'A' else (sim['z_um'] > 0)
    key  = 'A' if region == 'A' else 'B'
    Np = sim[f'N_{key}p'][:, mask].sum(1) * dz * 1e6
    Nm = sim[f'N_{key}m'][:, mask].sum(1) * dz * 1e6
    tot = Np + Nm + 1e-30
    return Np / tot, Nm / tot
