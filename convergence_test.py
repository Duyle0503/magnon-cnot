"""
Convergence test for the legacy ideal prescribed-coupling spatial GPE.

Self-contained: re-implements the second-order split-step Fourier solver with
*parameterised* grid size Nz and time step dt (the package pins these at import
time), runs the four truth-table cases at increasing resolution, and reports the
maximum change in the output populations. This checks only the numerical
discretisation of the assigned-coupling truth-table benchmark; it is not
geometry-resolved convergence or CNOT device validation.

Run:  python convergence_test.py
GPU:  uses CuPy automatically if available, else NumPy.
"""
import numpy as np

try:
    import cupy as xp
    _GPU = True
except ImportError:
    xp = np
    _GPU = False


def to_np(a):
    return xp.asnumpy(a) if (_GPU and isinstance(a, xp.ndarray)) else np.asarray(a)


# ── Physical parameters (mirror magnon_cnot/constants.py) ─────────────────────
hbar    = 1.054571817e-34
omega0  = 2.0 * np.pi * 3.75e9        # BEC frequency [rad/s]
alpha_G = 2.0e-4
Gamma0  = alpha_G * omega0            # Gilbert relaxation [rad/s]
D_zz    = 3.0e-4                      # dispersion [m^2/s]
T_self  = 1.0e-10
S_cross = 0.7e-10
muB_g   = 1.4e10                      # g*muB/h [Hz/T]
L_um    = 20.0                        # half-domain marker (wells at +/- L_um/2)
Lphys   = 2.0 * L_um * 1e-6           # 40 um domain

K_demo  = 2.0 * np.pi * 250e6         # assigned ideal coupling [rad/s]
TAU_G   = np.pi / (2.0 * K_demo)      # gate time [s]


def build_grid(Nz):
    dz   = Lphys / Nz
    z    = (np.arange(Nz) - Nz / 2) * dz
    kz   = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)
    maskA = (z < 0).astype(float)
    maskB = (z > 0).astype(float)
    return (dz, xp.asarray(z), xp.asarray(kz),
            xp.asarray(maskA), xp.asarray(maskB))


def potential(z_xp, dz, d_bar=1.7e-6, dB_bar=3e-3):
    omega_trap = 2.0 * np.pi * 50e6
    z_well = (L_um * 1e-6) / 2.0
    delta2 = xp.minimum((z_xp - z_well) ** 2, (z_xp + z_well) ** 2)
    V = (omega_trap ** 2 / (4.0 * D_zz)) * delta2
    Omega_bar = 2.0 * np.pi * muB_g * dB_bar
    sigma = dz * 2
    bar = 0.5 * (xp.tanh((z_xp + d_bar / 2) / sigma)
                 - xp.tanh((z_xp - d_bar / 2) / sigma))
    return V + Omega_bar * bar


def gauss(z_xp, dz, center_um, sigma_um=3.0):
    g = xp.exp(-((z_xp - center_um * 1e-6) ** 2) / (2 * (sigma_um * 1e-6) ** 2))
    return g / xp.sqrt(xp.sum(xp.abs(g) ** 2) * dz)


def init_state(ctrl, tgt, z_xp, dz):
    cA, cB, eps = -L_um / 2, +L_um / 2, 1e-3
    gA, gB = gauss(z_xp, dz, cA), gauss(z_xp, dz, cB)
    mk = {'0': (gA, eps * gA), '1': (eps * gA, gA)}
    mb = {'0': (gB, eps * gB), '1': (eps * gB, gB)}
    return (*mk[ctrl], *mb[tgt])


def renorm(Pp, Pm, mask, dz):
    n = xp.sqrt(xp.sum((xp.abs(Pp) ** 2 + xp.abs(Pm) ** 2) * mask) * dz + 1e-30)
    return Pp / n, Pm / n


def run_case(ctrl, tgt, Nz, dt, K_max=K_demo):
    """Noise-free, renormalised (pumped) run; returns (N_Bp, N_Bm) at t=tau_g."""
    dz, z_xp, kz_xp, maskA, maskB = build_grid(Nz)
    V = potential(z_xp, dz)
    Ap, Am, Bp, Bm = init_state(ctrl, tgt, z_xp, dz)
    kin = xp.exp(-1j * D_zz * kz_xp ** 2 * dt * 0.5)
    Nt = int(round(TAU_G / dt))

    for _ in range(Nt):
        Ap = xp.fft.ifft(kin * xp.fft.fft(Ap))
        Am = xp.fft.ifft(kin * xp.fft.fft(Am))
        Bp = xp.fft.ifft(kin * xp.fft.fft(Bp))
        Bm = xp.fft.ifft(kin * xp.fft.fft(Bm))

        nAp, nAm = xp.abs(Ap) ** 2, xp.abs(Am) ** 2
        nBp, nBm = xp.abs(Bp) ** 2, xp.abs(Bm) ** 2
        Ap *= xp.exp((-1j * (V + T_self * nAp + S_cross * nAm) - Gamma0) * dt)
        Am *= xp.exp((-1j * (V + T_self * nAm + S_cross * nAp) - Gamma0) * dt)
        Bp *= xp.exp((-1j * (V + T_self * nBp + S_cross * nBm) - Gamma0) * dt)
        Bm *= xp.exp((-1j * (V + T_self * nBm + S_cross * nBp) - Gamma0) * dt)

        NA_p = float(to_np(xp.sum(nAp * maskA))) * dz
        NA_m = float(to_np(xp.sum(nAm * maskA))) * dz
        K_eff = K_max * NA_m / (NA_p + NA_m + 1e-30)
        if K_eff > 0.0:
            c, s = np.cos(K_eff * dt), np.sin(K_eff * dt)
            Bp, Bm = c * Bp - 1j * s * Bm, c * Bm - 1j * s * Bp

        Ap = xp.fft.ifft(kin * xp.fft.fft(Ap))
        Am = xp.fft.ifft(kin * xp.fft.fft(Am))
        Bp = xp.fft.ifft(kin * xp.fft.fft(Bp))
        Bm = xp.fft.ifft(kin * xp.fft.fft(Bm))

        Ap, Am = renorm(Ap, Am, maskA, dz)
        Bp, Bm = renorm(Bp, Bm, maskB, dz)

    NBp = float(to_np(xp.sum(xp.abs(Bp) ** 2 * maskB))) * dz
    NBm = float(to_np(xp.sum(xp.abs(Bm) ** 2 * maskB))) * dz
    tot = NBp + NBm + 1e-30
    return NBp / tot, NBm / tot


CASES = [('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0')]
# (control, target_in, expected_target_out): '0'->measure N_Bp, '1'->N_Bm


def truth_table(Nz, dt):
    out = {}
    for c, t, eB in CASES:
        NBp, NBm = run_case(c, t, Nz, dt)
        out[(c, t)] = NBp if eB == '0' else NBm
    return out


if __name__ == '__main__':
    print(f"Backend: {'CuPy/GPU' if _GPU else 'NumPy/CPU'}")
    print(f"Gate time tau_g = {TAU_G*1e9:.4f} ns,  K/2pi = 250 MHz\n")
    print("Scope: ideal assigned-coupling benchmark; not geometry-resolved evidence.\n")

    grids = [
        ('base   (Nz=256, dt=5.0 ps)', 256, 5.0e-12),
        ('x2 z   (Nz=512, dt=5.0 ps)', 512, 5.0e-12),
        ('x2 t   (Nz=256, dt=2.5 ps)', 256, 2.5e-12),
        ('x2 z,t (Nz=512, dt=2.5 ps)', 512, 2.5e-12),
    ]
    results = {}
    for label, Nz, dt in grids:
        tt = truth_table(Nz, dt)
        results[label] = tt
        line = "  ".join(f"{k[0]}{k[1]}->{v:.5f}" for k, v in tt.items())
        print(f"{label}:  {line}")

    base = results[grids[0][0]]
    fine = results[grids[-1][0]]
    max_abs = max(abs(fine[k] - base[k]) for k in base)
    print(f"\nMax |change| base -> finest:  {max_abs:.2e}")
    print("This value quantifies only benchmark discretisation sensitivity.")
