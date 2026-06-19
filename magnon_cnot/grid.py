"""Spatial grid and GPU/CPU backend selection."""
import numpy as np

try:
    import cupy as cp
    xp  = cp
    GPU = True
except ImportError:
    xp  = np
    GPU = False


def to_np(a):
    """Convert CuPy array to NumPy (no-op if already NumPy)."""
    if GPU and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


# Grid parameters
L_um  = 20.0
Nz    = 256
Lphys = 2.0 * L_um * 1e-6
dz    = Lphys / Nz

z_np  = (np.arange(Nz) - Nz / 2) * dz
z_um  = z_np * 1e6
kz_np = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)

z_xp  = xp.asarray(z_np)
kz_xp = xp.asarray(kz_np)

mask_A = xp.asarray(z_np < 0).astype(xp.float64)
mask_B = xp.asarray(z_np > 0).astype(xp.float64)
