"""Spatial grid and GPU/CPU backend selection."""
import os
import sys

import numpy as np

FORCE_CPU = "--cpu" in sys.argv or os.environ.get("MAGNON_CNOT_FORCE_CPU", "0") == "1"

if FORCE_CPU:
    xp  = np
    GPU = False
else:
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device available")
        xp  = cp
        GPU = True
    except (ImportError, OSError, RuntimeError):
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
