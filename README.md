# magnon-cnot

**Classical CNOT gate via conditional Josephson coupling in a magnon Bose–Einstein condensate**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

We propose and numerically demonstrate a room-temperature classical CNOT gate
using two spatially separated magnon BEC qubits in yttrium iron garnet (YIG),
coupled through a magnetic barrier via WKB Josephson tunnelling.

**Key results:**
- Analytical fidelity bound: F ≥ 1 − πΓ/K<sub>max</sub>
- K* = 236 MHz for F ≥ 0.99 (YIG, α = 2×10⁻⁴)
- Gate time τ = 1.0 ns at K/2π = 250 MHz
- Ensemble fidelity F = 0.9906 (500 trajectories, T = 300 K), loss-limited
- Affine reversible gate set {CNOT, H, S, T} verified (classical, non-entangling;
  full reversible-logic universality needs a 3-bit primitive — see paper)

## Quick start

### Google Colab (recommended)
```
Cell 0:  !pip install cupy-cuda12x -q
Cell 1:  paste code/magnon_cnot_single_cell.py
```
Runtime: ~15 min on T4 GPU.

### Local (modular)
```bash
pip install -e .            # CPU
pip install -e ".[gpu]"     # GPU (CUDA 12)
python run_all.py
```

## Project structure

```
magnon-cnot/
├── magnon_cnot/            # Python package
│   ├── __init__.py
│   ├── constants.py        # Physical constants & YIG parameters
│   ├── theory.py           # WKB coupling, fidelity bound, gate time
│   ├── grid.py             # Spatial grid & GPU/CPU backend
│   ├── gpe.py              # Gross-Pitaevskii split-step solver
│   ├── reduced_model.py    # 4-mode ODE + ensemble fidelity
│   ├── universality.py     # {CNOT, H, S, T} gate set verification
│   └── figures.py          # Publication figures (APL format)
├── run_all.py              # Main entry point
├── examples/
│   └── colab_single_cell.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Citation

If you use this code, please cite:

> B.-D. Le, "Classical CNOT gate via conditional Josephson coupling in a magnon
> Bose–Einstein condensate," *Appl. Phys. Lett.* (2025). [submitted]

## References

- Mohseni et al., *Commun. Phys.* **5**, 196 (2022) — single-qubit magnon BEC gate
- Demokritov et al., *Nature* **443**, 430 (2006) — magnon BEC observation
- Bozhko et al., *Nature Phys.* **12**, 1057 (2016) — room-temperature magnon BEC

## License

MIT
