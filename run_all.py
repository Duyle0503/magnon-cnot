#!/usr/bin/env python3
"""
Ideal prescribed-coupling benchmark for conditional magnon-BEC logic.

This script assigns the on/off coupling and is not geometry-resolved evidence,
a receiver/transducer simulation, or a demonstrated CNOT device.

Run all simulations and generate publication figures.
Usage:
    python run_all.py           # full run (GPU recommended, ~15 min on T4)
    python run_all.py --cpu     # force CPU mode
"""
import os, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
np.random.seed(20260519)

from magnon_cnot.constants import Gamma0, kB, T_K, hbar, omega0, sig_th
from magnon_cnot.theory import legacy_K_WKB_proxy, F_bound, tau_gate, K_star, K_demo, T_demo
from magnon_cnot.grid import GPU, dz
from magnon_cnot.gpe import run_sim, integ
from magnon_cnot.reduced_model import fidelity_4m, ensemble_F
from magnon_cnot.universality import H_ov, bell_F, CLASSICAL_LOGIC_SCOPE

if GPU:
    import cupy as cp
    cp.random.seed(20260519)

# ── Print configuration ──────────────────────────────────────────────────────
print(f"{'✓  GPU mode (CuPy)' if GPU else '⚠  CPU mode (NumPy)'}")
print(f"   omega0/2π = {omega0/(2*np.pi)/1e9:.3f} GHz")
print(f"   Gamma0/2π = {Gamma0/(2*np.pi)/1e6:.3f} MHz")
print(f"   K*/2π     = {K_star/(2*np.pi)/1e6:.1f} MHz  (F≥0.99 threshold)")
print(f"   K_ideal   = {K_demo/(2*np.pi)/1e6:.1f} MHz   τ_gate = {T_demo:.2f} ns")
print("   scope     = assigned ideal coupling; not extracted from this geometry")

# ── Part 2: GPE truth table ──────────────────────────────────────────────────
truth_cases = [
    ('0', '0', r'$|00\rangle\!\to\!|00\rangle$', '0'),
    ('0', '1', r'$|01\rangle\!\to\!|01\rangle$', '1'),
    ('1', '0', r'$|10\rangle\!\to\!|11\rangle$', '1'),
    ('1', '1', r'$|11\rangle\!\to\!|10\rangle$', '0'),
]

print("\n── Part 2: Spatial GPE truth table ─────────────────────────────")
sims = {}
for ctrl, tgt, label, exp_B in truth_cases:
    t0 = time.time()
    sims[(ctrl, tgt)] = run_sim(ctrl, tgt, K_demo, T_demo,
                                 dt_ns=0.005, n_save=200, noise=True)
    nBp, nBm = integ(sims[(ctrl, tgt)], 'B')
    F = nBp[-1] if exp_B == '0' else nBm[-1]
    print(f"  {label:35s}  F={F:.4f}   [{time.time()-t0:.1f}s]")

tt_F = []
for ctrl, tgt, _, exp_B in truth_cases:
    nBp, nBm = integ(sims[(ctrl, tgt)], 'B')
    tt_F.append(nBp[-1] if exp_B == '0' else nBm[-1])
print(f"  F_avg (pumped) = {np.mean(tt_F):.4f}")

# ── Part 3: Parameter sweeps ─────────────────────────────────────────────────
print("\n── Part 3: Parameter sweeps ─────────────────────────────────────")
K_sw  = np.logspace(7, 9.5, 28)
G_sw  = np.logspace(5, 8, 24)
d_sw  = np.linspace(0.3, 3.0, 24)
dB_sw = np.linspace(0.5, 30.0, 24)

F_KG = np.array([[fidelity_4m(K, G) for G in G_sw] for K in K_sw])
F_Kd = np.zeros((len(d_sw), len(dB_sw)))
K_Kd = np.zeros_like(F_Kd)
for i, d in enumerate(d_sw):
    for j, dB in enumerate(dB_sw):
        Kv = legacy_K_WKB_proxy(d * 1e-6, dB * 1e-3)
        K_Kd[i, j] = Kv
        F_Kd[i, j] = fidelity_4m(min(Kv, 5e9))

idx = np.unravel_index(np.argmin(np.abs(K_Kd - K_star)), K_Kd.shape)
print(f"  Optimal: d={d_sw[idx[0]]:.2f}µm  dB={dB_sw[idx[1]]:.1f}mT"
      f"  K={K_Kd[idx]/(2*np.pi)/1e6:.1f}MHz  F={F_Kd[idx]:.4f}")

# ── Part 4: Ensemble fidelity ────────────────────────────────────────────────
print("\n── Part 4: Ensemble fidelity (500 traj, 300K, no pump) ─────────")
t0 = time.time()
F_ens, F_err, F_ens_mean, F_ens_std = ensemble_F(K_demo, N=500)
print(f"  Grand mean  F = {F_ens_mean:.4f} ± {F_ens_std:.4f}  [{time.time()-t0:.1f}s]")
for (c, t, _, eB), f, e in zip(truth_cases, F_ens, F_err):
    print(f"  {c}{t}→{'1' if eB == '1' else '0'}   F = {f:.4f} ± {e:.4f}")

# sweeps
noise_sw = np.array([0.000, 0.001, 0.003, 0.005, 0.010, 0.020])
F_n_sw   = [ensemble_F(K_demo, N=300, noise=n, Nt=300)[2] for n in noise_sw]
K_ens_sw = np.logspace(7.5, 9.5, 18)
F_K_sw   = [ensemble_F(K, N=200, Nt=250)[2] for K in K_ens_sw]
print("  Sweeps done.")

# ── Part 5: Ideal algebra reference ─────────────────────────────────────────
print(f"\n── Part 5: Ideal algebra reference (not device evidence) ───────")
print(f"  H overlap = {H_ov:.6f}")
print(f"  Formal two-bit vector overlap = {bell_F:.6f}")
print(f"  Scope: {CLASSICAL_LOGIC_SCOPE}")

# ── Generate figures ──────────────────────────────────────────────────────────
print("\n── Generating figures ───────────────────────────────────────────")
from magnon_cnot.figures import make_fig1, make_fig2, make_fig3, make_fig4

make_fig1();  print("  ✓ fig1_theory")
make_fig2(sims, tt_F);  print("  ✓ fig2_dynamics")
make_fig3(F_KG, K_sw, G_sw, F_Kd, d_sw, dB_sw, K_Kd, idx);  print("  ✓ fig3_sweep")
make_fig4(F_ens, F_err, F_ens_mean, F_ens_std,
          noise_sw, F_n_sw, K_ens_sw, F_K_sw);  print("  ✓ fig4_ensemble")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║  Assigned-coupling ideal benchmark summary                      ║
╠══════════════════════════════════════════════════════════════════╣
║  K*/2π  = {K_star/(2*np.pi)/1e6:6.1f} MHz   (F≥0.99 threshold)            ║
║  τ_gate = {tau_gate(K_demo)*1e9:6.2f} ns    at K/2π={K_demo/(2*np.pi)/1e6:.0f} MHz           ║
║                                                                 ║
║  GPE truth-table (pumped):  F_avg = {np.mean(tt_F):.4f}                  ║
║  Ensemble (500 traj, 300K): F_avg = {F_ens_mean:.4f} ± {F_ens_std:.4f}         ║
║  Analytical bound at K_demo:        {F_bound(K_demo):.4f}                ║
║                                                                 ║
║  Formal H overlap={H_ov:.4f}; two-bit vector overlap={bell_F:.4f}      ║
║  Classical affine-logic reference; no entanglement/BQP claim    ║
║                                                                 ║
║  LaTeX:  \\bar{{F}} = {F_ens_mean:.4f}  (T=300K)                        ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)
print("BOUNDARY: This output is an ideal assigned-coupling benchmark, not a geometry-resolved CNOT result.")

with open('magnon_cnot_summary.txt', 'w') as f:
    f.write(summary)
