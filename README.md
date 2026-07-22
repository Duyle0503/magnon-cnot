# magnon-cnot

**Conditional Josephson coupling toward an analog classical CNOT in a magnon Bose–Einstein condensate**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](pyproject.toml)

## Scope

This repository supports the revised numerical study of a conditional
Josephson-coupling concept for classical magnon-BEC logic in YIG. The evidence
is intentionally separated into three levels:

1. **Ideal prescribed-coupling benchmark.** The original one-dimensional GPE
   and four-mode calculations test the truth table after an ideal on/off
   coupling is assigned. The point `K/2π = 250 MHz` and the corresponding
   `F = 0.9906` ensemble result are design-envelope benchmarks, not parameters
   extracted from the reported geometry.
2. **Reduced receiver-mediated transfer model.** A geometry-resolved two-dimensional
   magnetostatic–spinor-GPE pipeline represents the wave-vector-selective
   receiver/transducer by an equivalent magnetic source. The spatial
   double-well WKB expression is used as a declared ansatz for the logical
   off-diagonal term; the microscopic `<+q0|H|-q0>` matrix element is not
   derived. At the reported
   20 nm grid it gives `K0/2π = 1.048 MHz`, `K1/2π = 1.306 MHz`, a 24.7%
   modulation, `Fcond,min = 0.586`, and `Fabs,min = 0.215`. This is not a
   high-fidelity CNOT result.
3. **Three-dimensional LLG subsystem audits.** Time-dependent MuMax3 calculations
   test weak-drive linearity, explicit field-defined double-well mode identity,
   combined-device equilibrium, mesh sensitivity, and the raw direct-dipolar
   `+q0/-q0` null control.

## Claim boundary

- The logical object is a classical macroscopic order parameter; no physical
  entanglement or quantum-computational universality is claimed.
- The word **CNOT** denotes the proposed conditional truth-table target.
- The MuMax3 barrier is a prescribed longitudinal Zeeman-field region.
- The calibrated wave-vector-selective receiver/transducer is not included in
  the present three-dimensional LLG geometry.
- The LLG even/odd spatial-mode audit does not validate the momentum-space
  `<+q0|H|-q0>` identification used in the reduced transfer ansatz.
- The reduced model uses a 120 nm centre-plane separation (56 nm facing-surface
  separation for 64 nm layers), whereas the combined LLG audit uses a 120 nm
  surface gap (184 nm centre-plane separation); these are separately declared
  geometries and are not a quantitative cross-model match.
- All apparent LLG even/odd separations remain below their native spectral
  resolution and are not reported as measured couplings.
- The 20-to-16 nm LLG refinement preserves mode identity but shifts the
  mode-family centre by 2.87%; absolute frequency is therefore mesh sensitive.
- The calculations define a constrained route toward conditional operation,
  not a simulated or demonstrated CNOT device.

## Repository structure

```text
magnon-cnot/
├── revised_model/
│   └── spinor2d_run_all.py        # authoritative reduced 2D revision pipeline
├── notebooks/
│   └── llg-mumax.ipynb            # curated Kaggle MuMax3 campaign with outputs
├── results/
│   └── reduced_model/
│       └── run_all_report.json     # machine-readable reduced-model report
├── magnon_cnot/                    # original ideal-benchmark Python package
├── run_all.py                      # ideal prescribed-coupling benchmark
├── convergence_test.py             # benchmark numerical convergence
├── magnon_cnot_single_cell.py      # legacy one-cell benchmark
├── pyproject.toml
├── LICENSE
└── README.md
```

## Reproducing the calculations

### 1. Ideal prescribed-coupling benchmark

```bash
python -m pip install -e .
python run_all.py
```

The resulting truth table verifies an assigned two-mode Hamiltonian. It must
not be cited as geometry-resolved evidence.

### 2. Reduced 2D receiver-mediated model

The revision pipeline is designed for a Kaggle GPU notebook:

```bash
python revised_model/spinor2d_run_all.py
```

It writes JSON/CSV reports, source arrays, figures, hashes, and a ZIP archive
under `/kaggle/working/jphysd_spinor2d_run_all`. The archived reference report
is available at [`results/reduced_model/run_all_report.json`](results/reduced_model/run_all_report.json).

For a slower independent CPU replay, set `MAGNON_CNOT_FORCE_CPU=1` and set
`MAGNON_CNOT_OUTPUT_DIR` to a path whose final directory name is
`jphysd_spinor2d_run_all`. The report records the execution backend.

The pipeline status is `NO-GO_REDUCED_MODEL` for the preregistered 80% device
threshold. This negative quantitative result is retained; no gain or fitted
constant is introduced to force a CNOT outcome.

### 3. MuMax3 full-LLG validation

Open [`notebooks/llg-mumax.ipynb`](notebooks/llg-mumax.ipynb) on Kaggle and:

1. enable **GPU T4 ×2**;
2. add the dataset `duyle09/mumax3-bin`;
3. keep internet access disabled;
4. follow the execution map at the top of the notebook.

The main manuscript-evidence path is
`K00 → K01 → K05 → K05B → K05C → K05D → K06A → K06B`.
K07 is standalone and may be run in a fresh session. K02–K04B are retained in
the notebook appendix as exploratory provenance and are not the final
explicit-double-well evidence.

A public Kaggle mirror will be linked here when its permanent project URL is
available.

## Citation

If you use this repository, please cite the associated manuscript:

> B.-D. Le, “Conditional Josephson coupling toward an analog classical CNOT in
> a magnon Bose–Einstein condensate,” *Journal of Physics D: Applied Physics*,
> manuscript JPhysD-143496 (under revision).

For the micromagnetic calculations, also cite:

> A. Vansteenkiste *et al.*, “The design and verification of MuMax3,”
> *AIP Advances* **4**, 107133 (2014), DOI: 10.1063/1.4899186.

## License

MIT. See [LICENSE](LICENSE).
