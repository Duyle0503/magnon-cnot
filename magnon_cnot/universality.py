"""Ideal matrix-algebra reference for the proposed classical truth table.

These matrices do not establish physical entanglement or quantum-computational
universality for a classical magnon-BEC order parameter. CNOT plus one-bit
operations generates affine reversible Boolean maps; nonlinear reversible
logic requires an additional genuine three-bit primitive.
"""
import math
import numpy as np


def Rx(t):
    c = math.cos(t / 2); s = math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], complex)


def Rz(t):
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])


# Gate matrices
H_mat    = Rz(np.pi / 2) @ Rx(np.pi / 2) @ Rz(np.pi / 2)
S_mat    = Rz(np.pi / 2)
T_mat    = Rz(np.pi / 4)
I2       = np.eye(2, dtype=complex)
CNOT_mat = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], complex)

# Formal matrix checks only; these are not device simulations.
H_tgt   = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], complex)
H_ov    = abs(np.trace(H_mat.conj().T @ H_tgt)) / 2

state00  = np.array([1, 0, 0, 0], complex)
bell     = CNOT_mat @ (np.kron(H_mat, I2) @ state00)  # formal complex vector
bell_tgt = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1], complex)
bell_F   = abs(np.vdot(bell_tgt, bell))**2

CLASSICAL_LOGIC_SCOPE = (
    "ideal affine reversible truth-table algebra; classical and non-entangling; "
    "not a BQP-universality claim"
)
