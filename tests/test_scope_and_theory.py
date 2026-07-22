import unittest

import numpy as np

from magnon_cnot.theory import K_WKB
from magnon_cnot.universality import CLASSICAL_LOGIC_SCOPE


class CorrectedWKBTests(unittest.TestCase):
    def test_corrected_prefactor_and_action(self):
        d = 0.8e-6
        dB = 0.05
        omega_mode = 2.0 * np.pi * 100e6
        omega_attempt = 2.0 * np.pi * 300e6

        value = K_WKB(
            d,
            dB,
            omega_mode=omega_mode,
            omega_attempt=omega_attempt,
        )
        forbidden = 2.0 * np.pi * 1.4e10 * dB - omega_mode
        expected = (omega_attempt / (2.0 * np.pi)) * np.exp(
            -d * np.sqrt(forbidden / 3.0e-4)
        )
        self.assertAlmostEqual(float(value), float(expected), places=12)
        self.assertLess(float(value), omega_attempt / (2.0 * np.pi))

    def test_non_forbidden_region_is_rejected(self):
        with self.assertRaises(ValueError):
            K_WKB(
                1e-6,
                1e-4,
                omega_mode=2.0 * np.pi * 100e6,
                omega_attempt=2.0 * np.pi * 300e6,
            )


class ScopeTests(unittest.TestCase):
    def test_classical_scope_is_explicit(self):
        text = CLASSICAL_LOGIC_SCOPE.lower()
        self.assertIn("classical", text)
        self.assertIn("non-entangling", text)
        self.assertIn("not a bqp", text)


if __name__ == "__main__":
    unittest.main()
