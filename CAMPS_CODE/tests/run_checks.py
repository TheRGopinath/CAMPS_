"""Dependency-free test runner: python -m tests.run_checks."""
from .test_simulation import (test_tableau, test_mps, test_random_circuits,
                              test_pauli_gate_conjugation, test_expectations_with_and_without_ofd)

if __name__ == "__main__":
    for check in (test_tableau, test_mps, test_pauli_gate_conjugation,
                  test_random_circuits, test_expectations_with_and_without_ofd):
        check()
        print(check.__name__, "passed")
