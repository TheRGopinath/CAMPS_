"""Dense state-vector reference ONLY for checking small n."""
import numpy as np
from mps.state import MATRICES


def dense_gate(state, gate, n):
    if gate.kind == "T":
        u = np.diag([1., np.exp(1j*np.pi/4)])
    elif gate.kind == "SDG":
        u = MATRICES["Z"]*0 + np.diag([1., -1j])
    elif gate.kind == "S":
        u = np.diag([1., 1j])
    elif gate.kind == "H":
        u = np.array([[1., 1.], [1., -1.]])/np.sqrt(2)
    elif gate.kind in ("CX", "CP"):
        tensor = state.reshape((2,)*n).copy()
        # Swap the two target-bit slices only where control=1 for X;
        # use the appropriate two-by-two target Pauli for CP.
        target = MATRICES["X" if gate.kind == "CX" else gate.axis]
        moved = np.moveaxis(tensor, (gate.a, gate.b), (0, 1))
        moved[1] = np.einsum("st,t...->s...", target, moved[1].copy())
        return tensor.reshape(-1)
    else:
        raise ValueError(gate.kind)
    tensor = state.reshape((2,)*n)
    return np.moveaxis(np.tensordot(u, tensor, axes=(1, gate.a)), 0, gate.a).reshape(-1)


def exact_state(circuit, n):
    state = np.zeros(2**n, complex)
    state[0] = 1
    for gate in circuit:
        state = dense_gate(state, gate, n)
    return state


def exact_expectation(state, p):
    from mps import MPS
    n = state.size.bit_length()-1
    return np.vdot(state, MPS.from_vector(state, n).pauli(p).vector())
