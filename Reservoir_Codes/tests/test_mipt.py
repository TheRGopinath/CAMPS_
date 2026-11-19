"""Cross-check fast monitored stabilizer tableau against original Tableau."""
import numpy as np
from mipt import FastStabilizer, two_qubit_clifford_words
from stabilizers_pbc import Tableau, Gate, Pauli


def run_checks():
    words=two_qubit_clifford_words()
    assert len(words)==11520
    # Bell state has one ebit, and Z measurement destroys it.
    t=FastStabilizer(2);t.h(0);t.cx(0,1)
    assert t.entropy([0])==1
    t.measure_z(0,np.random.default_rng(5))
    assert t.entropy([0])==0
    # Compare arbitrary gate/measurement sequences with original implementation.
    for n in (2,3,5):
        for seed in range(5):
            rng=np.random.default_rng(seed+13*n)
            fast=FastStabilizer(n); reference=Tableau(n)
            for step in range(35):
                if rng.random()<.7:
                    q=int(rng.integers(n));kind=('H','S')[int(rng.integers(2))]
                    if kind=='H':fast.h(q)
                    else:fast.s(q)
                    reference.apply(Gate(kind,q))
                else:
                    a,b=rng.choice(n,2,replace=False).tolist()
                    fast.cx(a,b);reference.apply(Gate('CX',a,b))
                q=int(rng.integers(n));p=Pauli.string('I'*q+'Z'+'I'*(n-q-1))
                assert fast.z_expectation(q)==reference.expectation(p)
                if rng.random()<.35:
                    # Independent but identically seeded RNGs give identical Born samples.
                    r1=np.random.default_rng(step+999)
                    r2=np.random.default_rng(step+999)
                    assert fast.measure_z(q,r1)==reference.measure(p,r2)
                    assert fast.z_expectation(q)==reference.expectation(p)
    print('MIPT checks passed: 11520 Clifford group, Bell entropy, original-tableau agreement.')


if __name__=='__main__':run_checks()
