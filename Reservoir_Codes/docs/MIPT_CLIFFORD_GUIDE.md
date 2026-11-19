# Monitored random Clifford circuits: first MIPT baseline

This is a **new Clifford-only extension** of the existing `ThesisCAMPS` repository. It does not change the original CAMPS/OFD implementation. The purpose is to reproduce the basic competition between entangling brickwork dynamics and local projective measurements using stabilizer tableaux.

## Run

From the `ThesisCAMPS/` repository root:

```bash
python -m tests.test_mipt
python -m examples.mipt_clifford
```

Requirements: Python 3.10+, `numpy`, `matplotlib`. In Jupyter, use `%run -m examples.mipt_clifford` from the repository root. Figures and a CSV appear in `examples/output/mipt/`.

## Exact protocol

- Qubits are labeled `0,...,N-1` with open boundaries; start from `|0>^N`.
- A **full brickwork cycle** consists of **odd bonds** `(0,1),(2,3),...`, **one independent Z-measurement attempt per site**, **even bonds** `(1,2),(3,4),...`, then **another independent Z-measurement attempt per site**.
- Each two-qubit gate is sampled **uniformly from all 11,520 projective two-qubit Clifford gates**. This is an exact unitary 2-design, not a generic Haar gate and not a short random walk. The program enumerates all group elements once by breadth-first search in their signed Pauli action and samples a uniformly random element.
- For each measurement attempt, draw `r~Uniform[0,1)` and measure if `r<p`. If measurement occurs, sample its eigenvalue `s=+/-1` from the Born rule and condition the state on the observed outcome.
- Repeat for `depth_factor*N` **full cycles**. The default is 3N cycles, meaning 6N two-qubit gate layers and 6N measurement layers.
- For each N,p, average the **pure-trajectory entanglement entropy**, not the entropy of the ensemble-averaged density matrix.

## Mathematics

For a pure stabilizer state on N qubits, the N independent stabilizer generators form the binary matrix

\[
G=[X\mid Z]\in\mathbb F_2^{N\times 2N}.
\]

For the left-half subsystem A and its complement B, the dimension of the stabilizer subgroup supported entirely on A is

\[
\dim \mathcal S_A=N-\operatorname{rank}_{\mathbb F_2}(G_B),
\]

where `G_B` is obtained by retaining the X and Z columns associated with B. Since the reduced state of a stabilizer state is proportional to a stabilizer-code projector,

\[
S_A=|A|-\dim\mathcal S_A
=\operatorname{rank}_{\mathbb F_2}(G_B)-|B|.
\]

The entropy is in bits, an **integer for each pure stabilizer trajectory**. Averages over trajectories need not be integers. It is the physical von Neumann entropy, and for stabilizer states it also equals all Renyi entropies of positive order. No exponential statevector is constructed.

Measurement of `Z_q` has projectors

\[
P_s=(I+sZ_q)/2,\qquad s\in\{-1,+1\}.
\]

If `Z_q` anticommutes with a stabilizer generator, each outcome has probability 1/2; a stabilizer generator is replaced by `s Z_q`, and other generators are updated to commute with it. Otherwise the outcome is deterministic and its sign is calculated using the destabilizer dual basis. This is the same measurement logic as the existing `stabilizers_pbc.Tableau`, implemented here in vectorized form for speed.

## Files and figures

- `mipt/clifford_mipt.py`: uniform two-qubit Clifford sampler, signed binary stabilizer tableau, Clifford conjugation, Born-rule Z measurement, GF(2) entropy, and trajectory evolution.
- `examples/mipt_clifford.py`: commented experiment with editable `CONFIG`, plotting, printed numerical tables, CSV export.
- `tests/test_mipt.py`: checks exact Clifford group size, Bell-pair entropy and collapse, and cross-checks randomized gate/measurement sequences against the original repository `Tableau`.
- `examples/output/mipt/circuit_diagram.png`: **actual sampled locations** of Clifford gates and measurement attempts, with a manageable 8-qubit illustration; the labels C do not encode the specific sampled Clifford matrices.
- `examples/output/mipt/single_trajectory_entropy.png`: entanglement evolution along one conditioned trajectory.
- `examples/output/mipt/entropy_vs_probability.png`: mean half-chain entropy with standard-error bars for multiple N.
- `examples/output/mipt/entropy_density_vs_probability.png`: normalized entropy, helpful when contrasting finite sizes.
- `examples/output/mipt/entanglement_vs_p.csv`: exact numerical values, N, depth, p, number of trajectories, mean, standard error, and measurement counts.
- `examples/output/mipt/RUN_NOTES.txt`: experiment settings and conventions.

## Interpreting the results carefully

The plots show a **finite-size entanglement crossover** consistent with competition between random unitary entangling dynamics and projective measurements. The default sample size is intended as an informative reproducible demonstration, **not a high-precision determination of the thermodynamic critical probability or critical exponents**. For a research-quality phase diagram, increase `trajectories`, simulate larger N, check convergence in depth, and perform an explicit finite-size scaling analysis with uncertainties. A critical probability depends on the measurement scheme and placement. Do not borrow numerical `p_c` values from a different monitored-circuit model.

This code intentionally has **no T gates, no magic, and no residual CMPS bond growth**. A Clifford-only state is a special case of CAMPS with a product-state residual representation. The next research step is to connect these monitored trajectories to the existing CAMPS machinery without conflating physical entanglement with residual MPS entanglement.
