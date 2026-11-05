# Thesis CAMPS demonstration (Python)

Small, readable implementation of binary Pauli/stabilizer operations, exact MPS operations, and the optimization-free disentangler (OFD) of Liu and Clark. The files are grouped by the mathematical role they serve:

| Folder | Purpose |
| --- | --- |
| `stabilizers_pbc/` | Pauli bits and phases; Clifford conjugation; stabilizer tableau and Pauli measurement |
| `mps/` | Open-boundary MPS, contraction, SVD, exact sum, Pauli rotations |
| `camps/` | T-to-twisted-Pauli conversion, OFD, CAMPS expectation values; small dense checker |
| `examples/` | Two seeded random circuit demonstrations and a binary preprocessing timing |
| `tests/` | Cross-checks with dense state vectors and Pauli conjugation |

## Run

From this directory, with Python 3.10+, NumPy and Matplotlib installed:

```bash
python -m examples.two_circuits
python -m tests.run_checks
```

In Jupyter, open a notebook inside this directory, run `%pip install numpy matplotlib`,
then run `%run -m examples.two_circuits`. The example prints each calculation
step, displays two circuit diagrams in the notebook, and saves them to
`examples/output/ofd_success.png` and `examples/output/ofd_failure.png`.
Each figure shows the original gate sequence (left to right) and the OFD
status and maximum MPS bond for each transformed T gate. To see the tests,
run `%run -m tests.run_checks` in another cell.
The failed-OFD example also prints a local tensor contraction of the selected
observable immediately after **each** failed step. These intermediate values
include the final Clifford frame but only the T rotations processed so far;
the separate final expectation includes every gate. To request the same
readings in your own circuit, call `simulate(circuit, n, probe=Pauli.string(...))`
and inspect `result.failure_expectations`.
The example prints times for circuit search, Pauli propagation, GF(2)
elimination, successful OFD updates, failed-step MPS updates, intermediate
and final expectation contractions, dense reference checks, and figure
creation. The `result.timings` dictionary and `result.step_timings` list
expose the core timings to your own scripts. Timing is a single illustrative
run, not a statistical benchmark; circuit selection includes multiple trial
simulations and is reported separately from the final CAMPS simulation.
`Tableau.expectation(pauli)` directly returns 0 or ±1 for a pure stabilizer
state, without measurement or mutation. `MPS.expectation(pauli)` contracts the
tensors site by site and can return non-integer values. `CAMPS.expectation`
uses the tableau for circuits with no T gates and the MPS contraction (after
Clifford conjugation of the observable) for circuits with T gates, including
the case where OFD has failed. In the failed-OFD example, the contractions
printed after each failure are taken on the partially evolved CAMPS state;
the final expectation is taken after every T has been processed.

`pip install -e .` makes the three computational packages importable elsewhere. No Julia runtime or external quantum library is required. The examples use a fixed random seed and select circuits satisfying the demonstrated conditions; they report how many candidates were tried.

## Conventions and cost

Qubit 0 is the leftmost character in a Pauli string and the leftmost tensor-product factor. `Pauli(x,z,phase)` represents `i**phase X**x Z**z`; hence `Y=iXZ`. Gates are listed in time order. `CAMPS.frame` is the Clifford circuit C in `|output> = C |MPS>`; frame gates are also stored in application order. The harmless global phase `exp(i*pi/8)` per T is omitted.

For each encountered T, `move_t_gates` records `C† Z C` and builds the final Clifford frame from the interleaved Cliffords. OFD selects the first free |0> site on which its *current* twisted Pauli is X or Y. Controlled Pauli gates disentangle it, consuming one free site; remaining Pauli strings are conjugated through that disentangler. When there is no pivot, the code applies `(alpha I + beta P)` by exact MPS addition and SVD. `initial_nullity` is the **left** nullity (`number of T strings - GF(2) rank`) of the X/Y binary matrix specified in the paper. A missing pivot is a failure of OFD at that step, not proof that all CAMPS representations fail.

`move_t_gates` uses O(t L n) simple binary operations when there are t T gates and at most L preceding Cliffords per gate (this implementation scans the frame for each T). GF(2) elimination is polynomial too. OFD successes keep the MPS bond dimension unchanged; failures can double it and thus lead to exponential MPS work. The large-n timing reports **binary preprocessing only**, and does not claim polynomial time for arbitrary Clifford+T circuits. Exact state vectors are used only for small-n verification.

## References

- Aaronson–Gottesman tableau update rules and their corrected x/z definitions in the supplied *Stabilizers.pdf*.
- Schollwöck MPS tutorial in the supplied *MPS.pdf*.
- Liu and Clark, *Classical simulability of Clifford+T circuits with Clifford-augmented matrix product states*, supplied *CAMPS.pdf*, especially Eqs. 1–7, Theorems 1–3, and Algorithms 1–2.

The tableau covers Clifford evolution and nondestructive Pauli measurements on pure stabilizer states. It is the binary Clifford/PBC foundation; this project does not claim to implement the paper's full magic-state PBC reduction or its optimization-based disentangler.
