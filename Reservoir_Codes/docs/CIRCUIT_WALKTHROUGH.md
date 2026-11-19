# What the two CAMPS circuits are doing

This guide accompanies `python -m examples.two_circuits`. It explains the exact
seeded examples in `examples/two_circuits.py`, the mathematics implemented by
the three code folders, and what the numerical output proves. Start with the
[OFD-success circuit](../examples/output/ofd_success.png) and the
[OFD-failure circuit](../examples/output/ofd_failure.png). In those drawings,
time runs from left to right, `q0` is the top wire, and yellow boxes are T
gates. The examples choose random gate locations using fixed seeds. The first
generator deliberately places H/T pairs on fresh qubits to make OFD succeed;
it is **not** a typical unrestricted random circuit. The second is selected
from seeded random candidates because it has at least one OFD failure and a
bond dimension greater than one. Candidate selection is part of the demo,
not a theorem about random circuits.

## 1. What is represented, and what is moved?

The input is `|000000⟩`. A gate list is written in *application order*:
`H(2), T(2), CX(2,1)` means H first, T second, CX last. The state after a
circuit `U` is `U|000000⟩`. All qubit labels and Pauli-string characters use
the order `q0,q1,...,q5`; `IIXIII` means X on `q2`.

Write a T gate (ignoring its global phase) as

\[
T_j = \alpha I+\beta Z_j,\qquad
\alpha=\cos(\pi/8),\quad \beta=-i\sin(\pi/8).
\]

Let `C_before` denote the product of **Clifford gates already encountered**
when reading the circuit from left to right. If the partial unitary is
`C_before R`, then adding a T on qubit `j` gives

\[
T_j C_{\rm before}R
=C_{\rm before}\bigl(\alpha I+\beta
  C_{\rm before}^{\dagger}Z_j C_{\rm before}\bigr)R.
\]

Thus each T becomes a rotation about a *twisted Pauli* `P_k` computed by
Clifford conjugation. A following Clifford gets appended to the Clifford
frame. At the end,

\[
U|0^N\rangle \doteq
C\,(\alpha I+\beta P_t)\cdots(\alpha I+\beta P_1)|0^N\rangle.
\]

The symbol `≐` means equality up to the omitted global phase
`exp(i t π/8)`. **The Pauli rotations remain in their original order.**
Moving the T gates to one side does not mean commuting arbitrary T gates
past one another. `move_t_gates` in `camps/simulation.py` implements this
identity by conjugating each encountered `Z_j` through the preceding
Cliffords. For example, `H Z H=X`; `CX(a,b) Z_b CX(a,b)=Z_a Z_b`.

Paulis use binary `x,z` vectors and a phase exponent:

\[
P=i^q\prod_j X_j^{x_j}Z_j^{z_j},\qquad
I=(0,0),\ X=(1,0),\ Z=(0,1),\ Y=(1,1)\text{ with }q=1.
\]

This phase bit matters: conjugating can change the **sign** of a Pauli even
when its letters look unchanged. A Clifford maps one signed Pauli string to
one signed Pauli string, so no `2^N` state vector is needed for this stage.

## 2. What is the CAMPS state?

The simulator stores `|ψ⟩ = C|φ⟩`, with a Clifford `C` (the `frame`) and an
MPS `|φ⟩` (the `state`). An MPS has one tensor per qubit:

\[
\phi_{s_0\cdots s_{N-1}}
=\sum_{a_0,\ldots,a_{N-2}}
A^{[0]s_0}_{1a_0}A^{[1]s_1}_{a_0a_1}\cdots
A^{[N-1]s_{N-1}}_{a_{N-2}1}.
\]

An internal index `a` has size called its **bond dimension**. Bond 1 means
the residual `|φ⟩` is a product state. It does **not** mean the physical
state `C|φ⟩` is unentangled: CX gates stored in `C` can entangle it. For the
first demo, every residual bond is 1, while the *physical* exact state has
Schmidt rank 2 across several cuts. In the second demo, the residual MPS
reaches bond 2; the physical state has larger Schmidt ranks because the
Clifford frame can add entanglement.

The bond dimension is the exact Schmidt rank of the represented state across
a cut when the MPS is in a suitable canonical form and numerical zero
singular values have been removed. Here the examples keep all singular
values above `1e-12`; they impose **no maximum bond cap**. The stated bond
values refer to the stored MPS after SVD cleanup, not the intermediate
direct-sum tensor shapes. The code discards singular values at or below an
absolute `1e-12` threshold, so "exact" here means exact up to floating-point
roundoff and this numerical cutoff; for a specially chosen state with
genuinely nonzero Schmidt values below that threshold, the cleanup would
introduce approximation. The six-qubit examples are checked against dense
results to approximately `1e-9`.

## 3. What does OFD decide at each T gate?

Before applying the current twisted rotation, OFD searches for a qubit `r`
which is still a free, unentangled `|0⟩` in the residual MPS **and** for
which the current Pauli has X or Y at `r`. Suppose
`P=A_r ⊗ (⊗_{j≠r} P_j)` with `A_r=X` or `Y`. Then

\[
(\alpha I+\beta P)|0\rangle_r|\eta\rangle
=\alpha |0\rangle_r|\eta\rangle
 +\beta A_r|0\rangle_r\!\bigotimes_{j\ne r}P_j|\eta\rangle.
\]

OFD applies a controlled-`P_j` from `r` to each other nonidentity site.
Nothing happens to the first branch (control 0); in the second branch the
same Pauli is applied a second time and cancels because `P_j²=I`. The result
is a one-qubit state at `r` times the unchanged `|η⟩`:

\[
D(\alpha I+\beta P)|0\rangle_r|\eta\rangle
=(\alpha|0\rangle_r+\beta A_r|0\rangle_r)|\eta\rangle.
\]

The code updates the **frame by `D†`**, rotates the local MPS tensor at `r`,
and marks `r` as consumed. This is a change of representation of the *same
physical state*. It also conjugates all later twisted Paulis through `D`;
otherwise subsequent OFD decisions would use the wrong Pauli strings.
Within the assumptions just stated, a successful OFD step need not grow any
MPS bond.

If no such free X/Y pivot exists, OFD fails **at that step**. The simulator
still applies the gate exactly:

\[
|\phi_{\rm new}\rangle
=\alpha|\phi\rangle+\beta P|\phi\rangle.
\]

`mps/state.py` applies P site by site, adds the two MPS states using a
block-diagonal/direct-sum construction, and uses right-to-left and then
left-to-right SVD sweeps to remove numerically zero Schmidt components. The
sum has an MPS representation with bonds at most **twice** their earlier
sizes before cleanup. A failed pivot can therefore leave the bond unchanged
or increase it. Failure of this OFD rule does **not** mean that an
expectation value cannot be evaluated, or that every possible Clifford
disentangler must fail.

## 4. First circuit: three T gates, three successful pivots

The example generated with seed 73 is

```text
H(2) T(2) CX(2,1)  H(3) S(3) T(3) CX(3,1)  H(4) T(4) CX(4,1)
```

The raw twisted Paulis, before applying OFD disentanglers, are:

| T step | Raw signed Pauli | OFD action | Residual maximum bond |
| --- | --- | --- | ---: |
| 1 | `+IIXIII` | Consume free `q2` | 1 |
| 2 | `+IIIXII` | Consume free `q3` | 1 |
| 3 | `+IIIIXI` | Consume free `q4` | 1 |

For the first T, `C_before=H(2)`, so `P₁=H₂ Z₂ H₂=X₂`.
At that point `q2` is free, hence
`(αI+βX₂)|0⟩₂=α|0⟩₂+β|1⟩₂`.
The next two T gates use different free sites. The CX gates are still part
of the physical circuit; their entanglement is tracked in the Clifford
frame. OFD reports **three successes and no failures**, and the residual
MPS ends with bond 1.

The program selects `O=IXXIII` as an illustrative observable. For this
particular final frame, binary Clifford conjugation gives
`C† O C=IIZIII`: only Z on residual site `q2` survives. Its expectation is

\[
\langle IXXIII\rangle=\langle\phi|Z_2|\phi\rangle
=|\alpha|^2-|\beta|^2=\cos(\pi/4)=1/\sqrt2
\approx0.707106781187.
\]

This is a useful hand calculation to explain to a professor: the physical
observable involves two X operators, while in the disentangled Clifford
frame the required MPS contraction is a single local Z expectation.

## 5. Second circuit: six failed pivots and one growing bond

The seeded search (seed 119; 31 candidates with this generator) selects:

```text
CX(5,3) H(0) T(1) S(2) CX(0,4) T(4)
H(0) H(0) T(0) CX(0,2) CX(3,2) T(2)
CX(5,2) S(0) T(3) S(5) H(3) T(2)
CX(3,1) CX(4,5) T(3) CX(0,3) S(4) T(3)
```

| T step | Current Pauli **after earlier disentanglers** | OFD | Bond |
| --- | --- | --- | ---: |
| 1 | `IZIIII` | No free X/Y pivot | 1 |
| 2 | `XIIIZI` | Consume free `q0` | 1 |
| 3 | `XIIIZI` | `q0` already consumed; fail | 1 |
| 4 | `XIZZZZ` | No free X/Y pivot; fail | 1 |
| 5 | `IIIZIZ` | No free X/Y pivot; fail | 1 |
| 6 | `XIZZZI` | No free X/Y pivot; fail | 1 |
| 7 | `IIYXII` | Consume free `q2` | 1 |
| 8 | `XIYIZI` | X at consumed `q0`, Y at consumed `q2`; fail | **2** |

The table records Pauli **letters**. Signs are carried in the code's phase
field even though the concise `OFD / MPS UPDATES` output omits them. The
initial raw strings are printed separately; they can differ from this table
because each earlier disentangler transforms the later strings.

Look at T1: its Pauli `IZIIII` acts as Z on a free `|0⟩`. Since
`Z|0⟩=|0⟩`, this rotation changes only the global phase of that particular
state; the code takes the failed-pivot branch, but the residual bond stays
1. The other failed steps likewise need not grow a bond. Only at T8 does
the exact MPS representation require bond 2. The point is to distinguish
**an algebraic OFD pivot failure** from **actual entanglement growth**.

The selected observable is `O=IXIXII`. The final Clifford frame changes it
to `C† O C=IIZZIZ`, so the final answer comes from contracting a three-Z
Pauli string with the residual MPS:

\[
\langle IXIXII\rangle
=\langle\phi|IIZZIZ|\phi\rangle=0.5.
\]

Here the MPS has bond list `[2,2,1,1,1]`. The code contracts it site by
site; the dense six-qubit calculation independently returns `0.5`.
It also contracts the same selected observable immediately after *each*
failed step: the displayed values are `1,1,1,1,1,0.5` at steps
`1,3,4,5,6,8` respectively. These intermediate objects include the full
final Clifford frame but only the T rotations through the reported step;
they are **not** expectation values of the physical circuit stopped at that
gate. The last value is the physical final-circuit expectation.

## 6. How is an expectation value calculated without a dense vector?

For a measured Pauli `O`, use the tracked Clifford frame:

\[
\langle O\rangle_{\psi}
=\langle\phi|C^{\dagger}OC|\phi\rangle.
\]

Clifford conjugation produces one signed Pauli `Q=C†OC` (and never an
exponentially large operator sum). If `Q=Q₀⊗⋯⊗Qₙ₋₁`, the MPS evaluates it
using a small environment matrix, initially `E₀=(1)`:

\[
(E_{j+1})_{b b'}=
\sum_{a,a',s,s'}(E_j)_{aa'}
\overline{A^{[j]s}_{ab}}\,(Q_j)_{ss'}\,A^{[j]s'}_{a'b'}.
\]

The final scalar `E_N` is the expectation. `MPS.expectation` implements
this by applying local Pauli matrices to a copy of the MPS and calling
`overlap`; it **does not construct the full state vector**. With bounded
physical dimension 2 and comparable bond dimensions χ, a simple local
contraction has polynomial cost in N and χ (typically of order `Nχ³` up
to constants/contraction order). This can nevertheless be exponential in N
when χ itself grows exponentially.

`Tableau.expectation` provides a different path for a purely Clifford
stabilizer state: it returns 0 when the Pauli anticommutes with any
stabilizer, otherwise +1 or −1 by checking membership and sign in the
stabilizer group. `CAMPS.expectation` uses this tableau path if the circuit
has no T gates. With T gates it uses the residual MPS path, including after
OFD failures. The exact state vector appears only in the six-qubit
demonstration and tests to verify these answers.

## 7. What is the GF(2) number in the output?

For each **raw** twisted Pauli, record an X-type bit on a qubit if that
letter is X or Y, and zero if I or Z. Stack the bit rows into a `t × N`
binary matrix `B`. This small matrix retains where the raw strings have
X/Y, not their full signs or Z components. The program prints its **left
nullity** `t − rank_GF(2)(B)`, i.e. the number of dependent rows. In the
success example `t=3`, rank 3, nullity 0. In the failure example `t=8`,
rank 2, nullity 6.

Liu and Clark relate this nullity to an exponential factor in the bond
bound for their specified OFD construction. It is a **structural bound**,
not a prediction that the observed bond must equal `2^nullity` and not a
universal obstruction to other CAMPS strategies. Indeed, our failure
example has nullity 6 but observed maximum residual bond only 2.

## 8. What do the timings establish?

The report separates (a) selecting a circuit, which may run many trial
simulations; (b) the binary Pauli propagation and GF(2) analysis;
(c) successful OFD updates; (d) failed-step MPS sums/SVDs; (e) local
expectation contractions; (f) dense-reference calculations; and (g) drawing
figures. `Core CAMPS simulation` times only the computation in `simulate`
and includes the optional intermediate expectation probes; it does not
include selection, plotting, or the final expectation call. `Entire
example` includes those extra operations and printing. The per-T times
include that step's probe contraction, if one was requested.

The Python implementation revisits the preceding Clifford list for each T.
If there are `t` T gates, at most `L` preceding Cliffords per T, and `N`
qubits in each binary Pauli, its straightforward propagation takes at most
`O(t L N)` elementary work in this implementation. The measured scaling
table (`N=16,32,64,128`) tests *that preprocessing only* and constructs
no MPS or dense vector. A tableau with cached Clifford images can change
the preprocessing tradeoff, but that optimization is **not** implemented
in this example. Repeated failed OFD steps can grow χ, making the later
MPS work expensive. Small measured times and a polynomial preprocessing
bound do **not** prove polynomial total simulation time for arbitrary
Clifford+T circuits.

## 9. How to rerun and where to read the code

From the extracted `ThesisCAMPS` directory:

```bash
python -m examples.two_circuits
python -m tests.run_checks
```

In a Jupyter notebook opened inside that directory, use
`%run -m examples.two_circuits`; figures appear below the cell and are
saved in `examples/output/`. Read `camps/simulation.py` for the T movement,
OFD decisions, timing, and observable frame change;
`stabilizers_pbc/pauli.py` for binary Clifford arithmetic;
`stabilizers_pbc/tableau.py` for Pauli measurements and stabilizer
expectations; and `mps/state.py` for sums, SVD cleanup, and contractions.

### Paper context

This walk-through uses the supplied Liu and Clark *CAMPS* paper for the
Clifford-frame representation and OFD, the supplied stabilizer-tableau
paper for binary Clifford updates, and the supplied MPS notes for SVD and
bond dimensions. It explains the **implemented demonstration**, not the
paper's optimization-based disentangler, full PBC magic-state reductions,
or a theorem of efficiency for all circuits.
