"""Matplotlib circuit diagram plus a compact OFD result panel."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_circuit(circuit, n, title, history, output):
    """Draw wires in qubit-index order, gates in time order, and OFD outcomes."""
    width = len(circuit)
    fig, (ax, summary) = plt.subplots(
        2, 1, figsize=(max(10, 0.64*width + 2), 3.2 + 0.42*len(history)),
        gridspec_kw={"height_ratios": [max(3, 0.6*n), max(1.6, 0.38*len(history))]},
        constrained_layout=True)
    for q in range(n):
        ax.plot((0.4, width+0.6), (q, q), color="#344054", lw=1.3, zorder=0)
        ax.text(0.27, q, f"q{q}", ha="right", va="center", fontsize=10)
    for index, gate in enumerate(circuit, start=1):
        if gate.kind in ("CX", "CP"):
            ax.plot([index, index], [gate.a, gate.b], color="#205c98", lw=1.5)
            ax.plot(index, gate.a, "o", color="#205c98", markersize=6)
            label = "X" if gate.kind == "CX" else gate.axis
            ax.add_patch(Rectangle((index-.23, gate.b-.23), .46, .46,
                                   facecolor="white", edgecolor="#205c98", lw=1.4, zorder=2))
            ax.text(index, gate.b, label, ha="center", va="center", color="#205c98", zorder=3)
        else:
            color = "#e9c46a" if gate.kind == "T" else "#dcebf5"
            ax.add_patch(Rectangle((index-.26, gate.a-.24), .52, .48,
                                   facecolor=color, edgecolor="#344054", lw=1, zorder=2))
            ax.text(index, gate.a, gate.kind, ha="center", va="center", fontsize=9, zorder=3)
    ax.set(xlim=(0, width+1), ylim=(n-.35, -.85), title=title)
    ax.axis("off")
    summary.axis("off")
    rows = [f"T{i}: {pauli}  |  OFD {status}  |  max bond {bond}"
            for i, (status, pauli, bond) in enumerate(history, 1)]
    summary.text(0.02, .98, "Transformed T gates and OFD steps\n" + "\n".join(rows),
                 transform=summary.transAxes, va="top", family="monospace", fontsize=9,
                 linespacing=1.5)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.show()  # Jupyter displays it under the cell; scripts save the PNG above.
    plt.close(fig)
    return output
