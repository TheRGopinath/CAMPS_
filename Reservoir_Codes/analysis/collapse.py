"""Li-Chen-Fisher Fig. 6(b)-style data collapse from entanglement_vs_p.csv.

y = |S(p,N) - S(p_c,N)|,  x = (p - p_c) N^(1/nu)
S(p_c,N) is linearly interpolated in p between the two grid points that
bracket p_c (the grid has no point at p_c itself).
Panel (a): literature values p_c=0.16, nu=1.3.  Panel (b): our best fit
(grid search minimising the spread between sizes in log y).
Usage: python collapse.py path/to/entanglement_vs_p.csv [out.png]
"""
import sys, csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open(sys.argv[1]))); out = sys.argv[2] if len(sys.argv) > 2 else 'collapse.png'
Ns = sorted({int(r['N']) for r in rows})
P_MAX = 0.5   # beyond this S ~ 0 for every N, so y just equals S(p_c,N)
ps = np.array(sorted({float(r['p']) for r in rows if float(r['p']) <= P_MAX}))
S = {n: np.array([next(float(r['mean_entropy_bits']) for r in rows if int(r['N']) == n and abs(float(r['p'])-p) < 1e-12) for p in ps]) for n in Ns}
E = {n: np.array([next(float(r['sem_bits']) for r in rows if int(r['N']) == n and abs(float(r['p'])-p) < 1e-12) for p in ps]) for n in Ns}

def points(pc, nu):
    out = {}
    for n in Ns:
        Spc = np.interp(pc, ps, S[n]); Epc = np.interp(pc, ps, E[n])
        keep = np.abs(ps - pc) > 1e-9
        x = (ps[keep] - pc) * n**(1/nu); y = np.abs(S[n][keep] - Spc)
        err = np.sqrt(E[n][keep]**2 + Epc**2)
        out[n] = (x, y, err, ps[keep])
    return out

def cost(pc, nu):
    """Mean squared mismatch in log y between each point and the other sizes' curves."""
    pts = points(pc, nu); total = cnt = 0
    for side in (-1, 1):
        curves = {}
        for n, (x, y, _, _) in pts.items():
            m = (np.sign(x) == side) & (y > 0.05)
            if m.sum() >= 2:
                o = np.argsort(x[m]); curves[n] = (x[m][o], np.log(y[m][o]))
        for ni, (xi, yi) in curves.items():
            for nj, (xj, yj) in curves.items():
                if ni == nj: continue
                inside = (xi >= xj.min()) & (xi <= xj.max())
                if inside.any():
                    total += np.sum((yi[inside] - np.interp(xi[inside], xj, yj))**2); cnt += inside.sum()
    return total / cnt if cnt > 20 else np.inf

pcs = np.arange(0.12, 0.2201, 0.0025); nus = np.arange(0.6, 3.01, 0.02)
C = np.array([[cost(pc, nu) for nu in nus] for pc in pcs])
i, j = np.unravel_index(np.argmin(C), C.shape); best = (pcs[i], nus[j])
# rough uncertainty: parameter range whose cost is within 2x the minimum
ok = C <= 2*C.min()
print(f'best fit: p_c = {best[0]:.3f} (range {pcs[ok.any(1)].min():.3f}-{pcs[ok.any(1)].max():.3f}), '
      f'nu = {best[1]:.2f} (range {nus[ok.any(0)].min():.2f}-{nus[ok.any(0)].max():.2f})')
print(f'cost at literature (0.16, 1.3) = {cost(0.16, 1.3):.4f}; at best fit = {C.min():.4f}')

INK2, GRID = '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.facecolor': '#fcfcfb', 'figure.facecolor': '#fcfcfb'})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, (pc, nu), title in zip(axes, [(0.16, 1.3), best],
                               ['(a) Literature values', '(b) Best fit to your data']):
    for c, (n, (x, y, err, _)) in zip(SERIES, points(pc, nu).items()):
        for side, mk in ((-1, 'o'), (1, 's')):
            m = (np.sign(x) == side) & (y > 0.02)
            ax.errorbar(np.abs(x[m]), y[m], yerr=err[m], fmt=mk, color=c, ms=7, capsize=2,
                        mfc=c if side < 0 else '#fcfcfb', mec=c, mew=1.6,
                        label=f'N = {n}' if side < 0 else None)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set(xlabel=r'$|p - p_c|\,N^{1/\nu}$', title=f'{title}: $p_c$ = {pc:.3f}, $\\nu$ = {nu:.2f}')
    ax.grid(color=GRID, lw=0.8, which='both'); ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel(r'$|S(p,N) - S(p_c,N)|$  (bits)')
axes[0].legend(frameon=False, loc='upper left', title='filled: p < p_c\nopen:   p > p_c', fontsize=9, title_fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=170); print('saved', out)
