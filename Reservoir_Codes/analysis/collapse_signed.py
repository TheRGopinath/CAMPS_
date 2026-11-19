"""Li-Chen-Fisher Fig. 6(b) layout: signed x = (p - p_c) N^(1/nu) on a linear
axis, y = |S(p,N) - S(p_c,N)| on a log axis. p < p_c falls to the left of 0 and
p > p_c to the right, so the collapsed data form a 'V': down to a dip at x = 0,
then back up. S(p_c,N) is interpolated linearly between the grid points either side of p_c.
Usage: python collapse_signed.py path/to/entanglement_vs_p.csv [p_c] [nu] [out.png]
"""
import sys, csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = sys.argv[1]
pc = float(sys.argv[2]) if len(sys.argv) > 2 else 0.16
nu = float(sys.argv[3]) if len(sys.argv) > 3 else 1.3
out = sys.argv[4] if len(sys.argv) > 4 else 'collapse_signed.png'
rows = list(csv.DictReader(open(src)))
Ns = sorted({int(r['N']) for r in rows})
ps = np.array(sorted({float(r['p']) for r in rows if float(r['p']) <= 0.5}))
val = lambda n, p, k: next(float(r[k]) for r in rows if int(r['N']) == n and abs(float(r['p'])-p) < 1e-12)

INK2, GRID = '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.facecolor': '#fcfcfb', 'figure.facecolor': '#fcfcfb'})
fig, ax = plt.subplots(figsize=(8.5, 5.4))
for c, n in zip(SERIES, Ns):
    S = np.array([val(n, p, 'mean_entropy_bits') for p in ps]); E = np.array([val(n, p, 'sem_bits') for p in ps])
    Spc, Epc = np.interp(pc, ps, S), np.interp(pc, ps, E)
    keep = np.abs(ps - pc) > 1e-9
    x = (ps[keep] - pc) * n**(1/nu); y = np.abs(S[keep] - Spc); err = np.sqrt(E[keep]**2 + Epc**2)
    m = y > 1e-3
    ax.errorbar(x[m], y[m], yerr=np.minimum(err[m], 0.95*y[m]), color=c, lw=1.5, marker='o', ms=6,
                capsize=2, mec='#fcfcfb', mew=1.2, label=f'N = {n}')
ax.axvline(0, color=INK2, lw=1, ls='--')
ax.set_yscale('log')
ax.set(xlabel=r'$(p - p_c)\,N^{1/\nu}$', ylabel=r'$|S(p,N) - S(p_c,N)|$  (bits)',
       title=f'Data collapse, Li–Chen–Fisher Fig. 6(b) layout ($p_c$ = {pc}, $\\nu$ = {nu})')
ax.text(0.02, 0.04, 'volume law\n(p < p_c)', transform=ax.transAxes, color=INK2, va='bottom', fontsize=9)
ax.text(0.98, 0.45, 'area law\n(p > p_c)', transform=ax.transAxes, color=INK2, va='top', ha='right', fontsize=9)
ax.legend(frameon=False, loc='lower right', fontsize=9)
ax.grid(color=GRID, lw=0.8, which='both'); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(out, dpi=170); print('saved', out)
