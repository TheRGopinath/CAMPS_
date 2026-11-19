"""Li-Chen-Fisher Fig. 4(a)-style plot: half-chain S vs N on log-log axes.
Usage: python loglog_S_vs_N.py path/to/entanglement_vs_p.csv [out.png]"""
import sys, csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]))); out = sys.argv[2] if len(sys.argv) > 2 else 'loglog_S_vs_N.png'
Ns = sorted({int(r['N']) for r in rows})
get = lambda n, p, k: next(float(r[k]) for r in rows if int(r['N']) == n and abs(float(r['p'])-p) < 1e-12)
INK2, GRID = '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
present = {round(float(r['p']), 4) for r in rows}
show = [q for q in (0.0, 0.05, 0.10, 0.13, 0.16, 0.20, 0.30, 0.40) if round(q, 4) in present][:7]
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.facecolor': '#fcfcfb', 'figure.facecolor': '#fcfcfb'})
fig, ax = plt.subplots(figsize=(9, 5.2))
for c, p in zip(SERIES, show):
    S = np.array([get(n, p, 'mean_entropy_bits') for n in Ns]); e = [get(n, p, 'sem_bits') for n in Ns]
    ax.errorbar(Ns, S, yerr=e, color=c, lw=2, marker='o', ms=6, capsize=2, mec='#fcfcfb', mew=1.2, label=f'p = {p:.2f}')
    ax.text(Ns[-1]*1.05, S[-1], f'{p:.2f}', va='center', fontsize=9)
guide = np.array([Ns[0], Ns[-1]])
ax.plot(guide, 0.5*guide*0.98, ls='--', color=INK2, lw=1)
xs = (Ns[0]*Ns[1])**0.5; ax.text(xs, 0.5*xs*1.3, 'slope 1 (volume law)', color=INK2, fontsize=9, rotation=25)
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xticks(Ns); ax.set_xticklabels(map(str, Ns)); ax.set_xlim(Ns[0]*0.9, Ns[-1]*1.3)
ax.set(xlabel='System size N (log)', ylabel='Mean half-chain entropy S (bits, log)',
       title='Half-chain entropy vs N, log-log (cf. Li–Chen–Fisher Fig. 4a)')
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.08, 1), fontsize=9)
ax.grid(color=GRID, lw=0.8, which='both'); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(out, dpi=170); print('saved', out)
