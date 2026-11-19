"""Extract the volume-law coefficient s(p) from entanglement_vs_p.csv.

s(p) is the slope of S_half versus the half-chain length L_A = N/2, i.e. the
quantity that tends to the advisor's sketch as N -> infinity:
  s > 0 for p < p_c (volume law), s = 0 for p > p_c (area law).
Usage: python s_of_p.py path/to/entanglement_vs_p.csv [out.png]
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else 's_of_p.png'
rows = list(csv.DictReader(open(src)))
ps = sorted({float(r['p']) for r in rows})
Ns = sorted({int(r['N']) for r in rows})
def get(n, p, key):
    return next(float(r[key]) for r in rows if int(r['N']) == n and abs(float(r['p'])-p) < 1e-12)

FIT_N = Ns[-3:] if len(Ns) >= 3 else Ns     # fit over the three largest sizes (smallest corrections)
if len(FIT_N) < 2: sys.exit('need at least two system sizes to fit a slope')
s, ds = [], []
for p in ps:
    LA = np.array([n/2 for n in FIT_N]); S = np.array([get(n, p, 'mean_entropy_bits') for n in FIT_N])
    err = np.array([max(get(n, p, 'sem_bits'), 1e-3) for n in FIT_N])
    W = np.diag(1/err**2); A = np.vstack([LA, np.ones_like(LA)]).T
    cov = np.linalg.inv(A.T @ W @ A); coef = cov @ A.T @ W @ S
    s.append(coef[0]); ds.append(np.sqrt(cov[0, 0]))
    print(f'p={p:.2f}  s={coef[0]:+.3f} +/- {np.sqrt(cov[0,0]):.3f} bits per qubit of A')

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': '#fcfcfb',
                     'figure.facecolor': '#fcfcfb'})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))

a1.axvspan(0.15, 0.20, color='#f0efec', zorder=0)
a1.errorbar(ps, s, yerr=ds, color=SERIES[0], lw=2, marker='o', ms=7, capsize=3,
            mec='#fcfcfb', mew=1.5, zorder=3)
a1.axhline(0, color=INK2, lw=1)
a1.text(0.175, 0.93, 'expected p_c\n(~0.16)', ha='center', va='top', color=INK2, fontsize=9)
a1.set(xlabel='Measurement probability p', ylabel='Volume-law coefficient s = dS / d|A|  (bits)',
       title=f'(a) Volume-law coefficient, fitted over N = {", ".join(map(str, FIT_N))}')
a1.set_ylim(-0.08, 1.05)

show = [q for q in (0.05, 0.10, 0.15, 0.16, 0.20, 0.30) if any(abs(q-x) < 1e-9 for x in ps)][:5]
for c, p in zip(SERIES, show):
    S = [get(n, p, 'mean_entropy_bits') for n in Ns]; e = [get(n, p, 'sem_bits') for n in Ns]
    a2.errorbar(Ns, S, yerr=e, color=c, lw=2, marker='o', ms=7, capsize=3, mec='#fcfcfb',
                mew=1.5, label=f'p = {p:.2f}')
    a2.text(Ns[-1]*1.04, S[-1], f'p={p:.2f}', color=INK, va='center', fontsize=9)
a2.set_xscale('log', base=2); a2.set_xticks(Ns); a2.set_xticklabels(map(str, Ns))
a2.set_xlim(Ns[0]*0.9, Ns[-1]*1.35)
a2.set(xlabel='System size N (log scale)', ylabel='Mean half-chain entropy (bits)',
       title='(b) Growing (volume law) vs flat (area law)')
a2.legend(frameon=False, loc='upper left')
for ax in (a1, a2):
    ax.grid(color=GRID, lw=0.8); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(out, dpi=170)
print('saved', out)
