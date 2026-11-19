"""Reproducible Clifford measurement-induced entanglement transition demo.

Run from repository root:
    python -m examples.mipt_clifford

Figures and CSVs are saved in examples/output/mipt/. Change the constants in
CONFIG below to trade runtime against ensemble precision. This is a PHYSICS
baseline with pure stabilizer states; no T gates or CMPS bond dimension yet.
"""
from pathlib import Path
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # notebook-safe: save PNGs even without a display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mipt import run_trajectory, two_qubit_clifford_words

CONFIG = {
    'seed': 20261116,
    'sizes': (8, 12, 16, 24),
    'depth_factor': 3,          # depth = 4*N complete odd/even cycles
    'trajectories': 12,         # increase to 100+ for publication-quality averages
    'probabilities': (0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.70,1.0),
    'example_n': 8,
    'example_depth': 12,
    'example_p': 0.20,
}
OUT = Path(__file__).resolve().parent / 'output' / 'mipt'


def draw_protocol(path, n=8, depth=8, p=.2, seed=19):
    """Draw an actual random brickwork gate/measurement LOCATION realization.

    This figure is a protocol realization, not a plot of Born-rule outcomes.
    Gates labeled C are independent uniform 2-qubit Clifford gates. Small
    orange M boxes denote measurement locations selected with probability p.
    """
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(max(11,depth*1.35),max(4,n*.56+1.8)))
    total = 4*depth
    for q in range(n):
        ax.plot([-.5,total+.5],[q,q],color='#8492a6',lw=1,zorder=0)
        ax.text(-.8,q,f'q{q}',ha='right',va='center',fontsize=9)
    for t in range(depth):
        for parity in (0,1):
            x=4*t+2*parity
            for q in range(parity,n-1,2):
                ax.add_patch(Rectangle((x-.34,q-.34),.68,1.68,facecolor='#b8d7f4',edgecolor='#2464a0',lw=1.1,zorder=2))
                ax.text(x,q+.5,'C',ha='center',va='center',fontsize=8,color='#174b7d',zorder=3)
            for q in range(n):
                if rng.random()<p:
                    ax.add_patch(Rectangle((x+.72,q-.19),.55,.38,facecolor='#ffd28e',edgecolor='#a66300',zorder=4))
                    ax.text(x+1.0,q,'M',ha='center',va='center',fontsize=7,color='#784500',zorder=5)
    ax.set_xlim(-1.4,total+.5);ax.set_ylim(n-.4,-.6)
    ax.set_yticks([])
    ticks = np.arange(0, total, 4)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(i) for i in range(len(ticks))])
    ax.set_xlabel('Full brickwork cycle (odd bonds + measurement + even bonds + measurement)')
    ax.set_title(f'Random Clifford brickwork + stochastic Z measurement locations (N={n}, p={p:.2f})')
    ax.spines[['top','right','left']].set_visible(False)
    fig.tight_layout();fig.savefig(path,dpi=190);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    words=two_qubit_clifford_words()
    print('\n=== CLIFFORD MEASUREMENT-INDUCED ENTANGLEMENT TRANSITION ===')
    print(f'Enumerated {len(words):,} distinct two-qubit Cliffords (modulo global phase).')
    print('Each gate is sampled UNIFORMLY from this group, an exact unitary 2-design.')
    print('Circuit: odd bonds -> stochastic Z measurements -> even bonds -> stochastic Z measurements.')
    print('All trajectories begin in |0>^N. Each measurement outcome is sampled by Born rule.')
    print('Entropy: S(A) = rank_GF2(stabilizer restriction to B) - |B|, in bits.')
    print('NOTE: A finite-N crossover is NOT by itself a thermodynamic critical-point estimate.\n')

    draw_protocol(OUT/'circuit_diagram.png',CONFIG['example_n'],CONFIG['example_depth'],CONFIG['example_p'])
    rng=np.random.default_rng(CONFIG['seed'])
    n=CONFIG['example_n']; depth=CONFIG['depth_factor']*n
    example=run_trajectory(n,depth,CONFIG['example_p'],rng,words,True)
    fig,ax=plt.subplots(figsize=(9,4.2))
    x=np.arange(len(example['history']))/2
    ax.step(x,example['history'],where='post',lw=1.5)
    ax.set(xlabel='Full brickwork cycles',ylabel='Half-chain entropy (bits)',
           title=f'One conditioned quantum trajectory: N={n}, p={CONFIG["example_p"]:.2f}')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'single_trajectory_entropy.png',dpi=190);plt.close(fig)
    print(f'Example trajectory: N={n}, depth={depth}, final S={example["entropy"]} bits, '
          f'measurements={example["measurements"]}, random outcomes={example["random_outcomes"]}')

    rows=[]
    for n in CONFIG['sizes']:
        depth=CONFIG['depth_factor']*n
        for p in CONFIG['probabilities']:
            ent=[];counts=[]
            for _ in range(CONFIG['trajectories']):
                res=run_trajectory(n,depth,float(p),rng,words)
                ent.append(res['entropy']);counts.append(res['measurements'])
            ent=np.asarray(ent,dtype=float)
            mean=float(ent.mean()); sem=float(ent.std(ddof=1)/np.sqrt(len(ent))) if len(ent)>1 else 0.
            rows.append({'N':n,'depth_cycles':depth,'p':float(p),'trajectories':len(ent),
                         'mean_entropy_bits':mean,'sem_bits':sem,
                         'mean_entropy_density':mean/(n/2),
                         'mean_measurement_count':float(np.mean(counts))})
            print(f'N={n:2d} depth={depth:3d} p={p:.2f} '
                  f'<S_half>={mean:6.3f} +/- {sem:.3f} bits; '
                  f'<S>/(N/2)={mean/(n/2):.3f}')

    with (OUT/'entanglement_vs_p.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    fig,ax=plt.subplots(figsize=(9,5.5))
    for n in CONFIG['sizes']:
        group=[r for r in rows if r['N']==n]
        ax.errorbar([r['p'] for r in group],[r['mean_entropy_bits'] for r in group],
                    yerr=[r['sem_bits'] for r in group],marker='o',capsize=2,label=f'N={n}')
    ax.set(xlabel='Measurement probability p',ylabel='Mean half-chain entropy (bits)',
           title='Monitored Clifford circuit: entanglement versus measurement rate')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'entropy_vs_probability.png',dpi=190);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5.5))
    for n in CONFIG['sizes']:
        group=[r for r in rows if r['N']==n]
        ax.errorbar([r['p'] for r in group],[r['mean_entropy_density'] for r in group],
                    yerr=[r['sem_bits']/(n/2) for r in group],marker='o',capsize=2,label=f'N={n}')
    ax.set(xlabel='Measurement probability p',ylabel='Mean entropy / (N/2) (bits per qubit)',
           title='Finite-size comparison: normalized half-chain entropy')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'entropy_density_vs_probability.png',dpi=190);plt.close(fig)

    with (OUT/'RUN_NOTES.txt').open('w') as f:
        f.write('Clifford MIPT experiment\n=======================\n')
        for k,v in CONFIG.items():f.write(f'{k}: {v}\n')
        f.write('\nTime step: odd Clifford gates, independent onsite Z measurements, '
                'even Clifford gates, independent onsite Z measurements.\n')
        f.write('Open boundaries; exact uniform two-qubit Clifford group samples.\n')
        f.write('Born-rule outcomes conditioned, pure trajectories.\n')
        f.write('Entropy from GF(2) stabilizer restriction, not MPS Schmidt values.\n')
        f.write('Uncertainty = standard error over independent trajectories.\n')
        f.write('These are finite-size numerical demonstrations, not fitted critical exponents.\n')
    print(f'\nSaved: {OUT}/')
    for filename in ['circuit_diagram.png','single_trajectory_entropy.png',
                     'entropy_vs_probability.png','entropy_density_vs_probability.png',
                     'entanglement_vs_p.csv','RUN_NOTES.txt']:
        print(' -',filename)
    print(f'Elapsed time: {time.perf_counter()-start:.1f} s')
    return rows


if __name__=='__main__':main()
