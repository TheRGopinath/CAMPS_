"""(a) Crossing diagnostic: dS = S(N) - S(N/2) vs p for N/2 -> N pairs.
   At p_c, S = alpha ln N + const, so dS = alpha ln 2 for every pair: curves cross.
(b) Volume-law coefficient s(p) from S = s|A| + alpha ln|A| + c (all sizes).
Usage: python crossing_and_s.py entanglement_vs_p.csv [out.png]"""
import csv, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=list(csv.DictReader(open(sys.argv[1]))); out=sys.argv[2] if len(sys.argv)>2 else 'crossing_and_s.png'
Ns=sorted({int(r['N']) for r in rows}); ps=np.array(sorted({float(r['p']) for r in rows}))
g=lambda n,p,k: next(float(r[k]) for r in rows if int(r['N'])==n and abs(float(r['p'])-p)<1e-9)
pairs=[(a,2*a) for a in Ns if 2*a in Ns]
INK2,GRID='#52514e','#e4e3df'; SERIES=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']
plt.rcParams.update({'font.size':10,'axes.edgecolor':INK2,'xtick.color':INK2,'ytick.color':INK2,
                     'axes.facecolor':'#fcfcfb','figure.facecolor':'#fcfcfb'})
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
m=(ps>=0.05)&(ps<=0.30)
for c,(a,b) in zip(SERIES,pairs):
    d=np.array([g(b,p,'mean_entropy_bits')-g(a,p,'mean_entropy_bits') for p in ps[m]])
    e=np.array([np.hypot(g(a,p,'sem_bits'),g(b,p,'sem_bits')) for p in ps[m]])
    a1.errorbar(ps[m],d,yerr=e,color=c,lw=2,marker='o',ms=6,capsize=2,mec='#fcfcfb',mew=1.2,label=f'S(N={b}) − S(N={a})')
a1.axhline(0,color=INK2,lw=1); a1.axvspan(0.14,0.16,color='#f0efec',zorder=0)
a1.set(xlabel='Measurement probability p',ylabel='Entropy gain on doubling N  (bits)',
       title='(a) Doubling test: curves cross at $p_c$')
a1.legend(frameon=False)
LA=np.array(Ns)/2; s=[]; ds=[]
for p in ps:
    S=np.array([g(n,p,'mean_entropy_bits') for n in Ns]); E=np.array([max(g(n,p,'sem_bits'),1e-3) for n in Ns])
    A=np.vstack([LA,np.log(LA),np.ones_like(LA)]).T; W=np.diag(1/E**2)
    cov=np.linalg.inv(A.T@W@A); c=cov@A.T@W@S; s.append(c[0]); ds.append(np.sqrt(cov[0,0]))
a2.errorbar(ps,s,yerr=ds,color=SERIES[0],lw=2,marker='o',ms=6,capsize=2,mec='#fcfcfb',mew=1.2)
a2.axhline(0,color=INK2,lw=1); a2.axvspan(0.14,0.16,color='#f0efec',zorder=0)
a2.set(xlabel='Measurement probability p',ylabel='Volume-law coefficient s  (bits per qubit of A)',
       title=r'(b) $s(p)$ from fit $S = s|A| + \alpha\ln|A| + c$, N = '+', '.join(map(str,Ns)))
for ax in (a1,a2): ax.grid(color=GRID,lw=.8); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig(out,dpi=170); print('saved',out)
