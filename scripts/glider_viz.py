import numpy as np, matplotlib, json
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
exec(open('universe.py').read().split("if __name__")[0])
_g = open('gliders.py').read()
exec(_g[_g.index('GLIDER = '):_g.index('print("Glider survival')])

FIG='/home/phuc/projects/if/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.facecolor':'#0e0f13','figure.facecolor':'#0e0f13',
  'text.color':'#e6e6e6','axes.labelcolor':'#c9c9c9','xtick.color':'#8a8a8a','ytick.color':'#8a8a8a','axes.edgecolor':'#2a2c35'})
res_cmap = LinearSegmentedColormap.from_list('res',['#0e0f13','#10243a','#1d5c7a','#39a0a8'])

# fine sweep for the survival boundary
dists = list(range(20, 50, 2))
surv, err = [], []
for d in dists:
    v = np.array([trial(d, seed=s)[0] for s in range(12)])
    surv.append(v.mean()); err.append(v.std(ddof=1)/np.sqrt(len(v)))
surv, err = np.array(surv), np.array(err)
# boundary = first distance where mean survival drops below half of max
half = surv.max()/2
bidx = np.argmax(surv < half)
boundary = dists[bidx] if surv[bidx] < half else None
print("distances:", dists); print("survival:", surv.round(1).tolist())
print("boundary:", boundary)

# a trajectory to draw
u = Universe(seed=3, inflow=4.0, hotspot_sigma=40.0)
u.heat += E_BIRTH*u.A.sum(); u.A[:]=0
for _ in range(30): u.step()
cy,cx = int(u.src[0]), int(u.src[1])
seed_glider(u, (cy+10)%(u.n-3), cx%(u.n-3)); u.heat -= E_BIRTH*u.A.sum()
traj=[]; frames={}
for t in range(220):
    if u.A.sum(): traj.append(np.array(np.where(u.A)).mean(1))
    if t in (0, 80, 160, 219): frames[t]=(u.A.copy(), u.R.copy())
    u.step()
traj=np.array(traj)

fig = plt.figure(figsize=(15,6.4))
gs = fig.add_gridspec(1,3, width_ratios=[1.15,1.15,1.5], wspace=0.22)

ax = fig.add_subplot(gs[0,0])
A,R = frames[219]
ax.imshow(R, cmap=res_cmap, vmin=0, vmax=3.0)
ax.plot(traj[:,1], traj[:,0], color='#ffd479', lw=1.3, alpha=0.9)
ax.scatter([traj[0,1]],[traj[0,0]], s=42, c='#7fd4d8', zorder=3, label='start')
ax.scatter([traj[-1,1]],[traj[-1,0]], s=42, c='#ff6b6b', zorder=3, label='t=220')
ax.set_title('A glider that survives\n(travels ~100 cells, pays every birth)', fontsize=10.5, color='#e6e6e6')
ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8, facecolor='#151720', edgecolor='#2a2c35', labelcolor='#c9c9c9')

ax = fig.add_subplot(gs[0,1])
u2 = Universe(seed=3, inflow=4.0, hotspot_sigma=40.0)
u2.heat += E_BIRTH*u2.A.sum(); u2.A[:]=0
for _ in range(30): u2.step()
seed_glider(u2, (int(u2.src[0])+55)%(u2.n-3), int(u2.src[1])%(u2.n-3)); u2.heat -= E_BIRTH*u2.A.sum()
A2 = u2.A.copy()
for _ in range(12): u2.step()
ax.imshow(u2.R, cmap=res_cmap, vmin=0, vmax=3.0)
ys,xs = np.where(A2); ax.scatter(xs, ys, s=26, c='#ff6b6b', marker='x')
ax.set_title('A glider that starves\n(seeded 55 cells out — dead in 10 steps)', fontsize=10.5, color='#e6e6e6')
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[0,2])
ax.errorbar(dists, surv, yerr=err, color='#ffd479', lw=1.8, marker='o', ms=4, capsize=3)
if boundary: 
    ax.axvline(boundary, color='#ff6b6b', ls='--', lw=1.2)
    ax.text(boundary+0.6, surv.max()*0.55, f'survival boundary\n≈ {boundary} cells', color='#ff8f8f', fontsize=9)
ax.set_xlabel('seeding distance from resource hotspot (cells)')
ax.set_ylabel('steps survived (max 220)')
ax.set_title('Selection with no fitness function in the rules', fontsize=11, loc='left', color='#e6e6e6')
ax.grid(alpha=0.12)

fig.suptitle('Emergent selection: mobile structures survive only within reach of free energy', fontsize=13.5, color='#f0f0f0', y=1.02)
fig.savefig(f'{FIG}/glider-selection.png', dpi=140, bbox_inches='tight', facecolor='#0e0f13')
json.dump({'distances':dists,'survival':surv.tolist(),'sem':err.tolist(),'boundary':boundary},
          open('/home/phuc/projects/if/evidence/glider_selection_2026_07_18.json','w'), indent=1)
print('wrote', f'{FIG}/glider-selection.png')
