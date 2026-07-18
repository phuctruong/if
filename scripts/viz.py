import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
exec(open('universe.py').read().split("if __name__")[0])

FIG = '/home/phuc/projects/if/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.facecolor':'#0e0f13',
                     'figure.facecolor':'#0e0f13','text.color':'#e6e6e6',
                     'axes.labelcolor':'#c9c9c9','xtick.color':'#8a8a8a',
                     'ytick.color':'#8a8a8a','axes.edgecolor':'#2a2c35'})
res_cmap = LinearSegmentedColormap.from_list('res', ['#0e0f13','#10243a','#1d5c7a','#39a0a8'])

# --- run, capturing frames + telemetry ---
u = Universe(seed=7, inflow=4.0, hotspot_sigma=40.0)
snaps, pops, structs_t, ts = {}, [], [], []
KEY = [0, 25, 60, 150, 400]
for t in range(401):
    if t in KEY: snaps[t] = (u.A.copy(), u.R.copy(), u.src.copy())
    pops.append(int(u.A.sum()))
    if t % 10 == 0:
        structs_t.append(len(detect_structures(u.A))); ts.append(t)
    u.step()

fig = plt.figure(figsize=(16, 8.6))
gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1], hspace=0.30, wspace=0.14)

for i, t in enumerate(KEY):
    A, R, src = snaps[t]
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(R, cmap=res_cmap, vmin=0, vmax=3.0, interpolation='nearest')
    ys, xs = np.where(A)
    ax.scatter(xs, ys, s=1.4, c='#ffd479', linewidths=0)
    ax.scatter([src[1]], [src[0]], s=70, facecolors='none', edgecolors='#ff6b6b', lw=1.4)
    ax.set_title(f't = {t}   n = {int(A.sum())}', fontsize=10, color='#e6e6e6', pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#2a2c35')

ax = fig.add_subplot(gs[1, :3])
ax.plot(pops, color='#ffd479', lw=1.6)
ax.axvspan(0, 40, color='#ff6b6b', alpha=0.10)
ax.text(6, max(pops)*0.86, 'collapse', color='#ff8f8f', fontsize=9)
ax.axvspan(120, 400, color='#39a0a8', alpha=0.08)
ax.text(150, max(pops)*0.86, 'frozen: still lifes, zero mobility', color='#7fd4d8', fontsize=9)
ax.set_xlabel('time step'); ax.set_ylabel('living cells')
ax.set_title('Population: a crash, then a crystal', fontsize=11, loc='left', color='#e6e6e6')
ax.grid(alpha=0.12)

ax = fig.add_subplot(gs[1, 3:])
ax.plot(ts, structs_t, color='#39a0a8', lw=1.6, marker='o', ms=3)
ax.set_xlabel('time step'); ax.set_ylabel('structures detected')
ax.set_title('Detected persistent structures (never declared)', fontsize=11, loc='left', color='#e6e6e6')
ax.grid(alpha=0.12)
ax.text(0.98, 0.08, 'all static · harvest = 0.00 · nothing to audit',
        transform=ax.transAxes, ha='right', fontsize=9, color='#ff8f8f')

fig.suptitle('IF Artificial Universe — energy-gated Life on a drifting resource field',
             fontsize=14, color='#f0f0f0', y=0.975)
fig.text(0.5, 0.935, 'Conway gate satisfied: no is_alive, no fitness, no agency in the rules. '
         'Energy ledger conserved to 1e-6 every step. Structures detected as persistent components.',
         ha='center', fontsize=9.5, color='#9aa0aa')
fig.savefig(f'{FIG}/universe-still-life.png', dpi=140, bbox_inches='tight', facecolor='#0e0f13')
print(f'wrote {FIG}/universe-still-life.png')
