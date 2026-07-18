"""Render the IF artificial-universe results as videos -> ~/projects/if/videos/.

Three films, each tied to a sealed result:
  1 mobility-regime.mp4    — random soup grows movers (sealed D4 regime, seed 101);
                             magenta trails = tracked emergent mobile structures
  2 starved-vs-abundant.mp4 — same rules, same seed: E_BIRTH=1.0/inflow=0.9 freezes
                             into still lifes; E_BIRTH=0.25/inflow=12 grows movers
  3 scramble-ignition.mp4  — the audit confound, watchable: intact mover vs its
                             count-preserving scrambled twin igniting a growth explosion
                             (seed 42, the W_C = -39.4 audit row)

Pure numpy frames piped to ffmpeg (rawvideo). Deterministic replay: pass 1 tracks,
pass 2 renders the identical universe with overlays.
"""
import os, copy, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mobility_search import UniverseX, RULES, track_universe, classify

VIDEOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'videos')
os.makedirs(VIDEOS, exist_ok=True)
SCALE, FPS = 4, 24
try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    FONT_SM = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
except OSError:
    FONT = FONT_SM = ImageFont.load_default()

CFG_MOBILE = dict(born=(3,), survive=(2, 3), e_birth=0.25, e_maint=0.01,
                  inflow=12.0, sigma=40.0, density=0.15)
CFG_STARVED = dict(born=(3,), survive=(2, 3), e_birth=1.0, e_maint=0.01,
                   inflow=0.9, sigma=14.0, density=0.12)


def rgb_frame(A, R, trail=None, r_max=3.0):
    """128x128 -> HxWx3 uint8. Resource = deep navy->teal; cells = warm gold;
    trail = magenta glow."""
    r = np.clip(R / r_max, 0, 1)[..., None]
    base = (np.array([8, 10, 28]) * (1 - r) + np.array([16, 95, 115]) * r)
    out = base.copy()
    if trail is not None:
        tr = np.clip(trail, 0, 1)[..., None]
        out = out * (1 - 0.85 * tr) + np.array([255, 60, 190]) * 0.85 * tr
    cells = A.astype(bool)
    out[cells] = np.array([255, 221, 130])
    return out.astype(np.uint8)


def upscale(img):
    return np.kron(img, np.ones((SCALE, SCALE, 1), dtype=np.uint8))


def annotate(img_arr, title, sub):
    im = Image.fromarray(img_arr)
    d = ImageDraw.Draw(im)
    d.text((10, 8), title, fill=(255, 255, 255), font=FONT,
           stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((10, 34), sub, fill=(200, 200, 200), font=FONT_SM,
           stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def encoder(path, w, h, fps=FPS):
    return subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', path],
        stdin=subprocess.PIPE)


def film_mobility(seed=101, steps=600):
    """Film 1: the sealed mobility regime, mover trails overlaid via replay."""
    u1 = UniverseX(seed=seed, **CFG_MOBILE)
    tracks = track_universe(u1, steps)
    mobile, _ = classify(tracks)
    # per-step list of mover COM cells (rounded) for the trail buffer
    hits = [[] for _ in range(steps + 1)]
    for tr in mobile:
        for i, p in enumerate(tr.path):
            t = tr.t0 + i
            if t <= steps:
                y, x = int(round(p[0])) % u1.n, int(round(p[1])) % u1.n
                hits[t].append((y, x))
    u = UniverseX(seed=seed, **CFG_MOBILE)   # deterministic replay
    trail = np.zeros((u.n, u.n))
    n_mob = len(mobile)
    path = os.path.join(VIDEOS, 'mobility-regime.mp4')
    enc = encoder(path, u.n * SCALE, u.n * SCALE)
    for t in range(steps + 1):
        if t > 0:
            u.step()
        trail *= 0.94
        for (y, x) in hits[t]:
            trail[max(y-1, 0):y+2, max(x-1, 0):x+2] = 1.0
        fr = upscale(rgb_frame(u.A, u.R, trail))
        fr = annotate(fr, 'THE MOBILITY REGIME',
                      f't={t:3d}  B3/S23 E_BIRTH=0.25 inflow=12  '
                      f'{n_mob} movers (magenta trails)')
        enc.stdin.write(fr.tobytes())
    enc.stdin.close(); enc.wait()
    print(f"wrote {path}")


def film_starved_vs_abundant(seed=7, steps=400):
    """Film 2: same rules + seed, two energy economies, side by side."""
    ua = UniverseX(seed=seed, **CFG_STARVED)
    ub = UniverseX(seed=seed, **CFG_MOBILE)
    n, gap = ua.n, 2
    w = (n * 2 + gap) * SCALE
    path = os.path.join(VIDEOS, 'starved-vs-abundant.mp4')
    enc = encoder(path, w, n * SCALE)
    for t in range(steps + 1):
        if t > 0:
            ua.step(); ub.step()
        fa, fb = rgb_frame(ua.A, ua.R), rgb_frame(ub.A, ub.R)
        strip = np.full((n, gap, 3), 30, dtype=np.uint8)
        fr = upscale(np.concatenate([fa, strip, fb], axis=1))
        fr = annotate(fr, 'STARVED                                    '
                          '                                   ABUNDANT',
                      f't={t:3d}   left: E_BIRTH=1.0 inflow=0.9 -> still lifes     '
                      f'right: E_BIRTH=0.25 inflow=12 -> movers  (same rules, same seed)')
        enc.stdin.write(fr.tobytes())
    enc.stdin.close(); enc.wait()
    print(f"wrote {path}")


def film_scramble(seed=42, warmup=200, window=100, T=100):
    """Film 3: intact mover vs scrambled twin (the audit's confound, visible)."""
    from mover_audit import live_movers_at_checkpoint
    u, movers = live_movers_at_checkpoint(seed)
    mv = movers[0]                      # the W_C = -39.4 row of the sealed audit
    ys, xs = mv['cys'], mv['cxs']
    bb = np.zeros((u.n, u.n), bool)
    bb[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    ui, us = copy.deepcopy(u), copy.deepcopy(u)
    us.step(scramble_mask=bb)
    n, gap = u.n, 2
    w = (n * 2 + gap) * SCALE
    y0, y1 = ys.min() * SCALE, (ys.max() + 1) * SCALE
    x0, x1 = xs.min() * SCALE, (xs.max() + 1) * SCALE
    path = os.path.join(VIDEOS, 'scramble-ignition.mp4')
    enc = encoder(path, w, n * SCALE, fps=15)
    for t in range(T + 1):
        if t > 0:
            ui.step(); us.step()
        fa, fb = rgb_frame(ui.A, ui.R), rgb_frame(us.A, us.R)
        strip = np.full((n, gap, 3), 30, dtype=np.uint8)
        fr = upscale(np.concatenate([fa, strip, fb], axis=1))
        im = Image.fromarray(fr); d = ImageDraw.Draw(im)
        for xoff in (0, (n + gap) * SCALE):
            d.rectangle([x0 + xoff, y0, x1 + xoff, y1], outline=(255, 60, 60), width=2)
        fr = np.asarray(im)
        fr = annotate(fr, 'INTACT MOVER                                '
                          '                        SCRAMBLED TWIN',
                      f't={t:3d}   same universe, same cells — right box shuffled once at t=0. '
                      f'Sealed audit W_C = -39.4: the corpse out-harvests the mover.')
        enc.stdin.write(fr.tobytes())
    enc.stdin.close(); enc.wait()
    print(f"wrote {path}")


if __name__ == '__main__':
    film_mobility()
    film_starved_vs_abundant()
    film_scramble()
