#!/usr/bin/env bash
# IF repo verify — fast integrity gate (target < 60s)
set -e
cd "$(dirname "$0")/.."
fail=0

echo "== 1. notebook code cells execute (00 + 04) =="
python3 - <<'PY'
import json
for nb_path in ('notebooks/00_if_prediction_contract.ipynb',
                'notebooks/04_if_causal_work_threshold.ipynb'):
    nb = json.load(open(nb_path))
    src = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    g = {'__name__':'__main__'}
    exec(src, g)
    print(f"  OK {nb_path}")
PY

echo "== 2. no ChatGPT artifacts in canonical papers =="
if grep -rl "citeturn\|:::writing" canon/papers/ 2>/dev/null; then echo "  FAIL: artifacts found"; fail=1; else echo "  OK"; fi

echo "== 3. layer-leak audit (science docs must not claim God/MaxLove) =="
# Allowed: quoted "MaxLove" with pointer, firewall mentions. Flag bare theological claims.
leaks=$(grep -rn "God is\|God exists\|proves God\|God's" canon/00-foundations canon/10-agency canon/20-cosmology canon/papers notebooks 2>/dev/null | grep -v "never\|not \|firewall\|forbidden\|LAYER" || true)
if [ -n "$leaks" ]; then echo "$leaks"; echo "  FAIL: possible layer leaks"; fail=1; else echo "  OK"; fi

echo "== 4. every notebook has a CONTRACT cell =="
python3 - <<'PY'
import json, glob, sys
bad = []
for f in sorted(glob.glob('notebooks/*.ipynb')):
    nb = json.load(open(f))
    first_md = next((''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown'), '')
    if 'CONTRACT' not in first_md and 'contract' not in first_md.lower():
        bad.append(f)
if bad: print("  FAIL:", bad); sys.exit(1)
print(f"  OK ({len(glob.glob('notebooks/*.ipynb'))} notebooks)")
PY

echo "== 6. falsified quantities not used as live (rung-641 exclusions) =="
python3 - <<'PY'
import glob, re, sys
DEAD = [r'eta\*', r'Theta\*', r'Upsilon_IF', r'\u03a5_IF', r'\u03b7\\?\*', r'\u0398\\?\*']
CTX = re.compile(r'falsif|kill|dead|retire|not universal|scatter|superseded|FAILED|FALSIFIED|'
                 r'exclusion|audit|obstruction|limitation|candidate|conjectur|restated|status update|'
                 r'did not pursue|pre-commit|stop rule|per-family', re.I)
WINDOW = 12   # lines of surrounding context, plus the whole enclosing section heading chain
bad = []
for f in glob.glob('canon/**/*.md', recursive=True):
    if 'extracted' in f or 'panels' in f: continue
    lines = open(f).read().split('\n')
    for i, line in enumerate(lines):
        if not any(re.search(d, line) for d in DEAD): continue
        lo, hi = max(0, i-WINDOW), min(len(lines), i+WINDOW+1)
        ctx = '\n'.join(lines[lo:hi])
        heads = '\n'.join(l for l in lines[:i+1] if l.startswith('#'))[-800:]
        if not (CTX.search(ctx) or CTX.search(heads)):
            bad.append(f"{f}:{i+1}: {line.strip()[:70]}")
if bad:
    print("  FAIL: falsified quantity used as live (no falsification context):")
    for b in bad[:8]: print("   ", b)
    sys.exit(1)
print("  OK (all uses carry falsification/limitation context)")
PY

echo "== 5. kill log non-empty and current =="
grep -q "2026-07-18" SCOREBOARD.md && echo "  OK" || { echo "  FAIL"; fail=1; }

[ $fail -eq 0 ] && echo "VERIFY: GREEN" || { echo "VERIFY: RED"; exit 1; }
