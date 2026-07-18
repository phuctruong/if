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

echo "== 5. kill log non-empty and current =="
grep -q "2026-07-18" SCOREBOARD.md && echo "  OK" || { echo "  FAIL"; fail=1; }

[ $fail -eq 0 ] && echo "VERIFY: GREEN" || { echo "VERIFY: RED"; exit 1; }
