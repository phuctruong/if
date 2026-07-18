#!/usr/bin/env python3
"""Drive the ChatGPT IF-Theory thread: send 'Next' until papers 3..14 harvested."""
import json, re, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp_eval import evaluate

TAB = "870A2A66FBE0C6657B0F3B0B762E5B40"
CONVO = "6a5a1d42-f6c4-83ea-a366-d49556560ba4"
OUT = "/home/phuc/projects/if/canon/extracted"
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver.log")

def log(s):
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def ev(expr, timeout=120):
    r = evaluate(TAB, expr, timeout)
    res = r.get("result", {})
    if "exceptionDetails" in res:
        raise RuntimeError(json.dumps(res["exceptionDetails"])[:500])
    return res.get("result", {}).get("value")

def get_state():
    """Return (current_node_id, role, status, end_turn, text) of tip message."""
    expr = """
(async () => {
  const sess = await fetch('/api/auth/session').then(r => r.json());
  const c = await fetch('/backend-api/conversation/%s', {
    headers: {'Authorization': 'Bearer ' + sess.accessToken}
  }).then(r => r.json());
  let nid = c.current_node, m = null;
  // walk up to nearest node with a visible assistant/user message
  while (nid) {
    const n = c.mapping[nid];
    if (n.message && ['assistant','user'].includes(n.message.author.role)) { m = n.message; break; }
    nid = n.parent;
  }
  if (!m) return JSON.stringify({node: c.current_node, role: null});
  const parts = (m.content.parts || []).filter(p => typeof p === 'string');
  return JSON.stringify({node: nid, role: m.author.role,
    status: m.status, end_turn: m.end_turn, text: parts.join('\\n')});
})()
""" % CONVO
    return json.loads(ev(expr, 60))

def send_next():
    expr = """
(async () => {
  const ta = document.querySelector('#prompt-textarea');
  if (!ta) return 'NO_COMPOSER';
  ta.focus();
  document.execCommand('selectAll', false, null);
  document.execCommand('insertText', false, 'Next');
  await new Promise(r => setTimeout(r, 800));
  const btn = document.querySelector('[data-testid="send-button"]') ||
              document.querySelector('button[aria-label="Send prompt"]');
  if (btn && !btn.disabled) { btn.click(); return 'CLICKED'; }
  // fallback: Enter key
  ta.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', code:'Enter', keyCode:13, bubbles:true}));
  return 'ENTER_FALLBACK';
})()
"""
    return ev(expr, 30)

def harvest(text):
    found = []
    blocks = re.findall(r':::writing\{[^}]*\}(.*?)(?=:::(?!writing)|\Z)', text, re.S) or [text]
    for blk in blocks:
        t = blk.strip().rstrip(':').strip()
        mm = re.search(r'\*\*Paper:\*\*\s*(\d+)', t)
        if not mm: continue
        num = int(mm.group(1))
        fn = f"{OUT}/paper-{num:02d}-extracted.md"
        with open(fn, "w") as f:
            f.write(f"<!-- extracted from ChatGPT (driver) on 2026-07-18 -->\n\n{t}\n")
        found.append((num, len(t)))
    return found

def main():
    have = set()
    for f in os.listdir(OUT):
        m = re.match(r'paper-(\d+)-extracted\.md', f)
        if m: have.add(int(m.group(1)))
    log(f"start; have papers: {sorted(have)}")
    target = set(range(0, 15))

    for iteration in range(20):
        if target <= have:
            log("ALL PAPERS HARVESTED"); break
        st = get_state()
        if st["role"] == "assistant" and st.get("status") == "in_progress":
            log("assistant still generating; waiting");
        else:
            prev_node = st["node"]
            r = send_next()
            log(f"sent 'Next' -> {r} (prev tip {prev_node[:8]})")
            if r == 'NO_COMPOSER':
                log("FATAL: composer not found"); break
            time.sleep(10)
        # wait for a NEW finished assistant message
        deadline = time.time() + 20 * 60
        done = False
        while time.time() < deadline:
            try:
                st2 = get_state()
            except Exception as e:
                log(f"poll error: {e}"); time.sleep(20); continue
            if (st2["role"] == "assistant" and st2.get("status") == "finished_successfully"
                    and st2.get("end_turn") and st2["node"] != st.get("node")):
                done = True; break
            time.sleep(20)
        if not done:
            log("TIMEOUT waiting for response; retrying loop"); continue
        got = harvest(st2["text"])
        for num, sz in got:
            have.add(num)
            log(f"harvested paper {num} ({sz} chars)")
        if not got:
            log(f"no paper block in response; first 200 chars: {st2['text'][:200]!r}")
        time.sleep(5)
    log(f"end; have: {sorted(have)}; missing: {sorted(target - have)}")

if __name__ == "__main__":
    main()
