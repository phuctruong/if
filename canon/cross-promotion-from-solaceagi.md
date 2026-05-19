# Cross-Promotion: solaceagi.com → IF Theory

**Sealed:** 2026-05-19 (operator directive: "use solaceagi.com to help promote phuc.net and IF theories")
**Authorization:** 65537
**Audience:** Solace operators, Hana (the outreach drafter), phuc.net editors

---

## The strategy in one sentence

**solaceagi.com is the commercial megaphone driving attention to IF Theory and
phuc.net; IF Theory is the intellectual foundation underpinning Solace's AGI
framing.** The relationship is structural, not opportunistic — Solace's
"informational field AI" claim is *literally* IF Theory's claim made
operational. Cross-promotion is honest because the framing stands alone.

---

## The five concrete cross-promotion mechanics

### 1. solaceagi.com footer — every page

Every page on solaceagi.com carries a footer line:

> *Built on IF Theory — read more at [phuc.net/if-theory](https://phuc.net/if-theory)*

Implementation: backend template footer partial in
`~/projects/solace-agi/app/templates/`. One edit, propagates everywhere.
Lightweight, persistent, non-intrusive. Honest framing — does not claim IF
Theory is proven, just that Solace is built on it as a foundation.

### 2. AGI definition page — `informational-field-ai-framing.md`

Solace's canonical AGI framing lives at
`~/projects/solace-hub/canon/standards/informational-field-ai-framing.md` and
its definition is:

> *AGI is a breathing informational field trying to get bigger with meaning.*

The "informational" load-bearing word is *exactly* IF Theory's claim — that
information is the first force, not derived from energy or matter. The
framing page must carry a "Why this is more than metaphor" subsection that
links to:

- `github.com/phuctruong/if` — the technical research repo
- `phuc.net/if-theory` — the long-form accessible primer
- `~/projects/if/canon/proposal-for-if.md` — the call for falsifiers

Honest disclosure in that subsection: cite IF Theory as a research program
in active falsifiability testing, not as proven physics. The metaphor stands
alone if IF Theory turns out to be wrong; that decoupling is intentional.

### 3. Hana outreach footnote — Pre-Series-A AI infra

Hana (the outreach-drafting worker on solaceagi.com) drafts to Pre-Series-A
AI-infrastructure rounds include a footnote:

> *Solace's canon-update-RSI thesis is structurally derivable from IF Theory's
> compression-is-cognition claim: information density Φ(r) on a canonical-prime-addressed
> substrate is what makes external-substrate growth compound faster than
> internal-weight-update growth. The technical foundation is open at
> github.com/phuctruong/if.*

This is load-bearing for the AI-infra audience because it gives them the
*structural reason* canon-update-RSI beats weight-update-RSI — not a vibes
argument. The honest discipline: footnote it, don't headline it. The
foundation is research-stage; the operational system on solaceagi.com works
whether or not the foundation is fully closed.

### 4. Newsletter + blog cadence on phuc.net

phuc.net is the long-form publication channel. The cadence:

- **Quarterly IF Theory update** — one article per quarter tracking technical
  progress (Mersenne lemma closures, new geo canon papers, external
  reproductions, falsifier attempts).
- **Cross-post from solaceagi.com blog** — anytime Solace ships a result
  that *operationalizes* an IF Theory claim (e.g., new OCI lift, new
  canonical-prime substrate channel), phuc.net cross-posts with the IF
  Theory link.
- **Falsifier-of-the-quarter** — when an external researcher attempts a
  kill-shot (succeeds or fails), feature them on phuc.net. The science
  community sees IF Theory inviting falsification, not hiding from it.

### 5. Customer-twin "Why Solace" pages

Every active solaceagi.com customer-twin site (gatan, metalmark, simplemdg,
maxsalesgroup, future) gets a short "Why Solace" page citing IF Theory as
substrate. Template:

> *Solace runs on a thermodynamic informational substrate — a research
> program by our founder, Phuc Truong, called IF Theory
> ([github.com/phuctruong/if](https://github.com/phuctruong/if)). You don't
> need to follow the physics to use Solace, but the structural reason a
> Solace fleet compounds across your tenant and ours is the same reason
> the universe's information density follows the prime number theorem:
> canonical-prime-addressed substrates accrete faster than energy-based
> substrates. That's the bet.*

Operator-approves the exact copy per customer. The structural claim survives
even if the physics is wrong — it is a metaphor that IF Theory happens to
make precise.

---

## The honesty discipline (non-negotiable)

- **Never overclaim IF Theory through Solace channels.** Always cite it as a
  research program, never as proven physics.
- **The BAO 2.1σ tension travels with every citation** of the DESI BAO global
  fit — even in marketing copy. χ²/dof = 1.72, p = 0.034. Same discipline as
  inside the IF repo.
- **POSTDICTION flags travel.** If Solace marketing copy ever references
  hubble_tension / s8_tension / cmb_cold_spot, it must say "shape, not
  amplitude" the same way the source files do.
- **The Mersenne uniqueness lemma is OPEN.** Never claim it is closed in any
  solaceagi.com page, blog post, or outreach draft.
- **Solace does not depend on IF Theory being right.** The framing decouples
  cleanly. If IF Theory turns out to be wrong, the Solace product still
  works; we adjust the "Why Solace" copy.
- **Cite "the substrate is the proof"** sparingly. It is the operator's
  framing; it works for the operator's audience. For physicists, it is
  insufficient — they need the falsifiers list.

---

## What this is NOT

- **Not a sales pitch for IF Theory.** IF Theory is OSS research. There is
  nothing to sell. The megaphone serves the *research community*, not the
  Solace revenue line.
- **Not academic kowtowing.** Per operator memory 2026-05-17 (no academic
  deference): we don't soften IF Theory's claims to make tenured reviewers
  comfortable. We invite falsification cleanly.
- **Not a dependency.** If solaceagi.com goes away, IF Theory continues.
  If IF Theory is falsified, solaceagi.com adjusts its "Why Solace" copy
  and continues. The bidirectional promotion is healthy because both
  artifacts stand on their own structural ground.

---

## How to verify the strategy is shipping

```bash
# 1. footer present?
grep -l "Built on IF Theory" ~/projects/solace-agi/app/templates/*.html

# 2. AGI framing page cross-links IF?
grep -E "if-theory|github.com/phuctruong/if" ~/projects/solace-hub/canon/standards/informational-field-ai-framing.md

# 3. Hana drip footnote in Pre-Series-A draft templates?
grep -r "IF Theory" ~/projects/solace-hub/workers/hana/

# 4. customer-twin "Why Solace" pages live?
for co in gatan metalmark simplemdg maxsalesgroup; do
  test -f ~/projects/$co/solace/wiki/why-solace.md && echo "$co: yes" || echo "$co: missing"
done

# 5. Firestore: companies/if entry exists with promoted_via field?
gcloud firestore documents get companies/if --project=solace-461818 | grep promoted_via
```

---

## See also

- `~/projects/if/canon/NORTHSTAR.md` — mission + falsifiability
- `~/projects/if/canon/proposal-for-if.md` — call for falsifiers (NOT a sales pitch)
- `~/projects/solace-hub/canon/standards/informational-field-ai-framing.md` — Solace's AGI definition
- `~/projects/solace-hub/canon/standards/no-academic-deference.md` (if exists) — honesty discipline
- `~/projects/phucnet/` — long-form publication channel
