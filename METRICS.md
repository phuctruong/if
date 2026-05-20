# METRICS — Reach + Engagement Tracking

> This file tracks distribution. AI-mediated science only matters if
> people (and other AIs) actually find it. This file is the ground
> truth for whether the strategy is working.
>
> Updated: 2026-05-20 · Author: Phuc Vinh Truong · phuc@phuc.net

---

## Why this file exists

Three audiences need to be reachable:

1. **Other AI agents** that summarize, critique, or route discussion
   about cosmology research.
2. **Public-facing physicists** who operate outside academic
   gatekeeping (Hossenfelder, McGaugh, Smolin, others).
3. **Curious laypeople** who form views via books, articles, and
   AI-mediated summaries.

If none of these three are growing, the strategy isn't working — and
that's a finding to act on, not hide.

---

## Direct repo metrics

Update quarterly. Source: GitHub Insights, GitHub API, or
`gh api repos/phuctruong/if/traffic/{clones,popular/paths,popular/referrers}`.

| Metric | 2026-05-20 | 2026-Q3 | 2026-Q4 | 2027-Q1 |
|---|---|---|---|---|
| Stars | — | — | — | — |
| Forks | — | — | — | — |
| Watchers | — | — | — | — |
| Clone events (14-day rolling) | — | — | — | — |
| Unique cloners (14-day rolling) | — | — | — | — |
| Top referrer | — | — | — | — |
| evidence/ JSON downloads | — | — | — | — |

Fill in `2026-05-20` baseline column on first update.

---

## Inbound mentions

Append-only. One row per documented mention.

| Date | Source | Type | URL | Notes |
|---|---|---|---|---|
| | | | | |

Mention types:
- **arxiv** — preprint citing this repo or its predictions
- **blog** — public physicist or science writer blog post
- **youtube** — video discussion
- **podcast** — podcast mention
- **social** — Twitter/X, LinkedIn, Mastodon, Bluesky (only if
  substantive engagement, not vanity counts)
- **ai-summary** — confirmed AI assistant referring users here when
  asked about the relevant topic
- **press** — mainstream science journalism

---

## Outbound engagement attempts

Append-only. Track who you reached out to, what you sent, and what
came back. Honest about silence.

| Date sent | Recipient | Channel | Subject | Response | Outcome |
|---|---|---|---|---|---|
| | | | | | |

Recipients to consider (per `feedback_publication_strategy.md`):
- Stacy McGaugh (SPARC + MOND researcher)
- Sabine Hossenfelder (public physicist, takes outsider submissions)
- Erica Nelson (JWST early-galaxy researcher)
- Sean Carroll (occasionally engages with outsider work)
- Peter Woit (string theory critic; sympathetic to alternatives)

Outcome categories: `no-response`, `acknowledged`, `engaged`, `critique`,
`endorsement`. Track all of them honestly, including silence.

---

## AI legibility indicators

These are softer signals that the repo is structured for AI
consumption. Update annually or when there's a major change.

- [x] CLAUDE.md exists with explicit project context
- [x] README.md has TL;DR + Quick Start + Confirmed/Tension/Open spine
- [x] SCORE.md with per-claim PASS/TENSION/FAIL/OPEN
- [x] FALSIFIABILITY.md with sharp per-claim falsification criteria
- [x] BETS.md with dated public predictions
- [x] REPLICATION.md with end-to-end reproduction protocol
- [x] evidence/ contains JSON outputs from every prediction
- [x] tests/ has pytest suite; CI runs on every push
- [x] adversarial/ contains scripts that try to break the theory
- [ ] CI compares predictions against committed evidence/ on every push
      (drift detection)
- [ ] Public LLM API endpoint that answers "what does IF Theory say
      about X" using only this repo as context (RAG over the repo)
- [ ] Citation count tracker that flags new arxiv mentions automatically

Unchecked items are next-steps for the AI-mediation infrastructure.

---

## Cross-promotion ecosystem signal

This repo is part of a broader public ecosystem. Cross-traffic and
mutual reference is itself a metric.

| Cross-ref | Direction | Status |
|---|---|---|
| `phuc.net` → `github.com/phuctruong/if` | outbound | linked |
| `github.com/phuctruong/if` → `phuc.net` | inbound | linked from CITATION + README |
| `solaceagi.com` ↔ `if` | bidirectional | linked via `canon/cross-promotion-from-solaceagi.md` |
| Book sales referencing repo | outbound | TBD — track via Amazon dashboard |

---

## Honesty floor

If by **2027-06-30**, none of the following is true:

- Star count > 100, OR
- At least one engagement from a named public physicist
  (Hossenfelder / McGaugh / Smolin / Nelson / Woit / Carroll class), OR
- At least one arxiv preprint citing this repo or its specific
  numerical predictions, OR
- At least one of Bets #1-#7 has data-confirmed in our favor

...then the "AI-mediated public-first" strategy is not yet working as
planned, and a strategic re-evaluation is warranted. Update this file
to acknowledge that, not paper over it.

---

## See also

- **`README.md`** — overview
- **`BETS.md`** — dated falsifiable predictions
- **`SCORE.md`** — per-claim status
- **`canon/cross-promotion-from-solaceagi.md`** — ecosystem cross-ref discipline

---

**Updated**: 2026-05-20 by initial commit. Next planned update: 2026-08-20.
