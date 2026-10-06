# Documentation

| Directory | Contents |
|---|---|
| `architecture/` | What the code does. `current.md` is authoritative. |
| `experiments/` | Validation reports and how to run a recorded experiment. |
| `decisions/` | The collaboration contract and the developer handoff. |
| `research/` | V3 notes and the original, more detailed V3 roadmap. |

## Read in this order

1. [`../README.md`](../README.md) — what this repository is and which part is production.
2. [`architecture/current.md`](architecture/current.md) — the implemented architecture.
3. [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — the rules before your first change.
4. [`decisions/collaboration-contract.md`](decisions/collaboration-contract.md) — shared contracts and ownership.
5. [`architecture/roadmap.md`](architecture/roadmap.md) — what comes next.

## Which documents are current

- `architecture/current.md` — current. Updated when behaviour changes.
- `architecture/roadmap.md` — current. Updated when a phase completes.
- `architecture/v2.3-history.md` — **historical**. Pre-reorganisation; paths are stale.
- `research/v3-architecture-notes.md` — discussion notes. Findings still valid, paths translated in the header.
- `research/v3-roadmap-original.md` — **superseded** by `architecture/roadmap.md`; kept for the detail.
- `decisions/developer-handoff.md` — background reading. Paths predate the reorganisation; the header says so.
- `experiments/validation-report*.md` — point-in-time records. Do not edit past reports.

## A note on stale paths

Documents written before the reorganisation quote paths that have since moved.
Rather than rewriting history, those documents carry a header explaining the
translation. If you find a stale path that is not covered by a header, that is a
doc bug: fix it or add the mapping, but do not silently leave a reader to discover
it.