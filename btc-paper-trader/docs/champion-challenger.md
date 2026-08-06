# Champion / challenger policy

Hardening spec WS6. This documents how models coexist in the paper trader and
how a challenger could eventually become a live-capital candidate. The promotion
gates are documented here, not implemented: nothing in this repo moves capital.

## The control is permanent

Artifact `943751e` (trained through 2025-12-31, exported 2026-03-24) is the
control. It is never retrained, replaced, or improved in place. Its identity is
pinned by content, not by re-export: `data_hash = 4e8b39deb7a9`, the stored
`reference_predictions` block, and `sklearn_version = 1.7.2`. The environment
integrity check (WS4) verifies the reference predictions on every startup, so a
swapped artifact or a drifted environment fails loudly rather than trading as if
it were the control.

The control's purpose is to keep accruing one clean out-of-sample record. The
30-day staleness alarm is repointed off it: a frozen control alarmed for being
frozen is the contradiction WS6 resolves. Challengers are not exempt.

## Challengers run beside it

The research agent's future exports deploy as challengers. Each is a roster
entry under `models:` in `config.yaml` with `role: challenger`; each gets its
own state, ledger, trade log, and report file, defaulting to a per-challenger
subtree (`logs/challengers/<id>/…`, `data/challengers/<id>/…`). They run through
the same pipeline, so a challenger can never write to the control's files.

`scripts/replay.py --side-by-side` replays the whole roster over one window into
separate ledgers and writes a single combined section comparing them. Because
each model books into its own subdirectory, adding a challenger is a config
change, not a code change.

## The control's realised behaviour

Recorded in the roster's `realised:` block so challengers are compared against
what the control did, not against a symmetric backtest it never traded:

- Long-only in practice. It took long entries and zero shorts across the 118
  trades of its 136-day live record. This is not a directional filter. The
  post-processing chain shrinks raw predictions roughly 5x, so a short needs a
  raw prediction below about -1.05 sigma to clear the -0.20 entry threshold,
  which never happened live.
- The backtest Sharpe that justified deployment was earned on a symmetric
  strategy. A challenger evaluated on symmetric assumptions is being compared
  against something the control did not actually do.

A caveat the side-by-side replay makes visible: over the Jan-Feb 2026
forward-data window the control did take shorts (BTC fell about 24% over that
stretch). "Long-only in practice" describes the live record from 2026-03-24
onward, not every possible window. The combined section's shorts column is per
replay window and can be nonzero.

## Promotion criteria (documented, not implemented)

A challenger becomes a live-capital candidate only by pre-registered gates on
its own clean record, decided before the record exists, not fitted to it after.
The gates are Ian's to set. Until they are set, treat this as the conservative
default:

- A minimum out-of-sample horizon on the hardened infrastructure, measured in
  decided hours (not calendar days), so uptime gaps do not inflate the count.
- Evidence stated on decided-only P&L, not P&L that rode frozen hours. WS2 keeps
  the two separate; a promotion argument uses the decided series.
- Gates registered before the window opens, with the trial count recorded, so a
  challenger promoted after N attempts is deflated for the N-1 that failed.
- The comparison is against the control's realised behaviour, not its backtest.

`TODO(ian):` set the specific thresholds. Nothing promotes automatically.
