# Paper trader hardening spec

Repo: `ijpatter1/autoresearch`, primarily `btc-paper-trader/`
Audience: coding agent executing the work, reviewed by Ian
Status: ready for implementation
Date: 2026-08-06 (rev 2, fortified against a code-level audit)
Suggested branches: `hardening/reliability`, `hardening/ledger`, `hardening/monitoring`

Revision note: rev 1 was drafted with access to `logs/` and `data/` only. Rev 2
corrects three numbers, reverses one source-of-truth decision, and adds seven
findings that required reading the code. Corrections are marked `[REV2]`.

---

## 1. Why this work exists

The paper trader has run a frozen model (artifact `943751e`, training data
through 2025-12-31, deployed 2026-03-24) untouched for 136 days. That freeze
discipline held, which makes the run a clean out-of-sample record. An audit of
the raw outputs and the source on 2026-08-06 found the record is badly degraded
by operational problems, not model problems:

- Uptime was 65.4%. Of 3,228 expected hourly runs, 1,117 are missing across 171
  outages (longest 66 hours). The host is a laptop, not the intended always-on
  device: `cron.log` carries a `/Users/ipatterson/dev/...` path, macOS cron does
  not fire during sleep, and missing hours peak at 05:00-09:00 UTC and trough at
  20:00 UTC, which is the shape of overnight sleep in US Pacific.
- The P&L is contaminated by the outages. Gross return splits into +1.16%
  earned during running hours and +1.14% earned across outage gaps while
  positions sat frozen. Total costs were 1.15%. Net of costs, at-frequency
  trading made roughly zero; the reported +1.13% net is, in effect,
  hold-through-downtime P&L. The system went dark while holding a position 16
  times. Half the gross result rests on those 16 rows.
- Alerting never worked. The nightly Telegram report 404'd since day one (YAML
  does not expand `${VAR}`, so the literal placeholder `TELEGRAM_BOT_TOKEN` was
  posted as the bot token). The alert log caught real problems and delivered
  none of them. Reporting was set to `delivery_method: file` on 2026-08-06 and
  the cron jobs were removed; the system is currently stopped.
- The daily report has three bugs: the max drawdown line mirrors current
  drawdown, activity is reported as position adjustments rather than episodes,
  and monthly returns disagree with every other source.

`[REV2]` Corrections to rev 1's evidence:

- Max drawdown is -1.03% (hourly reconstruction, 2026-06-10 11:00), not -0.43%.
  Decided-only, zeroing gap rows, it is -1.47%. The -0.51% in `daily_summary.csv`
  is a daily-sampling artifact.
- `daily_summary.csv` is not a usable source of truth, and rev 1 had the
  direction of the monthly-return disagreement backwards. See WS5.
- The scikit-learn version skew (pickled 1.7.2, loaded 1.8.0) did not corrupt
  the record. The artifact carries a `reference_predictions` block and live
  predictions are bit-exact against it. The environment is unpinned, which is
  worth fixing, but the 136-day record remains valid evidence.

The purpose of this system is to produce statistically interpretable evidence
about whether the model has edge. Every fix below serves that purpose. None of
it touches the model.

## 2. Longer-term intent (context the agent should design toward)

This repo is the foundation of one lane in a three-lane systematic trading
program: (1) cross-sectional small/mid-cap equity ranking, (2) event-driven NLP
on SEC filings, (3) perpetual futures carry and trend. The autoresearch harness
pattern used here, where the agent may modify only the model recipe while the
evaluation infrastructure stays fixed, is planned to be ported to a
cross-sectional version across roughly the top 20 perps by liquidity.

The reason for that port is statistical. `[REV2]` Rev 1 sized the wait using a
pooled 24h IC of ~0.13. That estimate is dominated by a single month: monthly IC
runs -0.42, -0.03, -0.10, +0.26, -0.10, +0.16, and pooled IC excluding June 2026
is -0.019 over 1,117 hours. Net return excluding June is +0.36%, not +1.13%.
Plan against a range that brackets zero rather than a point estimate. This
strengthens the case for the cross-sectional port rather than weakening it:
breadth across assets is the only affordable way to buy evidence when the
single-asset prior is indistinguishable from no edge.

So while this spec is BTC-only, design choices should not hard-code single-asset
assumptions: prefer storage and state keyed by `(venue, symbol, timestamp)`, and
prefer code paths that iterate over a symbol list of length one.

Standing decisions that follow from the program:

- Artifact `943751e` is a permanent control. It is never retrained, replaced, or
  improved in place. Future models run beside it as challengers.
- `[REV2]` The control's identity is pinned by content, not by re-export.
  `~/.cache/autotrader/` has been deleted, taking `last_train_data.parquet` and
  the eval counter with it, so the deterministic re-export path from commit
  e436a81 can no longer reproduce the artifact byte-for-byte. Record
  `data_hash = 4e8b39deb7a9`, the stored `reference_predictions` block, and the
  sklearn version as the control's identity, and verify against those.
- Evidence quality beats returns. Auditability beats cleverness. Reported
  numbers must be reproducible from raw files by an independent script.
- Ledger history is append-only. Restatements happen through migration scripts
  that preserve the originals.
- Boring, reviewable solutions preferred. Where a policy question arises that
  this spec does not answer, leave a `TODO(ian):` comment and pick the
  conservative default rather than inventing policy.

## 3. Invariants (do not violate in any PR)

1. No changes to `assets/btc_hourly/train.py`, the frozen artifact, or
   evaluation semantics in `core/`.
2. No live-capital code paths.
3. No destructive edits to existing logs, CSVs, or parquet files. Migrations
   write new files and leave originals in place.
4. Existing parity tests (feature, inference, portfolio math) must pass under
   the pinned environment before merge.
5. Diffs stay reviewable: one workstream per PR, tests alongside code.
6. `[REV2]` Any code path that writes OHLCV records the source venue. Never mix
   venues in one column without a discriminator. See WS8.

## 4. Audit evidence (use as fixtures)

| Finding | Value |
|---|---|
| Logged hours / expected | 2,111 / 3,228 (65.4%) |
| Outages | 171; longest 66h from 2026-07-10 20:00; others 53h (2026-05-22), 51h (2026-04-04), 38h (2026-04-26), 33h (2026-06-20) |
| Outages beginning while positioned | 16 |
| Gross P&L, running hours vs gap rows | +1.16% vs +1.14% (gap rows = 49.5% of gross) |
| Costs | fees 0.9229%, funding 0.2296%; 50.2% of gross P&L |
| Max drawdown `[REV2]` | -1.03% hourly combined; -1.47% decided-only; -0.51% is the daily-file artifact |
| Episodes | 11 (all long), 4 profitable; positioned 9.2% of logged hours; avg size on 0.24, max 0.64 |
| Largest single P&L row | +0.65% booked on resume after a 19h outage on 2026-05-04 |
| IC (24h) | pooled +0.090 Pearson / +0.108 Spearman on `pred_24_raw`; monthly -0.42 to +0.26; ex-June -0.019 |
| Monthly net `[REV2]` | Mar +0.00, Apr -0.00, May +0.17, Jun +0.78, Jul +0.18 (hourly reconstruction) |
| BTC over same window | -7.58% |
| Shorts taken | 0 of 113 trades |
| Archiver coverage | orderbook 65.5%, open interest 65.2%; not backfillable |
| OHLCV coverage since deploy `[REV2]` | 3,238 / 3,238 (100%) — `backfill_recent_gap` already works |
| Volume splice `[REV2]` | at 2026-03-01 00:00, median hourly volume 813 -> 0.14 (5,863x), 19 zero-volume hours |

The dated outages above are the replay test fixtures for WS1 and WS2.

## 5. Workstreams

### WS1: hosting and scheduling reliability (P0)

`[REV2]` Scope correction: OHLCV backfill is already implemented and working.
`btc-paper-trader/data/btcusdt_1h.parquet` has 3,238 of 3,238 expected rows
since deployment with zero gaps, because `backfill_recent_gap` refetches missed
candles on the next successful run. The hole is in the decision record, not the
price data. Do not rebuild candle backfill; extend it.

Tasks:

- Target an always-on Linux host (Raspberry Pi per the original design, or a
  small VPS; decision D4). Replace cron with a systemd service plus timer.
  `Restart=on-failure`, `OnCalendar=*-*-* *:05:00`, missed-run catch-up via
  `Persistent=true`.
- `[REV2]` `scripts/install_services.sh` already writes a systemd unit and calls
  `systemctl`, so it never ran on the macOS host. The liquidation websocket
  aggregator it installs has therefore never run: there is no
  `data/liquidations_1h.parquet` despite `config.yaml` declaring the path, and
  no log line mentions it in 136 days. Either wire `src/liquidations.py` in on
  the new host or delete it and the config key. Do not leave a config path
  pointing at a file nothing produces.
- On startup, detect the last processed hour from state, fetch OHLCV and funding
  for every missed hour, and process the gap hour by hour instead of booking one
  lumped multi-hour return. Backfillable data (candles, funding) is
  reconstructed; non-backfillable capture (order book, open interest) records an
  explicit gap marker rather than silently skipping rows.
- Idempotency: re-running an already-processed hour must not append duplicate
  rows or double-charge fees. Key all appends by timestamp.
- `[REV2]` The running crontab did not match the checked-in installer: it
  carried a `PYTHONWARNINGS` line that appears nowhere in `install_services.sh`.
  The deployment must be reproducible from the repo. A backup of the removed
  crontab is at `crontab.backup-2026-08-06.txt`.
- Update `scripts/` install tooling and the README to match the new deployment.

Acceptance:

- [ ] Replay with a simulated 12h outage injected mid-episode produces an
      hour-by-hour ledger identical to an uninterrupted replay, except rows carry
      the gap flag from WS2.
- [ ] Running the pipeline twice for the same hour changes nothing on the second
      run.
- [ ] Replay of the 2026-07-10 66h fixture books per-hour P&L, not one 66h lump.
- [ ] `[REV2]` The deployed unit file is byte-identical to the one in `scripts/`.

### WS2: gap accounting and the decided/frozen split (P0)

Tasks:

- Add a per-row flag distinguishing decided hours (a live run set or confirmed
  the position that hour) from frozen hours (position carried because the
  pipeline was down; includes reconstructed backfill hours whose position was
  inherited).
- Ledger and daily summary gain parallel series: decided P&L, frozen P&L,
  combined. Sharpe and drawdown computed and reported on decided-only and
  combined.
- One-off migration script restating the existing history (2026-03-24 through
  present) with the split, using the known 171 gaps. Writes
  `daily_summary_restated.csv` and a restated report; originals untouched
  (default per D5).
- `[REV2]` Version the CSV schema. `predictions.csv` and `trades.csv` have bare
  headers and no version marker; adding a gap-flag column silently breaks any
  reader that assumes positional columns, including `replay.py`. Add a
  `schema_version` column or a sidecar manifest before adding fields.
- Encode the resume policy: hold the inherited position and tag its P&L as
  frozen; do not flatten on resume (default per D1, configurable
  `flatten_on_resume: false`).

Acceptance:

- [ ] Restated history reproduces combined net +1.13% and attributes
      approximately +1.14% gross to frozen rows, matching the audit.
- [ ] `[REV2]` Restated max drawdown reproduces -1.03% combined and -1.47%
      decided-only.
- [ ] Decided-only and combined Sharpe both appear in the daily report.
- [ ] Unit tests cover: gap begins while flat, gap begins while positioned,
      back-to-back gaps, gap spanning a position change on resume.

### WS3: monitoring by absence (P0)

`[REV2]` Retitled and rescoped. Ian's requirement is that the system run
silently: no routine output to read, no digest, no nightly report to open. That
is compatible with a dead-man's switch and incompatible with the rest of rev 1's
WS3. A healthy system should produce nothing at all; the only notification that
ever fires is the one that says it stopped.

Tasks:

- Dead-man's switch: ping an external heartbeat service (healthchecks.io or
  equivalent, decision D3) on every successful run, so silence itself pages.
  This is the fix for the failure mode actually observed: absence, not errors.
- Startup config validation applies to the heartbeat URL only. If the heartbeat
  target is a placeholder or fails a send test, refuse to start. Monitoring that
  cannot deliver is a fatal misconfiguration.
- Reporting stays at `delivery_method: file`. Do not re-enable a push channel
  for routine reports. The report file is for on-demand reading, not delivery.
- `[REV2]` Fix the unreachable health check. `main.py` returns 1 at line 147
  when the candle fetch fails, but `run_health_checks` is not reached until line
  283, so the data-staleness check can never fire during the outage it exists to
  detect. All 10 fetch-failure aborts produced zero alert entries. Move the
  health checks ahead of the early returns, or run them in a `finally` block.
- Alert dedup for the alert file: identical warnings write once with a count,
  not 1,654 repeated lines.
- Log rotation for `cron.log` and `system.log` (7 MB and 6 MB, unrotated).
- `[REV2]` Repoint or retire the disk check. It calls `shutil.disk_usage("/")`,
  which on the macOS host reported the whole APFS container at 90-97% (the
  owner's `~/dev`, caches, and Time Machine snapshots) and had nothing to do
  with this project, which writes ~99 KB/day. On a dedicated Pi the check
  becomes meaningful; keep it, but alert on rate of change as well as level.

Acceptance:

- [ ] Killing the service produces an external notification within 90 minutes
      with no code running on the host.
- [ ] A placeholder heartbeat URL causes startup failure with a clear message.
- [ ] `[REV2]` A simulated total data-source outage produces an alert-log entry.
- [ ] `[REV2]` A full week of healthy operation produces zero notifications.
- [ ] Seven days of simulated identical warnings produce one alert-file entry
      with a count.

### WS4: environment integrity (P0)

Tasks:

- Pin the inference environment exactly (`uv.lock` exists; the deployed venv
  drifted anyway). Record library versions in artifact metadata at export time;
  at startup, compare runtime versions against artifact metadata and refuse to
  trade on mismatch (log-only mode permitted for diagnosis).
- Run the existing parity tests (feature, inference, portfolio math) under the
  pinned environment in CI or a make target; store the parity output hash next
  to the artifact.
- `[REV2]` Verify against `reference_predictions`, which the artifact already
  carries, on every startup. This is the check that proved the 1.7.2 -> 1.8.0
  skew had not corrupted anything, and it is cheaper and stricter than a version
  comparison. A version mismatch with matching reference predictions is a
  warning; mismatched reference predictions is fatal regardless of versions.

Acceptance:

- [ ] Version mismatch is an explicit startup decision, not a warning buried in
      a log.
- [ ] `[REV2]` Reference-prediction mismatch is a hard startup failure.
- [ ] Parity tests pass under the pinned environment against artifact `943751e`.

### WS5: reporting correctness (P1)

`[REV2]` Source-of-truth reversal. Rev 1 said to compound monthly returns from
`daily_summary.csv`. That file is unusable: `_maybe_log_daily_summary` writes on
the first run of each new date, so `compute_daily_summary` only ever sees one
hour. The evidence is `hours_flat == 1` on 120 of 130 rows, `daily_return` exactly
0.0 on 117 of 130, `n_trades_today` never above 1, and 21 distinct portfolio
values across 130 days. Compounding it yields May +0.53 / Jun +0.49 / Jul -0.33;
the hourly reconstruction yields May +0.17 / Jun +0.78 / Jul +0.18 and reproduces
`portfolio_state.json` to 13 decimal places. The report's Jun +0.9 / Jul +0.2 was
closer to correct than the file rev 1 proposed to trust.

The source of truth is an hourly ledger rebuilt from `predictions.csv`.
`daily_summary.csv` is regenerated from that ledger, not the reverse.

Tasks:

- `[REV2]` Fix `_maybe_log_daily_summary` to summarise the completed day rather
  than the first hour of the new one, and regenerate the historical file from
  the hourly ledger as part of the WS2 migration.
- Fix the max drawdown line (report running max, keep current drawdown as its
  own line).
- Fix monthly returns to compound from the hourly ledger.
- Report activity as both position adjustments and episodes (contiguous
  nonzero-position runs), with win rate defined per episode and per positioned
  hour, labeled as such.
- Add to the daily report: uptime (24h / 7d / inception), gap count, decided vs
  frozen P&L split, artifact hash and age, and 24h-horizon IC to date with a
  rough CI.
- `[REV2]` Fix the parity verdict in `replay.py:285`. It compares
  `seg_parity["n_trades"]`, which counts position adjustments (278), against the
  backtester's 17 direction changes, and prints MISMATCH on a run that actually
  matched. `janfeb_analysis.txt` records "Direction changes (backtester-equiv):
  17" alongside "Position adjustments: 278". Compare like with like.
- `[REV2]` Relabel the replay segments. `replay.py` calls Jan-Feb 2026 the
  "infrastructure parity check" and March onward the "genuine out-of-sample
  validation". The artifact's `train_data_end` is 2025-12-31, so Jan-Feb is
  forward data the model never saw and is the strongest out-of-sample evidence
  in the record (+4.30% while BTC fell 23.7%, IC +0.215). March has 2 positioned
  hours out of 721 and carries no information. The labels invert their value.
- Add `scripts/verify_report.py`: an independent script that recomputes every
  number in the daily report from the raw CSVs and exits nonzero on mismatch.
  Wire it into tests.

Acceptance:

- [ ] `verify_report.py` passes against a freshly generated report and fails
      when any report formula is deliberately broken in a test.
- [ ] `[REV2]` Regenerated `daily_summary.csv` reproduces the hourly ledger's
      monthly returns within 1 bp.
- [ ] `[REV2]` A replay whose direction changes match the backtester prints
      MATCH.

### WS6: champion/challenger policy (P1)

Tasks:

- Codify `943751e` as the permanent control in config. The research agent's
  future exports deploy as challengers: separate state, ledger, and report
  files, same pipeline, run in parallel.
- Repoint the 30-day staleness alarm at challengers only; the control is exempt
  by design, which resolves the current contradiction of a frozen model alarmed
  for being frozen.
- `[REV2]` Record the control's realised behaviour alongside its metadata, so
  challengers are compared against what it did rather than what it was assumed
  to do. Specifically: it is long-only in practice. It took 102 long entries and
  zero shorts in 136 days, not because of a directional filter but because the
  post-processing chain shrinks predictions about 5x, so a short needs a raw
  prediction below roughly -1.05 sigma to clear the -0.20 entry threshold, which
  never happened. Its backtest Sharpe was earned on a symmetric strategy. Any
  challenger comparison that assumes symmetry is comparing different things.
- Document (not implement) promotion criteria: a challenger becomes a
  live-capital candidate only by pre-registered gates on its own clean record.

Acceptance:

- [ ] Two artifacts run side by side in replay without state collision,
      producing two ledgers and one combined report section.
- [ ] No staleness alerts fire for the control.

### WS7: archiver hardening (P2, may trail as a follow-up PR)

Tasks:

- Move the order book / open interest / funding capture onto the same hardened
  scheduler so it stops inheriting pipeline gaps. Confirmed at 65.5% and 65.2%
  coverage since deployment, matching the pipeline's 65.4%. This data cannot be
  backfilled, so every hour lost is lost permanently.
- Atomic parquet appends, schema versioning, and `(venue, symbol, timestamp)`
  keys even while only BTC/Binance US flows. Note that `save_parquet` is already
  atomic (`to_parquet` to a temp path then `os.replace`, `src/data.py:233-244`);
  extend that pattern to the archiver rather than reinventing it.
- Stretch: add a second venue behind the same interface. Liquidations are
  already half-built (`src/liquidations.py`, 7,950 bytes, never run) — see WS1.

Acceptance:

- [ ] A hard kill during a write leaves no corrupt or partial parquet.
- [ ] Adding a second symbol is a config change, not a code change.

### WS8: OHLCV provenance (P0) `[REV2]` new

The historical series silently changed venue mid-record. Through 2026-02-28 it
is Binance.com global data seeded from `data.binance.vision`; from 2026-03-01 it
is Binance.US live API data. Close is continuous across the boundary (66,973 ->
66,793) but median hourly volume drops from 813 to 0.14, a factor of 5,863, with
19 zero-volume hours since. Nothing in the file records this.

Impact is currently limited and should be stated precisely rather than
overstated. Every volume feature in `train.py` is a ratio — `vol_weight =
volume / rolling_mean`, `volume_ratio = vol_24 / vol_168`, `vol_momentum =
vol_24 / vol_72_lagged - 1` — so a uniform scale change cancels. Only the first
168 hours after the splice, where rolling windows straddle it, are affected.
Live predictions from roughly 2026-03-08 onward are unaffected.

The risk is forward-looking: WS1 and WS7 both add backfill paths, and refetching
from Binance.US writes thin volume into what reads as a continuous history.

Tasks:

- Add a `venue` column to the OHLCV parquet and backfill it for existing rows
  (`binance_com` through 2026-02-28, `binance_us` after).
- Refuse to append a row whose venue differs from the file's declared primary
  venue without an explicit config override.
- Emit a provenance section in the daily report: venue, row count, and first/last
  timestamp per venue.
- `TODO(ian):` decide whether to re-source 2026-03-01 onward from a
  Binance.com-equivalent feed for continuity, or to treat 2026-03-01 as a venue
  boundary and never train across it. Conservative default: mark the boundary,
  do not re-source, and exclude cross-boundary windows from any future training
  set.

Non-task, recorded to prevent a plausible mistake: do not substitute
`btc-paper-trader/data/btcusdt_1h.parquet` for the deleted
`~/.cache/autotrader/btcusdt_1h.parquet` when re-establishing the research data.
It is complete and tempting (75,349 rows, 2018 to present, zero gaps) but would
splice Binance.US volume into training. `prepare.py:148` re-downloads Binance.com
monthly klines; let it.

Acceptance:

- [ ] Every OHLCV row has a venue.
- [ ] A backfill that would write a foreign venue fails loudly.

### WS9: repo detach and platform restructure (P0, lands last)

Sequencing note: this was drafted as "WS0, lands before everything" with its own PR 0.
It is deliberately resequenced to last. The public repo should be hardened before the
platform moves off it, so that what stays public is the finished artifact rather than a
half-fixed one. Renumbered WS0 -> WS9 so the number matches the order.

Premise correction: the draft justified the move by "the current repo is a public fork,
and GitHub does not allow changing a fork's visibility." That constraint does not apply
here. `gh repo view` reports `isFork: false` and `parent: null` for
`ijpatter1/autoresearch`. It is a standalone public repo that merely has an `upstream`
remote pointed at `karpathy/autoresearch`; a configured remote does not create a fork
relationship. This repo's visibility can be flipped in Settings at any time. The
migration is therefore a choice, not a forced move, and the reason to do it is the one
Ian gave: keep the hardened public repo as a public artifact and start the private
platform beside it.

The lineage is real and survives either path. The root commit of this history is
`b11d6f2 Andrej Karpathy, 2026-03-06, "initial commit"`, so the descent is in the commit
graph itself. What is absent is GitHub's fork badge, which was never there. A mirror
push preserves the lineage; nothing needs to be done to protect it.

Tasks:

- Mirror the full history into a new private repo (`git clone --mirror` then push): all
  605 commits, all 31 experiment branches, all tags. Do not squash; the experiment
  branches are part of the research record.
- Run a secrets scan (gitleaks or equivalent) over the full history. The history has been
  public since March, so anything found is treated as already exposed: rotate it, do not
  just delete it. A bounded pre-scan on 2026-08-06 found nothing: no credential-shaped
  file (`.pem`, `.key`, `.netrc`, `id_rsa`, `.env`, anything named token/secret/credential)
  appears anywhere in `git log --all --name-only`, and no AWS, GitHub, Slack, OpenAI,
  Telegram, or PEM key pattern matched across the most recent 400 of 605 commits. That is
  evidence of absence, not proof: it covered two thirds of the history with fixed
  patterns rather than entropy analysis. Run the real scanner, but nothing is gating on
  it.
- Keep the public repo public and unarchived through PR 1 to PR 4. Archive it only at
  this workstream, after the hardening has landed in it.
- Restructure to the platform layout with `git mv`, imports updated and nothing else:
  `core/` (registry, runner, holdout rotation, report verification), `data/` (skeleton
  directories for exchanges, equities, edgar, each with a README stating its intended
  contract), `lanes/perps_btc/` (the current `assets/btc_hourly/` plus
  `btc-paper-trader/`), `eval/` (skeleton). No evaluator interface extraction in this
  workstream; per the no-abstraction-until-two-implementations rule, that happens during
  the lane 1 build.
- Note for the restructure: `core/` is currently unreadable to a sandboxed agent because
  `.claude/settings.local.json` carries a `Read(**/core/**)` deny rule, and the same
  pattern blocks `pandas.core`, so any agent running pandas in-sandbox fails on import.
  Fix the rule while the paths are moving rather than porting the problem forward.
- Extend the agent write-scope contract platform-wide: the research agent may modify
  `lanes/*/train.py` and nothing else. Update `program.md` and `.claude/` config for the
  new paths, and add a guard test that fails if any experiment branch touches files
  outside that scope.
- Define the platform score record and commit its schema: experiment id, lane, artifact
  hash, trial count, score components, data as-of range, evaluator version. Write a
  migration script mapping the existing `results.tsv` and `experiment-log.md` rows into
  it. Originals preserved per invariant 3. Per-lane `results.tsv` files remain the working
  format; a roll-up script produces the platform view, including the cross-lane trial
  count used for deflation accounting.
- Cutover for the live control: because this lands last, the trader is already on the
  always-on host from WS1 and is running from a checkout of the old repo. This workstream
  re-points that checkout at the new remote and moves it to the new paths. State files
  (`portfolio_state.json`, ledgers, parquet archives) move byte-identical. The service
  unit paths change, so this costs a service restart of seconds, not an outage. WS2 gap
  tagging is already merged by this point, so the restart window is tagged correctly as
  it happens and needs no retroactive fix.

Acceptance:

- [ ] New private repo commit count and branch list match the source exactly (605 commits,
      31 branches); artifact `943751e` hash verified identical post-move.
- [ ] Secrets scan is clean, or every finding is documented as rotated.
- [ ] Public repo is archived only after PR 1 to PR 4 have landed in it, with no commits
      after the detach date.
- [ ] Full test suite passes after the restructure with only import-path changes in the
      diff.
- [ ] Replay of the 2026-05-04 fixture window produces byte-identical ledger output before
      and after the restructure.
- [ ] Guard test rejects a synthetic experiment branch that edits a file outside
      `lanes/*/train.py`.
- [ ] Score-record schema committed; migration script maps all existing experiment rows;
      originals untouched.
- [ ] The live control's restart window appears in the ledger as a tagged frozen gap.

## 6. PR sequencing

1. PR 1, reliability substrate: WS1 + WS4 + WS8. Nothing else matters while the
   clock can silently stop, the environment can drift, or the history can change
   venue without a record.
2. PR 2, accounting truth: WS2 + WS5, including the historical restatement
   migration.
3. PR 3, monitoring and policy: WS3 + WS6.
4. PR 4, archiver: WS7.
5. PR 5, platform migration: WS9. Lands last, and is the commit that moves
   everything to the new private repo.

`[REV2]` PR 1 through PR 4 land in the public repo `ijpatter1/autoresearch`. That
is the point of the ordering: what stays public is the hardened system, not a
half-fixed one. Only PR 5 detaches.

Each PR description should link this spec and check off its acceptance boxes.

## 7. Open decisions (defaults chosen; flag in PR if changed)

- D1: resume policy after an outage. Default: hold the inherited position and
  tag P&L as frozen. Alternative (flatten on resume) changes the strategy and is
  off by default.
- D2: outage definition for reporting. Default: any missed hour counts; gaps of
  6+ hours additionally listed individually in the daily report.
- D3: heartbeat provider. RESOLVED 2026-08-06: healthchecks.io, check created,
  ping URL held by Ian. The URL is a credential — anyone holding it can forge
  heartbeats and keep a dead trader looking alive, which defeats the switch — so
  it is never committed. Config stores the env var NAME
  (`heartbeat_url_env: HEARTBEAT_PING_URL`); the value reaches the process from
  an `EnvironmentFile` on the Pi, outside the repo, mode 600. Startup fails hard
  if it is unset, empty, or still a placeholder. This is the Telegram failure
  mode inverted: that one used an unexpanded `${VAR}` as a live value and 404'd
  silently for 136 days because nothing validated it.
- D4: host. RESOLVED 2026-08-06: the Raspberry Pi already owned (`ian-pi.local`,
  192.168.86.91, login account `ijpatter1` — not `ian`). Not a laptop, not a VPS.
  Per the shared-host rule below, the trader runs under its OWN service user, not
  `ijpatter1`; the login account is recorded only for access. SHARED, not
  dedicated: this Pi is also
  the tag-writer for the Tally project (`~/dev/tally`) — a PN532 NFC reader on its
  UART, programming NTAG213 tags via `nfcpy`. The two coexist: the trader is a
  systemd timer plus a Python venv, the tag-writer is on-demand and idle most of
  the time. Constraints that follow, for whoever deploys WS1:
  - Do not re-image or factory-reset the Pi. Tally's UART reclaim
    (`dtoverlay=disable-bt`, `enable_uart=1`, serial console disabled in
    `/boot/firmware/config.txt`) and its pinned `nfcpy` / `libusb1<3.1` install
    are hand-configured and would have to be rebuilt.
  - Install the trader under its own service user and venv; do not install trader
    Python deps system-wide, to avoid colliding with Tally's `nfcpy` stack.
  - The trader must not touch the serial UART or Bluetooth config the tag-writer
    depends on.
  - Watch shared-host resource contention on a Pi (CPU, memory, SD wear). The
    hourly trader run is light, but the WS3 disk check should stay on so a full
    card is caught early — it is now a genuine shared-tenant risk, not the macOS
    false alarm it was during the laptop run.
- D5: restatement presentation. Default: restated files sit beside originals;
  the daily report switches to restated series and says so in a footnote.
- D6 `[REV2]`: volume splice handling. Default: mark the venue boundary, do not
  re-source, exclude cross-boundary windows from future training.
- D7 `[REV2]`: notification budget. Default: heartbeat failure only. A healthy
  week produces zero notifications.
- D8 `[REV2]`: name and handling of the existing repo. `TODO(ian):` the new
  private repo's name is unresolved and is Ian's call. Default for everything
  else: `ijpatter1/autoresearch` stays public and is archived after PR 4; the new
  platform repo is private. Note this no longer blocks the first command — WS9
  lands last, so the name is needed at PR 5, not PR 1. The draft numbered these
  D6 and D7; renumbered to D8 and D9 because rev 2 already used those.
- D9 `[REV2]`: registry format. Default: per-lane `results.tsv` plus a platform
  roll-up now, a unified store only when a second lane exists.

## 8. Definition of done

The system runs unattended on an always-on host; a dead process pages within 90
minutes through an external channel while a healthy one says nothing at all;
every hour since deployment is either a decided row, a reconstructed frozen row,
or an explicitly marked capture gap; every OHLCV row names its venue; the daily
report is reproducible by `verify_report.py`; parity tests and the artifact's
own reference predictions both pass under a pinned environment; and the frozen
control keeps accruing clean evidence while challengers run beside it.

When those hold, the wall clock starts counting again and the codebase is one
config key away from the cross-sectional version it is meant to seed.

`[REV2]` One further clause, satisfied by WS9 and only after the above: the
platform's entire history lives in a private repo, and the public repo is
archived in its hardened state as the artifact of this work.
