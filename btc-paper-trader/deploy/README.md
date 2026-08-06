# Deployment

The trader runs as three systemd timers on an always-on Linux host — not cron
on a laptop. This is the WS1 fix: the audited run lost 35% of its hours because
macOS cron does not fire during sleep, and the failure was invisible because
nothing paged on absence.

## Target host

The Raspberry Pi `ian-pi.local` (hardening spec D4). It is **shared** with the
Tally NFC tag-writer, so the trader:

- runs as a dedicated `btctrader` system user, never as the login account;
- installs its Python deps into its own venv, never system-wide;
- is confined by the units to its own tree and must not touch the serial/UART/
  Bluetooth config Tally depends on. Do **not** run `raspi-config` or re-image.

## Layout

The units pin `/opt/btc-paper-trader`, but this project is a **subdirectory of
the repo**, so that path cannot be the clone itself. The deployed layout (as on
the Pi since 2026-08-06) is a clone at `/opt/autoresearch` plus a symlink:

| Path | What |
|---|---|
| `/opt/autoresearch` | the repo clone |
| `/opt/btc-paper-trader` | symlink -> `/opt/autoresearch/btc-paper-trader` (the path the unit files hard-code) |
| `/opt/btc-paper-trader/.venv` | venv pinned by `uv.lock` |
| `/opt/btc-paper-trader/artifacts/` | model artifact + **per-host** parity sidecar (copied/generated out of band; not in git) |
| `/etc/btc-paper-trader/btc-paper-trader.env` | EnvironmentFile, mode 600 (holds `HEARTBEAT_PING_URL`) |
| `/etc/systemd/system/btc-paper-trader*.{service,timer}` | installed verbatim from `deploy/systemd/` |

## Install

```bash
# On the Pi, as a user with sudo:
sudo git clone <repo> /opt/autoresearch
sudo ln -s /opt/autoresearch/btc-paper-trader /opt/btc-paper-trader
# Copy the model artifact into artifacts/ (gitignored):
#   scp artifacts/model_943751e.joblib \
#       ijpatter1@ian-pi.local:/opt/btc-paper-trader/artifacts/
sudo bash /opt/btc-paper-trader/scripts/install_services.sh
```

Two symlink gotchas, both hit on the first deploy:

- Invoke the installer by **absolute path through the symlink**, as above. A
  `cd` + relative path fails its location check: sudo scrubs `$PWD` and bash
  re-derives the physical path (`/opt/autoresearch/btc-paper-trader`), which is
  not the pinned one.
- `chown -R` does **not** traverse the symlink. After the install and after
  every `git pull`, re-own the physical tree:
  `sudo chown -R btctrader:btctrader /opt/autoresearch/btc-paper-trader`.

The installer creates the service user, syncs the venv from `uv.lock`
(`uv sync --frozen` — an unpinned pip install is what let the environment
drift), copies the unit files **verbatim**, verifies they are byte-identical to
the repo, scaffolds the mode-600 EnvironmentFile, and enables the timers.

Do not copy the laptop's parity sidecar: the `pred_final` byte hash it stores
is host-specific (float64 last-bit differences across arm64 macOS vs aarch64
Linux). Generate it on the Pi with `write_parity_sidecar`; the artifact's own
reference-prediction check (WS4) is the authoritative cross-host guard.

Updates:

```bash
sudo git -C /opt/autoresearch pull --ff-only
sudo chown -R btctrader:btctrader /opt/autoresearch/btc-paper-trader
sudo bash /opt/btc-paper-trader/scripts/install_services.sh   # re-sync venv + units if they changed
```

## Verify

```bash
systemctl list-timers 'btc-paper-trader*'          # next fire times
journalctl -u btc-paper-trader.service -n 50        # last run
sudo bash scripts/install_services.sh --check        # units still byte-identical to repo
cd /opt/btc-paper-trader && uv run scripts/verify_environment.py   # env matches the artifact
```

`verify_environment.py` recomputes the artifact's reference predictions under
the installed venv and compares the parity sidecar; it exits non-zero on any
mismatch. Run it after every deploy.

## Scheduling and catch-up

- `btc-paper-trader-archiver.timer` — hourly at `:02`, `Persistent=true`.
- `btc-paper-trader.timer` — hourly at `:05`, `Persistent=true`.
- `btc-paper-trader-report.timer` — daily at `00:15`, `Persistent=true`.

`Persistent=true` replays a run missed while the host was off. On the next run
the pipeline books every missed hour **individually** from backfilled candles
(WS1), rather than one lumped multi-hour return, and is idempotent: re-running
an already-processed hour changes nothing. A catch-up longer than
`trading.max_catchup_hours` (default 168h) is clamped and logged; the earlier
gap is left for WS2 frozen-gap tagging.

## Supplementary archiver (WS7)

`btc-paper-trader-archiver.timer` captures the order book snapshot (Binance US)
and open interest + funding (Kraken Futures) at `:02` every hour, as its own
service rather than a step of the trading pipeline. In the audited record,
capture ran inside the pipeline behind the candle fetch, so its coverage
matched the pipeline's 65.4% — and unlike candles, this data cannot be
backfilled; every hour lost is lost permanently.

Archive rows are keyed `(venue, symbol, timestamp)` — timestamps in UTC,
matching the OHLCV series — with a `schema_version` and a `capture_status` of
`captured` or `gap`. An hour the archiver missed (host down, fetch failed) is
written as an explicit `gap` row by the next successful run, so coverage is
queryable from the file itself rather than inferred from absences. Writes are
atomic (temp + rename); a hard kill mid-write leaves the previous file intact.
Symbols are added in `config.yaml` under `archiver.targets` — a config change,
not a code change.

On its first run the archiver upgrades the pre-WS7 parquets in place: stamps
the key columns, converts the old host-local (America/New_York) timestamps to
UTC, marks the historical gap hours, and keeps the originals once at
`<file>.pre-ws7.bak`.

The archiver logs to journald only (`journalctl -u
btc-paper-trader-archiver.service`) and never pages: per the notification
budget (D7), the heartbeat is the only page there is. Capture losses are
visible as gap rows, not notifications.

## Monitoring (WS3)

The system is meant to run silently: a healthy week produces no notifications at
all. The only page that ever fires is the one that says it stopped.

### Dead-man's switch (heartbeat)

On every successful run the trader pings a healthchecks.io check; if the pings
stop arriving, healthchecks.io pages Ian. Absence is what the audited run failed
to catch (35% of hours simply missing, nothing watching for it), so absence is
what alerts.

The ping URL is a credential and is never committed. `config.yaml` stores only
the name of the env var that holds it (`monitoring.heartbeat.url_env:
HEARTBEAT_PING_URL`); the value lives in `/etc/btc-paper-trader/btc-paper-trader.env`
(mode 600, outside the repo). Startup fails hard if that env var is unset, empty,
or still a placeholder — so a misconfigured switch stops the trader rather than
letting a dead process look alive. Set `monitoring.heartbeat.enabled: false` to
run without it (local dev only).

Configure the healthchecks.io check's period and grace so a stopped trader pages
within 90 minutes: hourly period, ~35-minute grace. A `/fail` ping is sent on a
critical failure (exit 2) for an immediate page; a transient data-fetch miss
(exit 1) pings neither and is absorbed by the next tick.

### Alert file

`logs/alerts.log` is deduplicated by signature: a warning that fires every hour
for months is one line with a count and a first/last window, not thousands of
repeated lines (the audited log had 1,654). The structured state is in
`logs/alerts.log.state.json`; the `.log` is a rendered view. A pre-WS3 append-only
alert log is preserved once at `logs/alerts.log.pre-ws3.bak`.

### Log rotation

`system.log` rotates in-process via a `RotatingFileHandler`
(`system_log_max_bytes` / `system_log_backup_count` in `config.yaml`), so no
external `logrotate` is added — it would rename the file out from under the
handler. `alerts.log` is bounded by dedup rather than rotation. `cron.log` no
longer exists: under systemd, run output goes to journald
(`journalctl -u btc-paper-trader.service`).

### Disk check

On this shared Pi a full SD card is a real risk (Tally shares the host), so the
disk check alerts on the fill *rate* as well as the level: a card trending toward
full within `alerts.disk_fill_horizon_days` pages before it hits
`alerts.disk_pct_threshold`. The rolling free-space history lives in
`logs/disk_history.json`.
