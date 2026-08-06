# Deployment

The trader runs as two systemd timers on an always-on Linux host — not cron on
a laptop. This is the WS1 fix: the audited run lost 35% of its hours because
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

| Path | What |
|---|---|
| `/opt/btc-paper-trader` | the checkout (pinned; the unit files hard-code this path) |
| `/opt/btc-paper-trader/.venv` | venv pinned by `uv.lock` |
| `/opt/btc-paper-trader/artifacts/` | model artifact + parity sidecar (copied out of band; not in git) |
| `/etc/btc-paper-trader/btc-paper-trader.env` | EnvironmentFile, mode 600 (holds `HEARTBEAT_PING_URL` once WS3 lands) |
| `/etc/systemd/system/btc-paper-trader*.{service,timer}` | installed verbatim from `deploy/systemd/` |

## Install

```bash
# On the Pi, as a user with sudo:
sudo git clone <repo> /opt/btc-paper-trader     # must be this exact path
cd /opt/btc-paper-trader
# Copy the model artifact + parity sidecar into artifacts/ (they are gitignored):
#   scp artifacts/model_943751e.joblib artifacts/model_943751e.parity.json \
#       ijpatter1@ian-pi.local:/opt/btc-paper-trader/artifacts/
sudo bash scripts/install_services.sh
```

The installer creates the service user, syncs the venv from `uv.lock`
(`uv sync --frozen` — an unpinned pip install is what let the environment
drift), copies the unit files **verbatim**, verifies they are byte-identical to
the repo, scaffolds the mode-600 EnvironmentFile, and enables the timers.

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

- `btc-paper-trader.timer` — hourly at `:05`, `Persistent=true`.
- `btc-paper-trader-report.timer` — daily at `00:15`, `Persistent=true`.

`Persistent=true` replays a run missed while the host was off. On the next run
the pipeline books every missed hour **individually** from backfilled candles
(WS1), rather than one lumped multi-hour return, and is idempotent: re-running
an already-processed hour changes nothing. A catch-up longer than
`trading.max_catchup_hours` (default 168h) is clamped and logged; the earlier
gap is left for WS2 frozen-gap tagging.

The liquidation websocket aggregator is **not** installed here — supplementary
capture is hardened in WS7 (PR4).

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
