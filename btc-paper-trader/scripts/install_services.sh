#!/bin/bash
# Install the BTC Paper Trader as systemd timers on the always-on Linux host
# (hardening spec WS1). Replaces the previous cron + liquidation-websocket
# installer, which only ran on macOS and never fired during sleep.
#
# What it installs:
#   - btc-paper-trader.timer         hourly inference at :05, Persistent catch-up
#   - btc-paper-trader-report.timer  daily report at 00:15 UTC
# Both drive oneshot services running as a dedicated `btctrader` user from a
# venv pinned by uv.lock. The unit files are copied VERBATIM from deploy/systemd
# and verified byte-identical, so the deployment is reproducible from the repo.
#
# The liquidation aggregator is intentionally NOT installed here; supplementary
# capture is hardened in WS7 (PR4).
#
# Usage (run as root on the Pi, from the pinned checkout path):
#   sudo bash scripts/install_services.sh          # install / update
#   sudo bash scripts/install_services.sh --check   # verify byte-identical units only
#
# Pi note (shared host): this touches only its own user, venv, unit files, and
# /etc/btc-paper-trader. It does not run raspi-config and does not modify serial,
# UART, or Bluetooth configuration that the Tally tag-writer depends on.

set -euo pipefail

INSTALL_DIR="/opt/btc-paper-trader"     # pinned: must match WorkingDirectory in the units
SERVICE_USER="btctrader"
ENV_DIR="/etc/btc-paper-trader"
ENV_FILE="${ENV_DIR}/btc-paper-trader.env"
SYSTEMD_DIR="/etc/systemd/system"
UNITS=(
    btc-paper-trader.service
    btc-paper-trader.timer
    btc-paper-trader-report.service
    btc-paper-trader-report.timer
)

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-install}"

fail() { echo "ERROR: $*" >&2; exit 1; }

# --- Verify units in the repo are byte-identical to what is (or would be) deployed ---
check_units() {
    local ok=1
    for unit in "${UNITS[@]}"; do
        local src="${SCRIPT_DIR}/deploy/systemd/${unit}"
        local dst="${SYSTEMD_DIR}/${unit}"
        [ -f "$src" ] || fail "missing repo unit: $src"
        if [ ! -f "$dst" ]; then
            echo "  $unit: not yet installed"
            ok=0
        elif cmp -s "$src" "$dst"; then
            echo "  $unit: byte-identical"
        else
            echo "  $unit: DIFFERS from repo"
            ok=0
        fi
    done
    return $((ok == 1 ? 0 : 1))
}

if [ "$MODE" = "--check" ]; then
    echo "Checking deployed units against repo..."
    check_units && echo "All units byte-identical." || fail "unit drift detected"
    exit 0
fi

# --- Preconditions ---
[ "$(id -u)" -eq 0 ] || fail "must run as root (use sudo)"
command -v systemctl >/dev/null 2>&1 || fail "systemd not found; this installer targets a Linux host"
if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
    fail "checkout is at $SCRIPT_DIR but the units pin $INSTALL_DIR.
       Clone the repo to $INSTALL_DIR (or adjust deploy/systemd/*.service) and re-run."
fi

echo "Installing BTC Paper Trader services..."
echo "  Install dir:  $INSTALL_DIR"
echo "  Service user: $SERVICE_USER"

# --- Service user (system account, no login shell) ---
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
    echo "  Created service user $SERVICE_USER"
fi

# --- Python venv pinned by uv.lock ---
if command -v uv >/dev/null 2>&1; then
    ( cd "$INSTALL_DIR" && uv sync --frozen )
    echo "  Synced venv from uv.lock (frozen)"
else
    fail "uv not found. Install uv (https://docs.astral.sh/uv/) so the venv is
       pinned by uv.lock; an unpinned pip install is what let the env drift."
fi

# --- Data/log dirs and ownership ---
mkdir -p "$INSTALL_DIR/data" "$INSTALL_DIR/logs"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# --- EnvironmentFile scaffold (mode 600, owned by the service user) ---
mkdir -p "$ENV_DIR"
if [ ! -f "$ENV_FILE" ]; then
    install -o "$SERVICE_USER" -g "$SERVICE_USER" -m 600 \
        "${SCRIPT_DIR}/deploy/btc-paper-trader.env.example" "$ENV_FILE"
    echo "  Scaffolded $ENV_FILE (mode 600) — populate HEARTBEAT_PING_URL before enabling the timer (WS3)"
fi

# --- Unit files: copy verbatim, then verify byte-identical ---
for unit in "${UNITS[@]}"; do
    install -m 644 "${SCRIPT_DIR}/deploy/systemd/${unit}" "${SYSTEMD_DIR}/${unit}"
done
echo "  Installed ${#UNITS[@]} unit files"
check_units || fail "post-install unit verification failed"

# --- Enable timers ---
systemctl daemon-reload
systemctl enable --now btc-paper-trader.timer btc-paper-trader-report.timer
echo "  Timers enabled and started"

cat <<EOF

Setup complete. Verify with:
  systemctl list-timers 'btc-paper-trader*'      # next fire times
  systemctl status btc-paper-trader.service       # last run
  journalctl -u btc-paper-trader.service -n 50     # run logs
  sudo bash scripts/install_services.sh --check    # unit files still byte-identical

The model artifact and its parity sidecar are NOT in git; copy them to
$INSTALL_DIR/artifacts/ out of band, then confirm the environment:
  cd $INSTALL_DIR && uv run scripts/verify_environment.py
EOF
