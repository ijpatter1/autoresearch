"""Champion/challenger roster (hardening spec WS6).

Artifact `943751e` is the permanent control: never retrained, replaced, or
improved in place. The research agent's future exports run beside it as
challengers with their own state, ledger, and report files — same pipeline, in
parallel — so a challenger can never collide with the control's records.

This module is the single place that resolves the roster. Config declares it
under `models:`; a config that predates WS6 (only the legacy `model:` /
`logging:` keys) synthesises a one-entry control roster, so PR1/PR2 configs and
tests keep working unchanged.

The control's *realised* behaviour is recorded alongside its metadata (WS6):
over the 136-day audited record it was long-only in practice — zero shorts —
not by a directional filter but because the post-processing chain shrinks raw
predictions ~5x, so a short needed a raw prediction below roughly -1.05 sigma to
clear the -0.20 entry threshold, which never happened. Challengers are compared
against what the control did, not against a symmetric backtest it never traded.
"""

from dataclasses import dataclass, field

CONTROL = "control"
CHALLENGER = "challenger"


@dataclass(frozen=True)
class ModelSpec:
    """One model in the roster and the files it owns."""

    id: str
    role: str
    artifact_path: str
    state_path: str
    prediction_log: str
    trade_log: str
    daily_summary_log: str
    report_path: str
    realised: dict = field(default_factory=dict)

    @property
    def is_control(self) -> bool:
        return self.role == CONTROL


def _short_id(spec_id: str) -> str:
    """A filesystem-safe token from a roster id, for deriving default paths."""
    return spec_id.split("-")[-1] if "-" in spec_id else spec_id


def _spec_from_entry(entry: dict) -> ModelSpec:
    """Build a ModelSpec from a `models:` entry, deriving unset paths from its id.

    A challenger that names only its id and artifact gets its ledger/state/report
    under a per-challenger subtree keyed by id, so it never shares files with the
    control or another challenger.
    """
    spec_id = entry["id"]
    role = entry.get("role", CHALLENGER)
    sub = _short_id(spec_id)
    default_dir = "" if role == CONTROL else f"challengers/{sub}/"
    return ModelSpec(
        id=spec_id,
        role=role,
        artifact_path=entry["artifact_path"],
        state_path=entry.get("state_path", f"data/{default_dir}portfolio_state.json"),
        prediction_log=entry.get("prediction_log", f"logs/{default_dir}predictions.csv"),
        trade_log=entry.get("trade_log", f"logs/{default_dir}trades.csv"),
        daily_summary_log=entry.get("daily_summary_log", f"logs/{default_dir}daily_summary.csv"),
        report_path=entry.get("report_path", f"logs/{default_dir}daily_report.txt"),
        realised=entry.get("realised", {}) or {},
    )


def _legacy_control(config: dict) -> ModelSpec:
    """Synthesise a single control spec from pre-WS6 config keys."""
    model_cfg = config.get("model", {})
    log_cfg = config.get("logging", {})
    report_cfg = config.get("reporting", {})
    data_cfg = config.get("data", {})
    import os

    state_path = os.path.join(
        os.path.dirname(data_cfg.get("parquet_path", "data/btcusdt_1h.parquet")),
        "portfolio_state.json",
    )
    return ModelSpec(
        id="control",
        role=CONTROL,
        artifact_path=model_cfg.get("artifact_path", ""),
        state_path=state_path,
        prediction_log=log_cfg.get("prediction_log", "logs/predictions.csv"),
        trade_log=log_cfg.get("trade_log", "logs/trades.csv"),
        daily_summary_log=log_cfg.get("daily_summary_log", "logs/daily_summary.csv"),
        report_path=report_cfg.get("daily_report_path", "logs/daily_report.txt"),
        realised={},
    )


def load_model_specs(config: dict) -> list[ModelSpec]:
    """Resolve the model roster. Exactly one control is required.

    Uses the `models:` block when present, else synthesises a one-entry control
    roster from the legacy keys.
    """
    entries = config.get("models")
    if not entries:
        return [_legacy_control(config)]

    specs = [_spec_from_entry(e) for e in entries]
    n_control = sum(s.is_control for s in specs)
    if n_control != 1:
        raise ValueError(
            f"roster must declare exactly one control model, found {n_control}"
        )
    return specs


def control_of(specs: list[ModelSpec]) -> ModelSpec:
    """The single control model in the roster."""
    for s in specs:
        if s.is_control:
            return s
    raise ValueError("roster has no control model")


def challengers_of(specs: list[ModelSpec]) -> list[ModelSpec]:
    """The challenger models, in roster order."""
    return [s for s in specs if not s.is_control]


def is_control_artifact(config: dict, artifact_path: str) -> bool:
    """Whether a running artifact is the permanent control.

    Drives the WS6 staleness exemption: the control is frozen by design and must
    not be alarmed for being frozen; an unrecognised or challenger artifact is
    not exempt.
    """
    specs = load_model_specs(config)
    ctrl = control_of(specs)
    return artifact_path == ctrl.artifact_path
