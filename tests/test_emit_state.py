"""Tests for scripts/emit_state.py — the STATE_CONTRACT emission.

The emission is a copy of values this repo already publishes, so these tests are
about the ways a copy goes wrong rather than about arithmetic:

  1. It emits a guess. A renamed or null key must stop the emission rather than
     produce a null that reads downstream as a state.
  2. It emits a state and its sub-states from DIFFERENT sessions. `meta.endDate`
     and `rolling[-1].date` reach the same session by two routes; if they ever
     disagreed the composite would be read against one date and its sub-states
     against another, which is a wrong state wearing a plausible date. That is
     the hardest failure to notice downstream, so it is asserted explicitly.
  3. It emits a stale file, or churns the repo with no-op commits.

NOTE: this repo has no CI test run, so these are a local guard rather than a
gate. Run them by hand after touching emit_state.py:

    python -m pytest tests/test_emit_state.py -q

Synthetic payloads stand in for the 8 MB on-disk file, so the tests are fast and
do not move with the market. Python datetime months are 1-indexed (January = 1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import emit_state  # noqa: E402

REQUIRED = {"as_of", "state", "value", "zone", "role", "horizon",
            "evidence_grade", "licence", "action_hint", "source_file"}
OPTIONAL = {"computed_at", "cadence"}
SIGNAL = "equity_defense_composite"


def _row(**over):
    d = {"date": "2026-08-25", "rolling8d": 12, "aboveSMA200": True,
         "canaryBad": 0, "canaryTotal": 2, "spy12mNeg": False,
         "belowSMA10m": False, "vixRaw": False}
    d.update(over)
    return d


def _signals(composite=0, end_date="2026-08-25", row=None, **over):
    d = {
        "meta": {"endDate": end_date, "compositeNow": composite, "vixRatioNow": 0.8484},
        "monitor": {"defenceActive": False, "cooldownRemaining": 0},
        "rolling": [_row(date="2026-08-24"), row if row is not None else _row(date=end_date)],
    }
    d.update(over)
    return d


@pytest.fixture
def store(monkeypatch):
    box = {"d": _signals()}
    monkeypatch.setattr(emit_state, "load_signals", lambda: box["d"])
    return box


# --- the state rule ---------------------------------------------------------

@pytest.mark.parametrize("composite,expected", [
    (0, "NEUTRAL"), (1, "NEUTRAL"), (2, "DEFENSIVE"), (3, "DEFENSIVE"), (5, "DEFENSIVE"),
])
def test_the_defensive_threshold_is_two_or_more(store, composite, expected):
    store["d"] = _signals(composite=composite)
    assert emit_state.build()["signals"][SIGNAL]["state"] == expected


def test_one_is_not_defensive(store):
    """The boundary in the wrong direction is the one worth pinning: a composite
    of 1 reading as DEFENSIVE would put the board into a risk-off state on a
    quiet day."""
    store["d"] = _signals(composite=1)
    s = emit_state.build()["signals"][SIGNAL]
    assert s["state"] == "NEUTRAL" and s["action_hint"] == "none"


def test_a_defensive_state_carries_a_watch_hint(store):
    store["d"] = _signals(composite=2)
    assert emit_state.build()["signals"][SIGNAL]["action_hint"] == "watch"


# --- the drift check --------------------------------------------------------

def test_a_session_mismatch_stops_the_emission(store):
    """meta.endDate and rolling[-1].date must describe the same session."""
    store["d"] = _signals(end_date="2026-08-25", row=_row(date="2026-08-22"))
    with pytest.raises(emit_state.EmitError, match="different sessions"):
        emit_state.build()


def test_matching_sessions_pass(store):
    store["d"] = _signals(end_date="2026-08-25", row=_row(date="2026-08-25"))
    assert emit_state.build()["signals"][SIGNAL]["as_of"] == "2026-08-25"


def test_it_reads_the_LAST_rolling_row_not_the_first(store):
    """A regression to rolling[0] would silently publish a state from 2005."""
    d = _signals()
    d["rolling"] = [_row(date="2005-01-03", rolling8d=999), _row(date="2026-08-25")]
    store["d"] = d
    s = emit_state.build()["signals"][SIGNAL]
    assert s["as_of"] == "2026-08-25"
    assert "999" not in s["zone"]


# --- the sub-state line -----------------------------------------------------

def test_the_sub_state_line_renders_the_benign_reading(store):
    assert emit_state.build()["signals"][SIGNAL]["zone"] == (
        "Blowup 12 vs 115 · above 200d · canary 0/2 · 12m positive · "
        "above 10M SMA · VIX quiet 0.8484")


def test_the_sub_state_line_capitalises_every_adverse_reading(store):
    store["d"] = _signals(row=_row(aboveSMA200=False, spy12mNeg=True,
                                   belowSMA10m=True, vixRaw=True, canaryBad=2))
    zone = emit_state.build()["signals"][SIGNAL]["zone"]
    assert "BELOW 200d" in zone
    assert "12m NEGATIVE" in zone
    assert "BELOW 10M SMA" in zone
    assert "VIX INVERTED" in zone
    assert "canary 2/2" in zone


def test_vix_raw_is_read_as_a_flag_not_a_level(store):
    """vixRaw is a BOOLEAN in this repo's output, beside a separate numeric
    vixRatio. Treating it as a level would make every session read INVERTED."""
    store["d"] = _signals(row=_row(vixRaw=False))
    assert "VIX quiet" in emit_state.build()["signals"][SIGNAL]["zone"]


# --- shape ------------------------------------------------------------------

def test_emits_exactly_the_one_signal(store):
    assert set(emit_state.build()["signals"]) == {SIGNAL}


def test_the_block_carries_the_required_fields_and_nothing_unknown(store):
    block = emit_state.build()["signals"][SIGNAL]
    assert REQUIRED <= set(block), f"missing {REQUIRED - set(block)}"
    assert set(block) <= REQUIRED | OPTIONAL, f"unknown {set(block) - REQUIRED - OPTIONAL}"


def test_no_score_or_weight_field_is_emitted(store):
    banned = {"score", "weight", "composite", "rank"}
    assert not (banned & set(emit_state.build()["signals"][SIGNAL]))


def test_computed_at_is_null_rather_than_invented(store):
    """This repo publishes no build timestamp. The envelope's emitted_at records
    when the EMITTER ran, which is a different fact and must not stand in."""
    p = emit_state.build()
    assert p["signals"][SIGNAL]["computed_at"] is None
    assert p["emitted_at"]


def test_the_envelope_names_its_version_and_source(store):
    p = emit_state.build()
    assert p["contract_version"] == "1"
    assert p["emitted_by"] == "equity-defense-dashboard"


# --- never emit a guess ------------------------------------------------------

@pytest.mark.parametrize("key", ["endDate", "compositeNow", "vixRatioNow"])
def test_a_missing_meta_key_stops_the_emission(store, key):
    del store["d"]["meta"][key]
    with pytest.raises(emit_state.EmitError, match=key):
        emit_state.build()


@pytest.mark.parametrize("key", ["defenceActive", "cooldownRemaining"])
def test_a_missing_monitor_key_stops_the_emission(store, key):
    del store["d"]["monitor"][key]
    with pytest.raises(emit_state.EmitError, match=key):
        emit_state.build()


@pytest.mark.parametrize("key", ["rolling8d", "aboveSMA200", "canaryBad",
                                 "canaryTotal", "spy12mNeg", "belowSMA10m", "vixRaw"])
def test_a_missing_rolling_key_stops_the_emission(store, key):
    row = _row()
    del row[key]
    store["d"] = _signals(row=row)
    with pytest.raises(emit_state.EmitError, match=key):
        emit_state.build()


def test_all_missing_rolling_keys_are_reported_together(store):
    row = _row()
    del row["rolling8d"], row["canaryBad"]
    store["d"] = _signals(row=row)
    with pytest.raises(emit_state.EmitError) as exc:
        emit_state.build()
    assert "rolling8d" in str(exc.value) and "canaryBad" in str(exc.value)


def test_an_empty_rolling_list_is_refused(store):
    d = _signals()
    d["rolling"] = []
    store["d"] = d
    with pytest.raises(emit_state.EmitError, match="empty"):
        emit_state.build()


def test_a_null_composite_is_refused_rather_than_read_as_zero(store):
    """A null composite defaulting to 0 would publish NEUTRAL on missing data —
    a reassuring state derived from nothing."""
    store["d"]["meta"]["compositeNow"] = None
    with pytest.raises(emit_state.EmitError, match="null"):
        emit_state.build()


def test_a_composite_of_the_wrong_type_is_refused(store):
    store["d"]["meta"]["compositeNow"] = "2"
    with pytest.raises(emit_state.EmitError, match="expected"):
        emit_state.build()


# --- a failed run must not leave a half-written file -------------------------

def test_a_failed_run_writes_nothing_and_exits_non_zero(store, monkeypatch, tmp_path, capsys):
    out = tmp_path / "state.json"
    out.write_text('{"previous": "emission"}', encoding="utf-8")
    monkeypatch.setattr(emit_state, "OUT", out)
    store["d"] = _signals(end_date="2026-08-25", row=_row(date="2026-08-01"))

    assert emit_state.main([]) == 1
    assert json.loads(out.read_text(encoding="utf-8")) == {"previous": "emission"}
    assert "FAILED" in capsys.readouterr().err


def test_an_unchanged_state_is_not_rewritten(store, monkeypatch, tmp_path):
    out = tmp_path / "state.json"
    monkeypatch.setattr(emit_state, "OUT", out)
    assert emit_state.main([]) == 0
    first = out.read_text(encoding="utf-8")
    assert emit_state.main([]) == 0
    assert out.read_text(encoding="utf-8") == first, "unchanged state was rewritten"


def test_a_changed_state_IS_rewritten(store, monkeypatch, tmp_path):
    out = tmp_path / "state.json"
    monkeypatch.setattr(emit_state, "OUT", out)
    assert emit_state.main([]) == 0
    store["d"] = _signals(composite=3)
    assert emit_state.main([]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) \
        ["signals"][SIGNAL]["state"] == "DEFENSIVE"


def test_check_mode_writes_nothing(store, monkeypatch, tmp_path):
    out = tmp_path / "state.json"
    monkeypatch.setattr(emit_state, "OUT", out)
    assert emit_state.main(["--check"]) == 0
    assert not out.exists()
