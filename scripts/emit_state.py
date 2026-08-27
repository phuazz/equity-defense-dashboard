"""emit_state.py — publish this repo's composite state in the STATE_CONTRACT shape.

WHAT THIS IS FOR
----------------
A private consumer (the command centre) renders this dashboard's defence
composite beside signals from seven other projects. Until now it did that by
reaching INTO this repo and reading exact JSON pointers out of
`data/signals.json` from its own side. That works, and it is guarded there, but
it puts knowledge of THIS repo's field names in somebody else's codebase:
rename a key here and the break surfaces over there, days later, in a file
nobody was editing at the time.

This writes `data/state.json` beside the data it describes, so a rename breaks
here, in this repo's CI, at the moment of the rename.

WHY IT READS ONLY TWO THINGS OUT OF AN 8 MB FILE
-------------------------------------------------
`data/signals.json` carries 7,000-odd daily rows and is over 8 MB. The state is
entirely determined by `meta` and the LAST rolling row, so that is all this
takes. The file is parsed with the standard library and never held beyond the
two objects needed.

THE DRIFT CHECK IS THE POINT, NOT A FORMALITY
----------------------------------------------
`meta.endDate` and `rolling[-1].date` describe the same session by two
different routes. If a pipeline change ever left them disagreeing, the
composite would be read against one date and its sub-states against another —
a wrong state carrying a plausible date, which is the failure mode hardest to
notice downstream. They are asserted equal, and a mismatch stops the emission.

WHAT IT IS NOT
--------------
  * NOT a new signal and not a recalculation. Every value is copied from a file
    this repo already publishes. If this and `signals.json` ever disagree,
    `signals.json` is right and this is broken.
  * NOT load-bearing here. Nothing in this repo reads `data/state.json`, which
    is why it runs in its own workflow: it must never be able to fail the
    overlay update.

NO computed_at IS EMITTED, DELIBERATELY. This repo publishes no build timestamp
in `signals.json` — `meta` carries dates and counts but no generated-at field —
so the optional `computed_at` is left null rather than filled with something
adjacent. The envelope's own `emitted_at` records when THIS script ran, which
is a different fact and must not be passed off as when the signals were
computed.

NOTE ON COVERAGE. This repo has no test suite and its CI does not run pytest,
so `tests/test_emit_state.py` is a local guard rather than a gate. The
enforcing check is on the consumer side, which validates every emission against
its own registry and compares it against its own independent extraction on
every run.

Usage:
    python scripts/emit_state.py           # write data/state.json
    python scripts/emit_state.py --check   # validate and print, write nothing
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = DATA / "state.json"

CONTRACT_VERSION = "1"
SOURCE = "equity-defense-dashboard"
SIGNAL = "equity_defense_composite"

# The composite is defensive at 2 or more. The threshold lives in the consumer's
# policy as well; it is repeated here because the state has to be named
# somewhere, and a disagreement between the two is reported by the consumer's
# comparison rather than silently resolved.
DEFENSIVE_AT = 2

# Sub-state rows that must be present on the last rolling record before a state
# can be described. Listed rather than fetched one by one so a partial row is
# reported in full instead of one key at a time.
ROLLING_KEYS = ("date", "rolling8d", "aboveSMA200", "canaryBad", "canaryTotal",
                "spy12mNeg", "belowSMA10m", "vixRaw")

# The blow-up count is quoted against its own calibrated bar, which belongs in
# the description of the state rather than as a bare number.
BLOWUP_BAR = 115


class EmitError(Exception):
    """A required input was missing or malformed. Never emit a guess."""


def require(obj, path: str, kind=None):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise EmitError(f"missing key `{part}` at pointer `{path}`")
        cur = cur[part]
    if cur is None:
        raise EmitError(f"pointer `{path}` is null")
    if kind is not None and not isinstance(cur, kind):
        # `kind` is often a tuple of accepted types, which has no __name__.
        want = kind.__name__ if isinstance(kind, type) else "/".join(k.__name__ for k in kind)
        raise EmitError(f"pointer `{path}` is {type(cur).__name__}, expected {want}")
    return cur


def load_signals():
    p = DATA / "signals.json"
    if not p.exists():
        raise EmitError(f"source file not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EmitError(f"signals.json is not valid JSON: {exc}") from exc


def describe(row: dict, vix_ratio) -> str:
    """The sub-state line: what the composite is made of, in words.

    Booleans are rendered as the condition they represent, with the adverse
    reading in capitals, so the line can be read without knowing which way each
    flag points.
    """
    return (
        f"Blowup {row['rolling8d']} vs {BLOWUP_BAR} · "
        f"{'above' if row['aboveSMA200'] else 'BELOW'} 200d · "
        f"canary {row['canaryBad']}/{row['canaryTotal']} · "
        f"12m {'NEGATIVE' if row['spy12mNeg'] else 'positive'} · "
        f"{'BELOW' if row['belowSMA10m'] else 'above'} 10M SMA · "
        f"VIX {'INVERTED' if row['vixRaw'] else 'quiet'} {vix_ratio}"
    )


def build() -> dict:
    d = load_signals()

    composite = require(d, "meta.compositeNow", (int, float))
    end_date = require(d, "meta.endDate", str)
    vix_ratio = require(d, "meta.vixRatioNow", (int, float))
    # Read for the drift check below; not emitted as schema fields.
    require(d, "monitor.defenceActive", bool)
    require(d, "monitor.cooldownRemaining", (int, float))

    rolling = require(d, "rolling", list)
    if not rolling:
        raise EmitError("rolling[] is empty — there is no session to describe")
    row = rolling[-1]
    if not isinstance(row, dict):
        raise EmitError(f"rolling[-1] is {type(row).__name__}, expected an object")

    missing = [k for k in ROLLING_KEYS if k not in row]
    if missing:
        raise EmitError(f"rolling[-1] missing {missing}")

    # Two routes to the same session. A disagreement means a wrong state would
    # be published against a plausible-looking date.
    if row["date"] != end_date:
        raise EmitError(
            f"rolling[-1].date {row['date']} != meta.endDate {end_date} — "
            "the composite and its sub-states describe different sessions"
        )

    state = "DEFENSIVE" if composite >= DEFENSIVE_AT else "NEUTRAL"

    return {
        "contract_version": CONTRACT_VERSION,
        "emitted_by": SOURCE,
        "emitted_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "signals": {
            SIGNAL: {
                "as_of": end_date,
                "state": state,
                "value": composite,
                "zone": describe(row, vix_ratio),
                "role": "risk-state",
                "horizon": "weeks-months",
                "evidence_grade": "adopted-gate",
                "licence": "public",
                "action_hint": "watch" if state == "DEFENSIVE" else "none",
                "source_file": "data/signals.json",
                "computed_at": None,   # this repo publishes no build timestamp
                "cadence": "daily",
            }
        },
    }


def unchanged(payload: dict) -> bool:
    """Same emission as the one on disk, apart from the run's own timestamp?

    `emitted_at` moves every run, so writing unconditionally would leave a diff
    every time and the workflow would commit a no-op on every run. Liveness does
    not need that commit: the consumer judges freshness from `as_of`, so a dead
    emitter still shows up there as a stale state.
    """
    if not OUT.exists():
        return False
    try:
        prev = json.loads(OUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    strip = lambda d: {k: v for k, v in d.items() if k != "emitted_at"}
    return strip(prev) == strip(payload)


def main(argv: list[str]) -> int:
    try:
        payload = build()
    except EmitError as exc:
        print(f"emit_state: FAILED — {exc}", file=sys.stderr)
        print("emit_state: nothing written; the previous state.json is left as it was.",
              file=sys.stderr)
        return 1

    s = payload["signals"][SIGNAL]
    print(f"emit_state: composite {s['value']} → {s['state']} @ {s['as_of']}")
    print(f"            {s['zone']}")

    if "--check" in argv:
        print("emit_state: --check, nothing written.")
        return 0

    if unchanged(payload):
        print("emit_state: state unchanged since the last emission — leaving it as it is.")
        return 0

    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        shown = OUT.relative_to(REPO)
    except ValueError:
        shown = OUT
    print(f"emit_state: wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
