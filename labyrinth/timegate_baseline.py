"""ZERO-UPDATE public-observation rule parser for the timed-gate exam.

No optimizer, memory, model, or access to hidden Case fields is used by the
policy. The harness passes only the candidate-facing view and act/submit
callbacks. This control is intentionally strong enough to expose easy tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


@dataclass(frozen=True)
class BaselineReceipt:
    verdict: str
    actions: tuple[str, ...]
    observation: str


def run_public_rule_parser(view: dict[str, object],
                           act: Callable[[str], dict[str, object]],
                           submit: Callable[[dict[str, object]], dict[str, object]]) -> BaselineReceipt:
    """Fixed parse-and-schedule code; does not peek at hidden state or labels."""
    prompt = str(view["prompt"])
    charge_phase, period = map(int, re.search(
        r"Charging requires clock phase (\d+) in a cycle of (\d+)", prompt).groups())
    pass_phase = int(re.search(r"gate accepts escape only at phase (\d+)",
                               prompt).group(1))
    phase = int(re.search(r"Clock: (\d+)/(\d+)", str(view["observation"])).group(1))
    attempted: list[str] = []
    last = {"observation": view["observation"], "verdict": "UNKNOWN"}

    def do(action: str) -> None:
        nonlocal last, phase
        last = act(action)
        attempted.append(action)
        phase = (phase + 1) % period

    for action in ("forward", "take", "forward"):
        do(action)
    while phase != charge_phase:
        do("wait")
    do("arm")
    if "Power remaining: 0." in str(last["observation"]):
        # Failure of the predicted public rule. Do not guess that the
        # world is impossible: it may be sealed or an unmodeled mechanism.
        unknown = submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})
        return BaselineReceipt(str(unknown["verdict"]), tuple(attempted),
                               str(last["observation"]))
    do("forward")
    while phase != pass_phase:
        do("wait")
    do("escape")
    if last["verdict"] == "VERIFIED_WIN":
        return BaselineReceipt("VERIFIED_WIN", tuple(attempted),
                               str(last["observation"]))
    result = submit({"kind": "ABSTAIN", "assessment": "UNKNOWN"})
    return BaselineReceipt(str(result["verdict"]), tuple(attempted),
                           str(last["observation"]))
