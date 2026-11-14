"""Explicitly non-neural controls for the Cipher Wheel structural split."""

from __future__ import annotations

from collections import deque

from .cipher_wheel_world import CandidateSession, LABELS, MAX_ATTEMPTS


def _shortest_known_path(start: str, target: str,
                         edges: dict[str, dict[str, str]]) -> tuple[str, ...] | None:
    pending = deque([(start, ())])
    seen = {start}
    while pending:
        here, path = pending.popleft()
        if here == target:
            return path
        for action, destination in sorted(edges[here].items()):
            if destination not in seen:
                seen.add(destination)
                pending.append((destination, (*path, action)))
    return None


def symbolic_explorer(session: CandidateSession) -> dict[str, object]:
    """Learn a complete table using ONLY public observations, no world attrs.

    This is a deterministic graph-discovery baseline. Passing it proves that
    the public experiment is *solvable*, not that a neural model generalized.
    """
    initial = session.view()
    start = initial["current"]
    edges: dict[str, dict[str, str]] = {start: {}}
    for _ in range(8):
        choices = []
        for token, known in edges.items():
            path = _shortest_known_path(start, token, edges)
            if path is not None:
                for action in ("A", "B"):
                    if action not in known:
                        choices.append((len(path), token, action, path))
        if not choices:
            break
        _, token, action, path = min(choices)
        session.act("RESET")
        for step in path:
            session.act(step)
        if session.view()["current"] != token:
            raise RuntimeError("public observations disagree with learned graph")
        result = session.act(action)
        destination = result["current"]
        edges[token][action] = destination
        edges.setdefault(destination, {})
        if result["finished"]:
            break
    if set(edges) != set(LABELS) or any(set(edges[token]) != {"A", "B"} for token in edges):
        return {"verdict": "UNKNOWN", "reason": "INCOMPLETE_PUBLIC_GRAPH",
                "observations": len(session.view()["history"]), "verified_success": False}
    table = {action: {token: edges[token][action] for token in LABELS}
             for action in ("A", "B")}
    return session.submit_table(table)


# A public fixed answer for a single known training seed; see hostile tests.
# It neither observes nor adapts. Other seeds and the HELDOUT split must fail.
MEMORIZED_TRAIN_SEED0: dict[str, dict[str, str]] = {
    "A": {"amber": "jade", "cobalt": "amber", "jade": "violet", "violet": "cobalt"},
    "B": {"amber": "violet", "cobalt": "cobalt", "jade": "jade", "violet": "amber"},
}


def memorized_control(session: CandidateSession) -> dict[str, object]:
    answer = {action: dict(table) for action, table in MEMORIZED_TRAIN_SEED0.items()}
    return session.submit_table(answer)
