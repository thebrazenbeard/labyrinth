"""Independent finite-policy decision for Rule-Reversal Tribunal.

This code deliberately does not import the tribunal transition, replay,
state, search, or observation functions. States are bit-packed primitives and
the transition is separately expressed from the declared causal XOR rules.
All admitted worlds (at most six) and budgets (at most eight) are supported.
This is a *second implementation* of declared rules, not independent custody
or a proof about new actions outside those rules.
"""

from functools import lru_cache


def can_guarantee(world_masks: tuple[tuple[int, int, int, int], ...],
                  budget: int, sensor: str) -> bool:
    """Can one observation-dependent policy reach both indicators in all worlds?"""
    if (type(world_masks) is not tuple or not 1 <= len(world_masks) <= 6
            or type(budget) is not int or not 1 <= budget <= 8
            or type(sensor) is not str or sensor not in ("full", "parity", "sealed")
            or any(type(w) is not tuple or len(w) != 4
                   or any(type(mask) is not int or mask not in (0, 1, 2, 3)
                          for mask in w) for w in world_masks)):
        raise ValueError("invalid independent finite-policy problem")

    def intervene(world_index: int, encoded: int, choice: int):
        bits = encoded & 3
        shifted = (encoded >> 2) & 1
        had_pre = (encoded >> 3) & 1
        had_post = (encoded >> 4) & 1
        if choice == 2:  # A legal SHIFT needs an earlier intervention.
            if shifted or not had_pre:
                return None
            shifted = 1
        else:
            bits ^= world_masks[world_index][choice + 2 * shifted]
            if shifted:
                had_post = 1
            else:
                had_pre = 1
        new_state = bits | shifted << 2 | had_pre << 3 | had_post << 4
        prefix = "POST:" if shifted else "PRE:"
        if sensor == "full":
            observed = prefix + format(bits, "02b")
        elif sensor == "parity":
            observed = prefix + "P" + str(((bits & 1) + ((bits >> 1) & 1)) % 2)
        else:
            observed = prefix + "SEALED"
        return new_state, observed

    @lru_cache(None)
    def exists(belief: tuple[tuple[int, int], ...], fuel: int) -> bool:
        if fuel < 1:
            return False
        # The remaining action is COMMIT. Encoded 31 denotes both indicators
        # on, SHIFT completed, and one physical intervention in each phase.
        if all(state == 31 for _, state in belief):
            return True
        if fuel == 1:
            return False
        for choice in range(3):  # A, B, SHIFT; COMMIT is checked above.
            by_observation: dict[str, list[tuple[int, int]]] = {}
            for world_index, state in belief:
                result = intervene(world_index, state, choice)
                if result is None:
                    break
                nxt, observed = result
                by_observation.setdefault(observed, []).append((world_index, nxt))
            else:
                if all(exists(tuple(part), fuel - 1)
                       for part in by_observation.values()):
                    return True
        return False

    return exists(tuple((index, 0) for index in range(len(world_masks))), budget)
