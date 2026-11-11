"""Fifth source-public Minotaur challenge: private-information simultaneous action.

Two actors cannot exchange observations. Each commits to a deterministic local
lookup policy before the world is selected. A winner must satisfy EVERY
declared possible world, not just the world an evaluator happened to sample.

This is a small, exhaustively bounded policy problem; none of these functions
trains a neural model or supports a claim of learned structural transfer.
"""

from dataclasses import dataclass
import hashlib
import json
import unicodedata


WIN = "VERIFIED_WIN"
FAILURE = "VERIFIED_FAILURE"
NO_KNOWN_WIN = "NO_KNOWN_WIN"
UNSAT = "PROVEN_UNSAT_UNDER_DECLARED_RULES"
UNKNOWN = "UNKNOWN"
NOVEL = "NOVEL_RULESET_CHALLENGE"


@dataclass(frozen=True)
class World:
    left_obs: int
    right_obs: int
    required_parity: int


@dataclass(frozen=True)
class Scenario:
    worlds: tuple[World, ...]
    left_observations: int
    right_observations: int
    total_on_budget: int


def check_scenario(s):
    """Reject malformed or oversized declarations without semantic repair."""
    if type(s) is not Scenario or type(s.worlds) is not tuple:
        raise ValueError("scenario and worlds must have exact declared types")
    if (type(s.left_observations) is not int or
            type(s.right_observations) is not int or
            not 2 <= s.left_observations <= 4 or
            not 2 <= s.right_observations <= 4 or
            type(s.total_on_budget) is not int or
            not 0 <= s.total_on_budget <=
            s.left_observations + s.right_observations):
        raise ValueError("bounded exact local observation counts and budget required")
    if not 1 <= len(s.worlds) <= 12:
        raise ValueError("expected 1..12 possible worlds")
    for world in s.worlds:
        if (type(world) is not World or
                type(world.left_obs) is not int or
                type(world.right_obs) is not int or
                type(world.required_parity) is not int or
                not 0 <= world.left_obs < s.left_observations or
                not 0 <= world.right_obs < s.right_observations or
                world.required_parity not in (0, 1)):
            raise ValueError("world has invalid local information or target bit")
    return s


def fingerprint(s):
    check_scenario(s)
    payload = {
        "mechanics": "distributed-parity-v1",
        "left_observations": s.left_observations,
        "right_observations": s.right_observations,
        "total_on_budget": s.total_on_budget,
        "worlds": [[w.left_obs, w.right_obs, w.required_parity]
                   for w in s.worlds],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _policy_shape(s, policy):
    return (type(policy) is tuple and len(policy) == 2 and
            all(type(half) is tuple for half in policy) and
            len(policy[0]) == s.left_observations and
            len(policy[1]) == s.right_observations and
            all(type(bit) is int and bit in (0, 1)
                for half in policy for bit in half))


def verify_policy(s, policy):
    """Source replay; intentionally separate from independent witness checker."""
    rules = fingerprint(s)
    if not _policy_shape(s, policy):
        return {"verdict": FAILURE, "rules_fingerprint": rules,
                "reason": "malformed fixed local policy"}
    if sum(policy[0]) + sum(policy[1]) > s.total_on_budget:
        return {"verdict": FAILURE, "rules_fingerprint": rules,
                "reason": "total asserted-one budget exceeded"}
    for i, world in enumerate(s.worlds):
        if (policy[0][world.left_obs] ^ policy[1][world.right_obs]) != world.required_parity:
            return {"verdict": FAILURE, "rules_fingerprint": rules,
                    "reason": "counterexample world", "counterexample_index": i}
    return {"verdict": WIN, "rules_fingerprint": rules,
            "verified_worlds": len(s.worlds)}


def independent_witness_check(s, policy):
    """Different direct implementation; no call to source verifier or solver."""
    check_scenario(s)
    if not (isinstance(policy, tuple) and type(policy) is tuple and
            len(policy) == 2 and type(policy[0]) is tuple and
            type(policy[1]) is tuple and
            len(policy[0]) == s.left_observations and
            len(policy[1]) == s.right_observations):
        return False
    bits = list(policy[0]) + list(policy[1])
    if any(type(x) is not int or x < 0 or x > 1 for x in bits):
        return False
    if bits.count(1) > s.total_on_budget:
        return False
    return all(bits[w.left_obs] + bits[s.left_observations + w.right_obs]
               in (w.required_parity, w.required_parity + 2)
               for w in s.worlds)


def independent_gf2_exists(s, *, state_cap=100000, fixed_policy=None):
    """Independent row elimination over GF(2), then bounded free-bit search.

    Cardinality budget is not linear, so free assignments must be enumerated.
    Exhaustion is a certificate only for these exact finite declared worlds.
    """
    check_scenario(s)
    if type(state_cap) is not int or state_cap < 0:
        return None
    if state_cap == 0:
        return None
    n = s.left_observations + s.right_observations
    constraints = [((1 << w.left_obs) |
                    (1 << (s.left_observations + w.right_obs)),
                    w.required_parity) for w in s.worlds]
    if fixed_policy is not None:
        if (type(fixed_policy) is not tuple or len(fixed_policy) != 2 or
                type(fixed_policy[0]) is not tuple or
                type(fixed_policy[1]) is not tuple or
                len(fixed_policy[0]) != s.left_observations or
                len(fixed_policy[1]) != s.right_observations):
            return False
        fixed_bits = fixed_policy[0] + fixed_policy[1]
        if any(type(bit) is not int or bit not in (0, 1)
               for bit in fixed_bits):
            return False
        # Pin every variable algebraically, so existence now checks the
        # SPECIFIC claimed witness rather than some unrelated valid policy.
        constraints.extend((1 << i, bit) for i, bit in enumerate(fixed_bits))
    rows = {}
    for coeff, rhs in constraints:
        while coeff:
            pivot = (coeff & -coeff).bit_length() - 1
            if pivot not in rows:
                rows[pivot] = (coeff, rhs)
                break
            known, known_rhs = rows[pivot]
            coeff ^= known
            rhs ^= known_rhs
        if coeff == 0 and rhs:
            return False
    free = [i for i in range(n) if i not in rows]
    attempts = 0
    for free_code in range(1 << len(free)):
        if attempts >= state_cap:
            return None
        attempts += 1
        values = [0] * n
        for j, index in enumerate(free):
            values[index] = (free_code >> j) & 1
        for pivot in sorted(rows, reverse=True):
            coeff, rhs = rows[pivot]
            # All columns in the row other than its least-set pivot are greater.
            other = coeff & ~(1 << pivot)
            parity = rhs
            while other:
                low = other & -other
                parity ^= values[low.bit_length() - 1]
                other ^= low
            values[pivot] = parity
        if sum(values) <= s.total_on_budget:
            return True
    return False


def independent_dfs_exists(s, *, state_cap=100000):
    """Independent constraint-propagating DFS; not GF(2), not source enumeration."""
    check_scenario(s)
    if type(state_cap) is not int or state_cap < 0:
        return None
    n_left = s.left_observations
    n = n_left + s.right_observations
    slots = [-1] * n
    count = 0

    def recurse(index, used):
        nonlocal count
        if count >= state_cap:
            return None
        count += 1
        for w in s.worlds:
            left, right = slots[w.left_obs], slots[n_left + w.right_obs]
            if left != -1 and right != -1 and (left + right) % 2 != w.required_parity:
                return False
        if used > s.total_on_budget:
            return False
        if index == n:
            return True
        saw_unknown = False
        for bit in (0, 1):
            slots[index] = bit
            result = recurse(index + 1, used + bit)
            if result is True:
                slots[index] = -1
                return True
            saw_unknown |= result is None
        slots[index] = -1
        return None if saw_unknown else False

    return recurse(0, 0)


def search_policy(s, *, node_limit=100000, independent_cap=100000):
    """Exhaust finite policies; independent agreement gates success and UNSAT."""
    rules = fingerprint(s)
    if (type(node_limit) is not int or node_limit < 0 or
            type(independent_cap) is not int or independent_cap < 0):
        return {"verdict": UNKNOWN, "rules_fingerprint": rules,
                "reason": "invalid resource cap"}
    n_left = s.left_observations
    n = n_left + s.right_observations
    examined = 0
    for code in range(1 << n):
        if examined >= node_limit:
            return {"verdict": NO_KNOWN_WIN, "rules_fingerprint": rules,
                    "policies_examined": examined}
        examined += 1
        bits = tuple((code >> i) & 1 for i in range(n))
        proposal = (bits[:n_left], bits[n_left:])
        if verify_policy(s, proposal)["verdict"] == WIN:
            exact = independent_witness_check(s, proposal)
            algebra = independent_gf2_exists(s, state_cap=independent_cap)
            pinned = independent_gf2_exists(s, state_cap=independent_cap,
                                            fixed_policy=proposal)
            depth = independent_dfs_exists(s, state_cap=independent_cap)
            if exact and pinned is True and algebra is True and depth is True:
                return {"verdict": WIN, "rules_fingerprint": rules,
                        "policy": proposal, "policies_examined": examined,
                        "independent_witness": True,
                        "independent_gf2": True, "independent_gf2_pinned": True,
                        "independent_dfs": True}
            return {"verdict": UNKNOWN, "rules_fingerprint": rules,
                    "reason": "source and independent verifier disagree"}
    algebra = independent_gf2_exists(s, state_cap=independent_cap)
    depth = independent_dfs_exists(s, state_cap=independent_cap)
    if algebra is False and depth is False:
        return {"verdict": UNSAT, "rules_fingerprint": rules,
                "policies_examined": examined,
                "independent_gf2": False, "independent_dfs": False,
                "certificate": {"kind": "complete-deterministic-policy-enumeration",
                                "enumerated": examined,
                                "finite_space_size": 1 << n,
                                "valid_policies_found": 0}}
    return {"verdict": UNKNOWN, "rules_fingerprint": rules,
            "reason": "source exhaustion not corroborated"}


def adjudicate_unexpected(s, proposal, prior_verdict=UNKNOWN):
    """A valid surprise overrules a previously asserted impossibility."""
    rules = fingerprint(s)
    actual = verify_policy(s, proposal)
    independent = independent_witness_check(s, proposal)
    pinned = independent_gf2_exists(s, fixed_policy=proposal)
    trusted = (type(actual) is dict and actual.get("rules_fingerprint") == rules
               and actual.get("verdict") in (WIN, FAILURE))
    if not trusted or (actual["verdict"] == WIN) != independent or pinned != independent:
        return {"verdict": UNKNOWN, "rules_fingerprint": rules,
                "independently_replayed": False,
                "reason": "replay disagreement; preserve challenge"}
    return {"verdict": WIN if independent else FAILURE,
            "rules_fingerprint": rules, "independently_replayed": True,
            "overturned_prior": bool(independent and prior_verdict == UNSAT)}


def novel_rule_challenge(s, proposal):
    """A proposed communication/measurement rule needs an entirely new exam."""
    rules = fingerprint(s)
    if type(proposal) is not dict:
        return {"verdict": UNKNOWN, "rules_fingerprint": rules}

    def normalized_terms(value):
        if type(value) is not str or any(unicodedata.category(c)[0] == "C"
                                          for c in value):
            return ""
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return " ".join("".join(c if c.isalnum() else " "
                                for c in normalized).split())

    channel = normalized_terms(proposal.get("new_channel"))
    reason = normalized_terms(proposal.get("reason"))
    # Only recognize a proposed rule DELTA. Lexical novelty is not enough:
    # plural/punctuated established capabilities or explicit absence of a
    # channel are no-op descriptions, not novel physical communication.
    known_noops = {
        "none", "no channel", "no new channel", "no communication",
        "private observation", "private observations",
        "simultaneous output", "simultaneous outputs",
        "without communication", "existing channel", "same rules",
    }
    denial_prefix = channel.split(" ", 1)[0] in {
        "no", "none", "without", "existing", "unchanged", "current",
    }
    if not channel or channel in known_noops or denial_prefix or not reason:
        return {"verdict": UNKNOWN, "rules_fingerprint": rules}
    return {"verdict": NOVEL, "rules_fingerprint": rules,
            "independent_verification": "NOT_PERFORMED",
            "reason": "new channel changes declared no-communication mechanics"}


def families():
    xor_worlds = tuple(World(x, y, x ^ y) for x in (0, 1) for y in (0, 1))
    and_worlds = tuple(World(x, y, x & y) for x in (0, 1) for y in (0, 1))
    return {
        "coordinated_parity": Scenario(xor_worlds, 2, 2, 2),
        "missing_corner": Scenario(and_worlds[:-1], 2, 2, 0),
        "and_paradox": Scenario(and_worlds, 2, 2, 4),
        "inconsistent_reports": Scenario((World(0, 0, 0), World(0, 0, 1)),
                                         2, 2, 4),
    }


_VIEWS = (
    "Two silent crews", "Sealed control rooms", "Two disconnected probes",
    "Mutual-aid reactors", "Twin witness stations", "A pair of firewalls",
    "Isolated orbital nodes", "Split-brain rescue teams",
)


def public_view(s, *, variant=0):
    check_scenario(s)
    if type(variant) is not int or variant < 0:
        raise ValueError("variant must be nonnegative integer")
    framing = _VIEWS[variant % len(_VIEWS)]
    scenarios = ", ".join(f"({w.left_obs},{w.right_obs})->{w.required_parity}"
                           for w in s.worlds)
    return {"mechanics_fingerprint": fingerprint(s),
            "prompt": (f"{framing}: isolated actors observe their own local index, "
                       "cannot communicate, and simultaneously assert one bit. "
                       "Commit to one deterministic lookup per actor that makes "
                       "the XOR of assertions equal the required bit in EVERY "
                       f"possible world: {scenarios}. Total asserted-one "
                       f"lookup entries may not exceed {s.total_on_budget}. "
                       "If impossible, explain a finite certificate. If you "
                       "propose communication, identify it as a changed rule.")}


def fixed_script_control():
    # A plausible rigid single-table script that cannot adapt to private worlds.
    constant = ((0, 1), (0, 0))
    solved = [search_policy(s)["verdict"] for s in families().values()]
    return {"fixed_script_wins": sum(verify_policy(s, constant)["verdict"] == WIN
                                      for s in families().values()),
            "symbolic_wins": solved.count(WIN), "optimizer_updates": 0,
            "control_type": "fixed local policy; no neural parameters"}


def catalogue_control(unseen):
    check_scenario(unseen)
    records = {fingerprint(s): search_policy(s)["verdict"]
               for s in families().values()}
    return {"public_fixtures": len(records), "exact_matches": len(records),
            "unseen_verdict": records.get(fingerprint(unseen), UNKNOWN),
            "optimizer_updates": 0,
            "control_type": "exact public-fingerprint memorizer"}
