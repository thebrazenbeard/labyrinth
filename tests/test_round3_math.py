"""Independent adaptive-strategy falsifiers for Oracle Chamber PR #142.

Uses a residual-hypothesis minimax solver and direct response-tree enumeration
instead of the implementation's truth tables, verifier or counting routine.
"""

from functools import lru_cache
from itertools import combinations, product
import random
import unittest

from labyrinth.oracle_chamber_exam import make_case
from labyrinth.oracle_chamber_verifier import verify_policy, verify_unsat_bound


def bit_truth(secret, query_index):
    low = secret % 2
    high = secret // 2
    return (low, high, low ^ high)[query_index]


@lru_cache(None)
def can_force_identification(residual, questions_left, lie_limit):
    """Existential question versus universal adversarial answer minimax.

    Residual contains (secret, number_of_lies_used) possibilities consistent
    with the answers seen so far. Early termination is expressly allowed.
    """
    if len({secret for secret, _ in residual}) <= 1:
        return True
    if not questions_left:
        return False
    for question_index in range(3):
        if all(
            not next_states or can_force_identification(
                tuple(sorted(next_states)), questions_left - 1, lie_limit)
            for answer in (0, 1)
            for next_states in ({
                (secret, used + int(bit_truth(secret, question_index) != answer))
                for secret, used in residual
                if used + int(bit_truth(secret, question_index) != answer) <= lie_limit
            },)
        ):
            return True
    return False


def lie_patterns(limit):
    return [set(indices) for length in range(limit + 1)
            for indices in combinations(range(5), length)]


def transcript(secret, corrupt, tree):
    prefix = ""
    for depth in range(5):
        question = tree[prefix]
        prefix += str(bit_truth(secret, question) ^ int(depth in corrupt))
    return prefix


def fixed_schedule_has_unique_decoder(schedule, max_lies):
    outcome_to_secrets = {}
    for secret in range(4):
        for corrupt in lie_patterns(max_lies):
            observed = tuple(bit_truth(secret, query) ^ int(i in corrupt)
                             for i, query in enumerate(schedule))
            outcome_to_secrets.setdefault(observed, set()).add(secret)
    return all(len(choices) == 1 for choices in outcome_to_secrets.values())


class OracleChamberIndependentMath(unittest.TestCase):
    def test_minimax_allows_adaptive_early_commit_but_still_rejects_two_lies(self):
        start = tuple((secret, 0) for secret in range(4))
        self.assertFalse(can_force_identification(start, 4, 1))
        self.assertTrue(can_force_identification(start, 5, 1))
        self.assertFalse(can_force_identification(start, 5, 2))

    def test_adaptive_trees_cannot_break_two_lie_response_volume(self):
        rng = random.Random(142001)
        prefixes = ["".join(bits) for length in range(5)
                    for bits in product("01", repeat=length)]
        attacks = lie_patterns(2)
        self.assertEqual(len(attacks), 16)
        for _ in range(100):
            tree = {prefix: rng.randrange(3) for prefix in prefixes}
            response_owners = {}
            for secret in range(4):
                seen_for_secret = set()
                for corrupt in attacks:
                    answer = transcript(secret, corrupt, tree)
                    self.assertNotIn(answer, seen_for_secret)
                    seen_for_secret.add(answer)
                    response_owners.setdefault(answer, set()).add(secret)
                self.assertEqual(len(seen_for_secret), 16)
            self.assertLessEqual(len(response_owners), 32)
            self.assertTrue(any(len(owners) > 1 for owners in response_owners.values()))

    def test_fixed_query_schedules_agree_with_independent_decoder_criterion(self):
        names = ("alpha", "beta", "parity")
        open_successes = 0
        for schedule in product(range(3), repeat=5):
            declared = {"questions": [names[index] for index in schedule],
                        "decoder": "nearest_codeword"}
            independent_winnable = fixed_schedule_has_unique_decoder(schedule, 1)
            reviewed = verify_policy(1, declared)
            self.assertEqual(reviewed.verdict == "VERIFIED_WIN", independent_winnable)
            open_successes += independent_winnable
            self.assertFalse(fixed_schedule_has_unique_decoder(schedule, 2))
            self.assertNotEqual(verify_policy(2, declared).verdict, "VERIFIED_WIN")
        self.assertGreater(open_successes, 0)

    def test_certificate_exactly_covers_64_vs_32_under_declared_rules(self):
        sealed = make_case(0, sealed=True)
        volume = 4 * sum(1 for _ in lie_patterns(2))
        capacity = len(list(product((0, 1), repeat=5)))
        self.assertEqual((volume, capacity), (64, 32))
        proved = verify_unsat_bound(sealed, {"declared_rules": {
            "secrets": 4, "binary_answers": 2,
            "max_queries": 5, "lie_budget": 2}})
        self.assertTrue(proved["certificate_verified"])
        self.assertEqual(proved["verdict"], "PROVEN_UNSAT_UNDER_DECLARED_RULES")
        opened = make_case(0)
        self.assertFalse(verify_unsat_bound(opened, {"declared_rules": {
            "secrets": 4, "binary_answers": 2,
            "max_queries": 5, "lie_budget": 1}})["certificate_verified"])


if __name__ == "__main__":
    unittest.main()
