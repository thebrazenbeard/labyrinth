"""Independent oracles and split checks for length-matched OOD tasks."""

from dataclasses import FrozenInstanceError
from collections import Counter
import re
import unittest

from labyrinth.matched_holdouts import (
    freeze_matched_manifest,
    generate_matched_length_pairs,
)
from labyrinth.program_maze import generate_program_challenges


def parse_and_run(prompt: str) -> int:
    """Compute a result from the serialized *prompt*, never from its answer."""
    if "Initial registers: A=" in prompt:
        match = re.search(r"Initial registers: A=(\d+), B=(\d+)", prompt)
        assert match
        state = {"A": int(match[1]), "B": int(match[2])}
        steps = re.findall(r"^\d+\. (ADD|SUB|MUL) ([AB]) ([AB]|\d+)$", prompt, re.M)
    else:
        match = re.search(r"Initial state \(B first\): B=(\d+), A=(\d+)", prompt)
        assert match
        state = {"A": int(match[2]), "B": int(match[1])}
        steps = []
        translated = {"plus": "ADD", "minus": "SUB", "times": "MUL"}
        for dst, op, rhs_type, rhs in re.findall(
            r"^row-\d+ :: destination ([AB]) :: (plus|minus|times) "
            r":: (register|literal) ([AB]|\d+)$", prompt, re.M
        ):
            assert (rhs_type == "register") == (rhs in ("A", "B"))
            steps.append((translated[op], dst, rhs))
    assert steps
    for op, dst, rhs in steps:
        value = state[rhs] if rhs in state else int(rhs)
        if op == "ADD":
            state[dst] = (state[dst] + value) % 97
        elif op == "SUB":
            state[dst] = (state[dst] - value) % 97
        elif op == "MUL":
            state[dst] = (state[dst] * value) % 97
        else:
            raise AssertionError("unknown operator")
    return state["A"]


class MatchedHoldoutTests(unittest.TestCase):
    def test_paraphrases_have_same_answer_and_independent_oracle(self):
        pairs = generate_matched_length_pairs(4343, 160)
        self.assertEqual(pairs, generate_matched_length_pairs(4343, 160))
        for first, changed in pairs:
            self.assertEqual(first.answer, changed.answer)
            self.assertNotEqual(first.prompt, changed.prompt)
            self.assertNotEqual(first.id, changed.id)
            self.assertEqual(first.split, "final_matched")
            self.assertEqual(changed.split, "final_matched")
            self.assertEqual(int(first.answer), parse_and_run(first.prompt))
            self.assertEqual(int(changed.answer), parse_and_run(changed.prompt))
            self.assertEqual(dict(first.metadata)["pair_id"], dict(changed.metadata)["pair_id"])

    def test_length_and_operator_histogram_match_train_without_structure_overlap(self):
        samples = [pair[0] for pair in generate_matched_length_pairs(5151, 80)]
        train = generate_program_challenges(5151, 1000, "train")
        train_structures = {dict(x.metadata)["structure_fingerprint"] for x in train}
        matched_structures = {dict(x.metadata)["structure_fingerprint"] for x in samples}
        self.assertTrue(matched_structures.isdisjoint(train_structures))
        self.assertEqual(len(matched_structures), 4)

        def profile(task):
            operations = [s.split()[0] for s in dict(task.metadata)["structure"].split(";")]
            return len(operations), tuple(sorted(Counter(operations).items()))

        expected = Counter({p: 20 for p in set(map(profile, train)) if p[0] == 3})
        expected.update({p: 40 for p in set(map(profile, train)) if p[0] == 4})
        self.assertEqual(Counter(map(profile, samples)), expected)
        # The two distinct four-step structures each contribute 20 items.
        self.assertEqual({len(dict(t.metadata)["structure"].split(";")) for t in samples}, {3, 4})

    def test_manifest_is_immutable_source_bound_and_contains_no_answer_fields(self):
        pairs = generate_matched_length_pairs(88, 5)
        sha = "277b37a0131788e4db7621f3bc54b438737262e3"
        m1 = freeze_matched_manifest(88, pairs, source_commit=sha)
        m2 = freeze_matched_manifest(88, pairs, source_commit=sha)
        self.assertEqual(m1, m2)
        self.assertEqual(len(m1.entries), 10)
        self.assertEqual(len({e.case_id for e in m1.entries}), 10)
        self.assertEqual(m1.commitment, m2.commitment)
        self.assertNotIn('"answer"', m1.to_json())
        with self.assertRaises(FrozenInstanceError):
            m1.seed = 5
        changed = freeze_matched_manifest(89, generate_matched_length_pairs(89, 5), source_commit=sha)
        self.assertNotEqual(changed.commitment, m1.commitment)
        with self.assertRaises(ValueError):
            freeze_matched_manifest(88, pairs, source_commit="main")

    def test_invalid_inputs(self):
        for seed, count in [(True, 1), (-1, 1), (0, True), (0, 0), (0, 1001)]:
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                generate_matched_length_pairs(seed, count)


if __name__ == "__main__":
    unittest.main()
