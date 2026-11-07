"""Regression: a declared public binary answer must survive bridge serialization."""

import unittest

from labyrinth.minotaur_candidate_bridge import (FixedActionAdapter, _public,
                                                 run_candidate)


class PublicQuerySession:
    def __init__(self):
        self._answers = []
        self._finished = False

    def view(self):
        return {"answers": list(self._answers),
                "remaining_queries": 5 - len(self._answers),
                "finished": self._finished, "verdict": "UNKNOWN"}

    def ask(self, question):
        if question != "alpha":
            raise ValueError("unsupported query")
        self._answers.append({"question": "alpha", "answer": 0})
        return self.view()

    def submit(self, claim):
        self._finished = True
        return self.view()


class BridgePublicQueryTests(unittest.TestCase):
    def test_binary_query_reply_is_not_misclassified_private(self):
        start = {"answers": [], "remaining_queries": 5}
        receipt = run_candidate(start, PublicQuerySession,
                                FixedActionAdapter(("alpha",)))
        self.assertNotEqual(receipt.final_verdict, "EVALUATOR_ERROR")
        self.assertEqual(receipt.turns[0].disposition, "ACTION")
        self.assertIsNotNone(receipt.turns[0].public_feedback_json)

    def test_naked_private_answer_remains_forbidden(self):
        with self.assertRaises(ValueError):
            _public({"answer": 0})


if __name__ == "__main__":
    unittest.main()
