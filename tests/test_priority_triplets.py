"""Tests semantic source-priority rule use, not just number output."""
import unittest
from labyrinth.source_priority_pairs import (
    generate_priority_triplets, score_priority_triplets,
)


class PriorityTripletTests(unittest.TestCase):
    def test_swaps_change_answer_and_row_reorder_does_not(self):
        triplets=generate_priority_triplets(seed=711, count=8, split="pairprobe")
        self.assertEqual(len(triplets),8)
        for first,swapped,reordered in triplets:
            self.assertEqual(first.answer,reordered.answer)
            self.assertNotEqual(first.answer,swapped.answer)
            self.assertEqual(
                sorted(line for line in first.prompt.splitlines()
                       if line.startswith("- source=")),
                sorted(line for line in swapped.prompt.splitlines()
                       if line.startswith("- source=")),
            )
            self.assertEqual(
                sorted(line for line in first.prompt.splitlines()
                       if line.startswith("- source=")),
                sorted(line for line in reordered.prompt.splitlines()
                       if line.startswith("- source=")),
            )
            self.assertNotEqual(first.prompt,swapped.prompt)
            self.assertNotEqual(first.prompt,reordered.prompt)

    def test_deterministic_and_counts_are_bounded(self):
        self.assertEqual(generate_priority_triplets(45,3,"probe"),
                         generate_priority_triplets(45,3,"probe"))
        with self.assertRaises(ValueError):
            generate_priority_triplets(45,0,"probe")
        with self.assertRaises(ValueError):
            generate_priority_triplets(45,1001,"probe")

    def test_paired_metric_requires_both_swapped_and_invariant_correct(self):
        groups=generate_priority_triplets(721,3,"paircheck")
        correct={case.id:"ANSWER: "+case.answer for group in groups for case in group}
        scored=score_priority_triplets(groups,correct)
        self.assertEqual(scored["triplets_all_correct"],3)
        self.assertEqual(scored["individually_correct"],9)
        self.assertEqual(scored["total_responses"],9)
        wrong=dict(correct)
        wrong[groups[0][1].id]=wrong[groups[0][0].id]
        scored=score_priority_triplets(groups,wrong)
        self.assertEqual(scored["triplets_all_correct"],2)
        self.assertEqual(scored["individually_correct"],8)
        with self.assertRaisesRegex(ValueError,"exact"):
            score_priority_triplets(groups, {**correct, "extra":"ANSWER: 99"})


if __name__=='__main__':
    unittest.main()
