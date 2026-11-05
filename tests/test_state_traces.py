import unittest

from labyrinth.program_maze import generate_program_challenges
from labyrinth.state_traces import verified_state_trace


class StateTraceTests(unittest.TestCase):
    def test_independent_worked_example(self):
        record = {
            "case_id": "sample", "split": "train", "answer": "3",
            "prompt": (
                "Initial registers: A=32, B=13\n"
                "1. ADD B A\n2. MUL B 5\n3. SUB A B\n4. ADD A 2\n"
            ),
        }
        self.assertEqual(
            verified_state_trace(record),
            "STATE 1: A=32, B=45\nSTATE 2: A=32, B=31\n"
            "STATE 3: A=1, B=31\nSTATE 4: A=3, B=31\nANSWER: 3",
        )

    def test_generator_consistency_and_final_label(self):
        for challenge in generate_program_challenges(192, 100, "train"):
            record = dict(case_id=challenge.id, split=challenge.split,
                          prompt=challenge.prompt, answer=challenge.answer)
            trace = verified_state_trace(record)
            self.assertTrue(trace.endswith(f"ANSWER: {challenge.answer}"))
            self.assertIn("STATE 1:", trace)
            self.assertEqual(trace.count("STATE "), challenge.prompt.count(". ADD ")
                             + challenge.prompt.count(". SUB ")
                             + challenge.prompt.count(". MUL "))

    def test_rejects_eval_and_wrong_gold(self):
        ch = generate_program_challenges(21, 1, "train")[0]
        record = dict(case_id=ch.id, split="dev", prompt=ch.prompt, answer=ch.answer)
        with self.assertRaisesRegex(ValueError, "train-only"):
            verified_state_trace(record)
        record["split"] = "train"
        record["answer"] = str((int(ch.answer) + 1) % 97)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            verified_state_trace(record)

    def test_rejects_ambiguous_state_and_instruction_count(self):
        base = dict(case_id="sample", split="train", answer="2",
                    prompt="Initial registers: A=1, B=0\n1. ADD A 1")
        self.assertEqual(verified_state_trace(base), "STATE 1: A=2, B=0\nANSWER: 2")
        for prompt in (
            "Initial registers: A=1, B=0\nInitial registers: A=1, B=0\n1. ADD A 1",
            "Initial registers: A=1, B=0\n2. ADD A 1",
            "Initial registers: A=99, B=0\n1. ADD A 1",
            "Initial registers: A=1, B=0\n1. ADD A 1\n2. DIV B 2",
        ):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                verified_state_trace({**base, "prompt": prompt})


if __name__ == "__main__":
    unittest.main()
