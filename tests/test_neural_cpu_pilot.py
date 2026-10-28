"""Contract tests for train-only evidence and immutable CPU pilot inputs."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from labyrinth.neural_cpu_pilot import (
    file_digest, read_training, supervised_target, training_prefix,
    render_supervised_tokens, training_visit_plan, zero_trainable_gradients,
)
from labyrinth.program_maze import (
    generate_graduated_train_challenges, generate_program_challenges,
)


class _CharTokenizer:
    eos_token = "</s>"
    chat_template = "fake-template-v1"

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("the SFT path must not duplicate BOS/EOS")
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=[ord(char) for char in text])

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize:
            raise AssertionError("need explicit, auditable loss mask boundary")
        result = "".join(f"<[{msg['role']}]>{msg['content']}</>" for msg in messages)
        if add_generation_prompt:
            result += "<[assistant]>"
        # Prefix formatting must exactly match the full assistant-start token.
        if len(messages) == 2:
            user, assistant = messages
            return f"<[user]>{user['content']}</><[assistant]>{assistant['content']}</>"
        return result


class _NoTemplate(_CharTokenizer):
    chat_template = None


class _BrokenBoundary(_CharTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if len(messages) == 2:
            return "NOT_THE_SAME_PREFIX" + messages[1]["content"]
        return super().apply_chat_template(messages, tokenize=tokenize,
                                           add_generation_prompt=add_generation_prompt)


class _PairTokenizer(_CharTokenizer):
    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("unexpected implicit special tokens")
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=[sum(ord(ch) for ch in text[i:i + 2])
                                          for i in range(0, len(text), 2)])


class _QuadTokenizer(_CharTokenizer):
    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("unexpected implicit special tokens")
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=[sum(ord(ch) for ch in text[i:i + 4])
                                          for i in range(0, len(text), 4)])


class NeuralCpuPilotTests(unittest.TestCase):
    def test_graduated_v2_attestation_and_trace_provenance(self):
        from labyrinth.neural_cpu_pilot import (
            attest_generated_train_only, training_visit_plan,
        )
        from labyrinth.state_traces import verified_state_trace

        cases = generate_graduated_train_challenges(2801, 128)
        records = [{"case_id": c.id, "split": c.split, "prompt": c.prompt,
                    "answer": c.answer} for c in cases]
        self.assertEqual(attest_generated_train_only(
            records, generator_seed=2801, generator_version="v2"), records)
        visited, exposure = training_visit_plan(records, 64)
        self.assertEqual(exposure["available_training_cases"], 128)
        self.assertEqual(exposure["case_visits"], 64)
        self.assertEqual(exposure["unique_training_cases"], 64)
        self.assertEqual(exposure["visited_case_ids"], [c["case_id"] for c in visited])
        self.assertEqual(visited, records[:64])
        self.assertTrue(all("-t1-" in c["case_id"] or "-t2-" in c["case_id"]
                            for c in visited))
        for record in (records[0], records[33], records[65], records[99]):
            self.assertTrue(verified_state_trace(record).endswith(
                "ANSWER: " + record["answer"]))
        for wrong in (
            {**records[0], "split": "dev"},
            {**records[0], "answer": str((int(records[0]["answer"]) + 1) % 97)},
        ):
            with self.assertRaises(ValueError):
                attest_generated_train_only([wrong], generator_seed=2801,
                                            generator_version="v2")
        with self.assertRaises(ValueError):
            attest_generated_train_only(records, generator_seed=2801,
                                        generator_version="v1")
        with self.assertRaises(ValueError):
            attest_generated_train_only(records, generator_seed=2802,
                                        generator_version="v2")
        with self.assertRaises(ValueError):
            attest_generated_train_only(records, generator_seed=2801,
                                        generator_version="v3")

    def test_graduated_script_generates_nonoverwriting_source_attested_fixture(self):
        from labyrinth.neural_cpu_pilot import attest_generated_train_only

        repo = Path(__file__).resolve().parents[1]
        output = Path(self.tmp.name) / "graduated.json"
        cmd = [sys.executable, str(repo / "scripts" / "generate_neural_train.py"),
               "--seed", "2801", "--count", "128", "--generator-version", "v2",
               "--output", str(output)]
        generated = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        metadata = json.loads(generated.stdout)
        records = read_training(output)
        self.assertEqual(metadata["generator_version"], "v2")
        self.assertEqual(metadata["sha256"], file_digest(output))
        self.assertEqual(len(records), 128)
        self.assertEqual(attest_generated_train_only(records, generator_seed=2801,
                                                    generator_version="v2"), records)
        refused = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(refused.returncode, 0)
        self.assertEqual(file_digest(output), metadata["sha256"])

    def test_committed_graduated_128_fixture_matches_reconstruction(self):
        from labyrinth.neural_cpu_pilot import attest_generated_train_only

        fixture = Path(__file__).resolve().parents[1] / "experiments" / "data" / "train_graduated_seed2801_128.json"
        self.assertEqual(file_digest(fixture),
                         "0e740105d92a21d3212f771f858f6a6cd518d5c8860fc04f555cc0fcaa17a643")
        cases = read_training(fixture)
        self.assertEqual(len(attest_generated_train_only(
            cases, generator_seed=2801, generator_version="v2")), 128)
        self.assertEqual([sum("-t" + str(t) + "-" in c["case_id"] for c in cases)
                          for t in range(1, 5)], [32, 32, 32, 32])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "train.json"
        self.sample = [{"case_id": "train-1", "split": "train", "prompt": "Compute 2+2",
                        "answer": "4"}]

    def write(self, records):
        self.path.write_text(json.dumps(records), encoding="utf-8")

    def test_training_receipt_counts_actual_visits_not_available_cases(self):
        # In the 32-step/128-example experiment only 32 examples can be seen.
        examples = [{"case_id": f"train-{i}"} for i in range(128)]
        visits, metrics = training_visit_plan(examples, 32)
        expected_ids = [f"train-{i}" for i in range(32)]
        self.assertEqual([item["case_id"] for item in visits], expected_ids)
        self.assertEqual(metrics["case_visits"], 32)
        self.assertEqual(metrics["available_training_cases"], 128)
        self.assertEqual(metrics["unique_cases_seen"], 32)
        self.assertEqual(metrics["unique_training_cases"], 32)
        self.assertEqual(metrics["visited_case_ids"], expected_ids)
        self.assertEqual(metrics["training_selection_policy"], "ordered-round-robin-v1")

    def test_training_receipt_tracks_wraparound_and_matched_schedule(self):
        examples = [{"case_id": f"train-{i}"} for i in range(3)]
        expected = ["train-0", "train-1", "train-2", "train-0", "train-1"]
        for _ in ("trained", "sham", "corrupted"):
            visits, metrics = training_visit_plan(examples, 5)
            self.assertEqual([item["case_id"] for item in visits], expected)
            self.assertEqual(metrics["visited_case_ids"], expected)
            self.assertEqual(metrics["unique_cases_seen"], 3)
            self.assertEqual(metrics["case_visits"], 5)
        with self.assertRaises(ValueError):
            training_visit_plan([], 1)
        with self.assertRaises(ValueError):
            training_visit_plan(examples, 0)

    def test_valid_train_only_data_and_digest(self):
        self.write(self.sample)
        self.assertEqual(read_training(self.path), self.sample)
        self.assertEqual(file_digest(self.path), hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_chat_sft_masks_entire_user_and_assistant_header(self):
        tok = _CharTokenizer()
        ids, prefix_length, metadata = render_supervised_tokens(
            tok, self.sample[0], policy="answer", arm="trained", step=0, mode="chat")
        transcript = "".join(map(chr, ids))
        self.assertEqual(transcript[:prefix_length],
                         "<[user]>Compute 2+2</><[assistant]>")
        self.assertEqual(transcript[prefix_length:], "ANSWER: 4</>")
        self.assertEqual(metadata["prompt_mode"], "chat")
        self.assertEqual(len(metadata["chat_template_sha256"]), 64)
        self.assertNotIn("ANSWER: 4", transcript[:prefix_length])
        self.assertGreater(len(ids) - prefix_length, len("ANSWER: 4"))

    def test_chat_corrupted_target_does_not_change_prompt_tokens(self):
        tok = _CharTokenizer()
        real, real_prefix, _ = render_supervised_tokens(
            tok, self.sample[0], policy="answer", arm="trained", step=2, mode="chat")
        bad, bad_prefix, _ = render_supervised_tokens(
            tok, self.sample[0], policy="answer", arm="corrupted", step=2, mode="chat")
        self.assertEqual(real_prefix, bad_prefix)
        self.assertEqual(real[:real_prefix], bad[:bad_prefix])
        self.assertNotEqual(real[real_prefix:], bad[bad_prefix:])

    def test_chat_trace_rewrites_user_instruction_before_assistant_states(self):
        ch = generate_program_challenges(1702, 1, "train")[0]
        sample = {"case_id": ch.id, "split": ch.split,
                  "prompt": ch.prompt, "answer": ch.answer}
        user_header = training_prefix(sample, policy="trace")
        assistant = supervised_target(sample, policy="trace", arm="trained", step=0)
        self.assertIn("STATE N: A=<integer>, B=<integer>", user_header)
        self.assertNotIn("Respond with exactly one line:", user_header)
        self.assertIn("STATE 1:", assistant)
        self.assertIn("ANSWER: " + ch.answer, assistant)
        self.assertNotIn("STATE 1:", user_header)
        ids, prefix_length, metadata = render_supervised_tokens(
            _QuadTokenizer(), sample, policy="trace", arm="trained", step=0, mode="chat")
        self.assertGreater(len(ids), prefix_length)
        self.assertEqual(metadata["prompt_mode"], "chat")

    def test_raw_sft_remains_byte_identical_to_legacy(self):
        tok = _CharTokenizer()
        ids, prefix_length, metadata = render_supervised_tokens(
            tok, self.sample[0], policy="answer", arm="trained", step=0, mode="raw")
        self.assertEqual("".join(map(chr, ids)), "Compute 2+2\nANSWER: 4</s>")
        self.assertEqual(prefix_length, len("Compute 2+2\n"))
        self.assertEqual(metadata["prompt_mode"], "raw")

    def test_chat_template_missing_or_nonprefix_rejected(self):
        for tok in (_NoTemplate(), _BrokenBoundary()):
            with self.subTest(tokenizer=type(tok).__name__), self.assertRaises(ValueError):
                render_supervised_tokens(tok, self.sample[0], policy="answer",
                                         arm="trained", step=0, mode="chat")

    def test_chat_loss_prefix_is_exact_inference_token_prefix_under_bpe_merges(self):
        tok = _PairTokenizer()
        ids, prefix_length, _ = render_supervised_tokens(
            tok, self.sample[0], policy="answer", arm="trained", step=0, mode="chat")
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": self.sample[0]["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        expected = tok(rendered, add_special_tokens=False).input_ids
        self.assertEqual(ids[:prefix_length], expected)
        self.assertGreater(len(ids), prefix_length)

    def test_compute_matched_sham_suppresses_gradients_not_backward_work(self):
        class _Grad:
            def __init__(self, value):
                self.value = value

            def zero_(self):
                self.value = 0

        class _Param:
            def __init__(self, gradient):
                self.grad = gradient

        live = _Grad(5)
        params = [_Param(live), _Param(None), _Param(_Grad(-2))]
        zero_trainable_gradients(params)
        self.assertEqual(params[0].grad.value, 0)
        self.assertIsNone(params[1].grad)
        self.assertEqual(params[2].grad.value, 0)

    def test_reject_eval_gold_and_duplicate_cases(self):
        for bad in (
            [{**self.sample[0], "split": "final"}],
            self.sample * 2,
            [{**self.sample[0], "secret": "leak"}],
            [{**self.sample[0], "answer": "ANSWER: 4"}],
            [{**self.sample[0], "answer": "-4"}],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.write(bad)
                read_training(self.path)

    def test_reject_empty_and_oversized_data(self):
        for bad in ([], [{}], self.sample * 257):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                self.write(bad)
                read_training(self.path)

    def test_generator_attestation_rejects_relabelled_holdout(self):
        from labyrinth.neural_cpu_pilot import attest_generated_train_only

        seed = 1807
        generated = generate_program_challenges(seed, 4, "train")
        valid = [
            {"case_id": c.id, "split": c.split, "prompt": c.prompt,
             "answer": c.answer}
            for c in generated
        ]
        attest_generated_train_only(valid, generator_seed=seed)
        # Self-reported split used to be the only barrier. Changing it is
        # insufficient evidence that this is a train-generator example.
        holdout = generate_program_challenges(seed, 1, "dev")[0]
        forged = {"case_id": holdout.id, "split": "train",
                  "prompt": holdout.prompt, "answer": holdout.answer}
        with self.assertRaisesRegex(ValueError, "generator"):
            attest_generated_train_only([forged], generator_seed=seed)
        with self.assertRaisesRegex(ValueError, "generator"):
            attest_generated_train_only([{**valid[0], "answer": "0"}], generator_seed=seed)
        with self.assertRaisesRegex(ValueError, "generator"):
            attest_generated_train_only(valid, generator_seed=seed + 1)
        with self.assertRaisesRegex(ValueError, "generator"):
            attest_generated_train_only(valid[::-1], generator_seed=seed)

    def test_existing_train_only_128_case_corpus_attests(self):
        from labyrinth.neural_cpu_pilot import attest_generated_train_only

        repo = Path(__file__).resolve().parents[1]
        for name, seed, count in (
            ("train_cpu_pilot_seed1701.json", 1701, 12),
            ("train_cpu_pilot_seed1702_32.json", 1702, 32),
            ("train_cpu_pilot_seed1807_128.json", 1807, 128),
        ):
            with self.subTest(fixture=name):
                fixture = repo / "experiments" / "data" / name
                self.assertEqual(
                    len(attest_generated_train_only(
                        read_training(fixture), generator_seed=seed)), count,
                )

    def test_trace_target_and_corrupted_labels_are_distinct(self):
        ch = generate_program_challenges(1702, 1, "train")[0]
        sample = {"case_id": ch.id, "split": ch.split,
                  "prompt": ch.prompt, "answer": ch.answer}
        answer = supervised_target(sample, policy="answer", arm="trained", step=0)
        trace = supervised_target(sample, policy="trace", arm="trained", step=0)
        corrupt = supervised_target(sample, policy="trace", arm="corrupted", step=0)
        self.assertEqual(answer, "ANSWER: " + ch.answer)
        self.assertIn("STATE 1:", trace)
        self.assertTrue(trace.endswith(answer))
        self.assertNotEqual(trace, corrupt)
        self.assertTrue(corrupt.endswith("ANSWER: " + str((int(ch.answer) + 1) % 97)))
        self.assertEqual(trace.count("STATE "), corrupt.count("STATE "))

    def test_invalid_policy_step_and_holdout_rejected(self):
        ch = generate_program_challenges(1702, 1, "train")[0]
        sample = {"case_id": ch.id, "split": "train", "prompt": ch.prompt,
                  "answer": ch.answer}
        for kwargs in ({"policy": "unknown", "arm": "trained", "step": 0},
                       {"policy": "trace", "arm": "other", "step": 0},
                       {"policy": "trace", "arm": "trained", "step": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                supervised_target(sample, **kwargs)
        with self.assertRaises(ValueError):
            supervised_target({**sample, "split": "final"}, policy="trace",
                              arm="trained", step=0)

    def test_trace_policy_rewrites_one_line_instruction_to_trace_contract(self):
        ch = generate_program_challenges(1702, 1, "train")[0]
        sample = {"case_id": ch.id, "split": ch.split,
                  "prompt": ch.prompt, "answer": ch.answer}
        self.assertEqual(training_prefix(sample, policy="answer"), ch.prompt + "\n")
        prompt = training_prefix(sample, policy="trace")
        self.assertIn("STATE N: A=<integer>, B=<integer>", prompt)
        self.assertNotIn("Respond with exactly one line:", prompt)
        self.assertIn("Initial registers:", prompt)
        self.assertNotIn(ch.answer, prompt.split("Then finish with exactly one line:")[-1])
        with self.assertRaises(ValueError):
            training_prefix({"prompt": "unexpected prompt"}, policy="trace")

    def test_repository_scripts_run_without_pythonpath_and_never_overwrite(self):
        repo = Path(__file__).resolve().parents[1]
        preflight = subprocess.run(
            [sys.executable, str(repo / "scripts" / "preflight_trace_prefix.py"), "--help"],
            cwd=self.tmp.name, capture_output=True, text=True,
        )
        self.assertEqual(preflight.returncode, 0, preflight.stderr)
        generator = repo / "scripts" / "generate_neural_train.py"
        output = Path(self.tmp.name) / "one.json"
        args = [sys.executable, str(generator), "--seed", "1901",
                "--count", "1", "--output", str(output)]
        first = subprocess.run(args, cwd=self.tmp.name, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        original = output.read_bytes()
        second = subprocess.run(args, cwd=self.tmp.name, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(output.read_bytes(), original)

    def test_train_cli_requires_generator_seed_before_any_neural_load(self):
        cmd = [sys.executable, "-m", "labyrinth.neural_cpu_pilot", "train",
               "--base", "missing", "--train-data", "missing", "--output", "missing",
               "--arm", "trained"]
        run = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("--train-generator-seed", run.stderr)


if __name__ == "__main__":
    unittest.main()
