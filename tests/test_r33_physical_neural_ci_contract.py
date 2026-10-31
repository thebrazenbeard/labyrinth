"""R33 RED acceptance contract: pinned CPU CI must execute neural *physical* gates.

Source-only structural checks. These do not train a model; they prevent green
dependency-present CI from silently omitting the R22/R25 physical tests and
their opt-in CLIs. In particular, Python's general test job may legitimately
skip both physical modules because it does not install Torch.
"""

from pathlib import Path
import re
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "minotaur-neural-cpu.yml"
MODULES = {
    "tests.test_fullstate_physical_gate_r8": 1,
    "tests.test_optimizer_runner_v2_r19": 7,
    "tests.test_checkpoint_manifest_v2_r19": 6,
    "tests.test_v2_checkpoint_cpu_smoke_r22": 5,
    "tests.test_qualify_v2_weight_updates_cpu_r25": 18,
}
EXECUTABLES = (
    "scripts.qualify_v2_checkpoint_cpu_r22",
    "scripts.qualify_v2_weight_updates_cpu_r25",
)
PATH_FILTERS = tuple(
    "tests/" + module.removeprefix("tests.") + ".py"
    for module in MODULES
) + (
    "scripts/qualify_v2_checkpoint_cpu_r22.py",
    "scripts/qualify_v2_weight_updates_cpu_r25.py",
)


def neural_cpu_job_text() -> str:
    """Exclude PR path-filter labels, where a filename is not an invocation."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    marker = "jobs:\n  neural-cpu:"
    if workflow.count(marker) != 1:
        raise AssertionError("missing or ambiguous pinned neural-cpu job")
    return workflow.split(marker, 1)[1]


def required_cpu_test_block(job_text: str) -> str:
    """Restrict exact no-skip membership to the existing enforced CPU suite."""
    match = re.search(
        r"new_required\s*=\s*\((.*?)\)\s*for module_name, min_count in new_required:",
        job_text,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError("required CPU test inventory and enforcement missing")
    return match.group(1)


class R33PhysicalNeuralCiContractTests(TestCase):
    def test_both_physical_modules_are_required_under_torch_with_no_skips(self):
        job = neural_cpu_job_text()
        inventory = required_cpu_test_block(job)
        for module, count in MODULES.items():
            escaped = re.escape(module)
            match = re.search(r'\("' + escaped + r'",\s*(\d+)\)', inventory)
            self.assertIsNotNone(match, f"required physical test module missing: {module}")
            self.assertGreaterEqual(int(match.group(1)), count,
                                    f"insufficient physical test floor: {module}")

        enforcement = job.split("for module_name, min_count in new_required:", 1)[1]
        enforcement = enforcement.split("PY", 1)[0]
        self.assertRegex(enforcement, r"result\.testsRun\s*<\s*min_count")
        self.assertRegex(enforcement, r"not result\.wasSuccessful\(\)")
        self.assertRegex(enforcement, r"or result\.skipped")
        self.assertIn("raise SystemExit", enforcement)

    def test_both_event_path_filters_react_to_physical_gate_edits(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        expected = {
            "pull_request": r"(?ms)^  pull_request:\n    paths:\n(.*?)(?=^  push:)",
            "push": r"(?ms)^  push:\n    branches:.*\n    paths:\n(.*?)(?=^  workflow_dispatch:)",
        }
        for event, pattern in expected.items():
            match = re.search(pattern, workflow)
            self.assertIsNotNone(match, f"missing {event} path-filter block")
            patterns = re.findall(r'^\s+-\s+"([^"]+)"\s*$', match.group(1), re.MULTILINE)
            for path in PATH_FILTERS:
                self.assertIn(path, patterns,
                              f"{event} will not trigger on physical-gate edits: {path}")

    def test_both_physical_clis_execute_and_check_structured_success(self):
        job = neural_cpu_job_text()
        # A direct CLI without --execute is deliberately only a preflight.
        # The pinned job needs a check=True process boundary and must parse
        # and validate the JSON outcome, not merely observe an exit status.
        for module in EXECUTABLES:
            self.assertIn('"' + module + '"', job,
                          f"physical CLI invocation missing: {module}")
        self.assertRegex(job, r'"--execute"')
        self.assertRegex(job, r"subprocess\.run\(")
        self.assertRegex(job, r"check=True")
        self.assertRegex(job, r"json\.loads\(")
        for required_claim in (
            '"physical_cpu_toy_executed"',
            '"actual_trainable_tensor_updates_both_cycles"',
            '"frozen_base_weights_unchanged"',
            '"adamw_moments_advanced_both_cycles"',
            '"optimizer_fullstate_continuity_matches_uninterrupted_toy"',
            '"final_fresh_fullstate_restore_verified"',
            '"graduation_authorized"',
        ):
            self.assertIn(required_claim, job,
                          f"no asserted physical outcome field: {required_claim}")

    def test_dependency_present_neural_job_cannot_soft_fail(self):
        job = neural_cpu_job_text()
        self.assertIn('"torch==2.5.1+cpu"', job)
        self.assertIn('"safetensors==0.8.0"', job)
        self.assertNotRegex(job, r"(?m)^\s*continue-on-error:\s*true\s*$")
        self.assertNotRegex(job, r"(?m)^\s*[^#\n]*\|\|\s*true\s*$")
