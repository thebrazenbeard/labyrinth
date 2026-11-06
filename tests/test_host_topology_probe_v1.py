"""Pure fake-host probes; no Windows, NVIDIA, Torch or remote host access."""

import json
import unittest
from unittest.mock import patch

from labyrinth.host_topology_probe_v1 import (
    GIB, PACKAGES, GPUInventory, HostObservation,
    collect_local_observation, evaluate_observation, main,
    parse_nvidia_smi_csv,
)


def worker(**changes):
    facts = dict(
        role="lappy-worker", os_name="Windows", machine_arch="AMD64",
        python_version=(3, 12), cpu_count=8, ram_bytes=32 * GIB,
        free_disk_bytes=40 * GIB, root_exists=True, root_is_authoritative=True,
        package_versions=tuple((package, "1.2.3") for package in PACKAGES),
        gpu_status="observed", gpus=(GPUInventory(4096, 2800, "552.01"),),
        torch_cuda=True,
    )
    facts.update(changes)
    return HostObservation(**facts)


def states(report):
    return {check.check_id: check.state for check in report.checks}


class HostTopologyProbeTests(unittest.TestCase):
    def test_candidate_worker_resources_can_pass_but_never_authorize_deploy(self):
        report = evaluate_observation(worker())
        self.assertEqual(report.resource_gate, "RESOURCE_CHECKS_PASS_UNATTESTED")
        self.assertEqual(states(report)["host_identity"], "UNKNOWN")
        self.assertEqual(states(report)["controller_worker_transport"], "UNKNOWN")
        self.assertEqual(report.deployment_gate, "HARD_HOLD_NO_AUTHORIZED_DEPLOYMENT")
        self.assertFalse(report.physical_training_authorized)
        self.assertFalse(report.deployment_performed)
        self.assertFalse(report.host_identity_attested)
        self.assertFalse(report.transport_verified)
        self.assertEqual(report.evidence_class, "LOCAL_SELF_REPORTED_READ_ONLY_INVENTORY")
        json_data = json.loads(report.as_json())
        self.assertEqual(json_data["available_gpu_mib"], 2800)
        self.assertNotIn("hostname", json_data)
        self.assertNotIn("root_path", json_data)

    def test_actual_documented_cpu_only_torch_blocks_gpu_worker(self):
        report = evaluate_observation(worker(torch_cuda=False))
        self.assertEqual(report.resource_gate, "BLOCKED")
        self.assertEqual(states(report)["torch_cuda"], "FAIL")
        self.assertFalse(report.physical_training_authorized)

    def test_uninspected_gpu_cuda_are_unknown_not_success(self):
        observation = worker(gpu_status="uninspected", gpus=(), torch_cuda=None)
        report = evaluate_observation(observation)
        self.assertEqual(report.resource_gate, "INDETERMINATE")
        self.assertEqual(states(report)["gpu"], "UNKNOWN")
        self.assertEqual(states(report)["torch_cuda"], "UNKNOWN")

    def test_missing_nvidia_cli_torch_or_required_packages_fail(self):
        missing_package = tuple((p, None if p == "peft" else "1.0") for p in PACKAGES)
        report = evaluate_observation(worker(
            package_versions=missing_package, gpu_status="unavailable", gpus=(),
            torch_cuda=False,
        ))
        self.assertEqual(report.resource_gate, "BLOCKED")
        self.assertEqual(states(report)["gpu"], "FAIL")
        self.assertEqual(states(report)["package_peft"], "FAIL")
        self.assertEqual(states(report)["torch_cuda"], "FAIL")

    def test_wrong_os_root_disk_ram_and_gpu_budget_fail(self):
        report = evaluate_observation(worker(
            os_name="Linux", root_exists=False, root_is_authoritative=False,
            ram_bytes=4 * GIB, free_disk_bytes=1 * GIB,
            gpus=(GPUInventory(4096, 2000, "552.01"),),
        ))
        for key in ("worker_os", "authoritative_root", "ram", "free_disk", "gpu"):
            self.assertEqual(states(report)[key], "FAIL")
        self.assertEqual(report.resource_gate, "BLOCKED")

    def test_vn1_controller_does_not_require_gpu_or_training_packages(self):
        report = evaluate_observation(HostObservation(
            role="vn1-controller", os_name="Linux", machine_arch="x86_64",
            python_version=(3, 11), cpu_count=4, ram_bytes=4 * GIB,
            free_disk_bytes=8 * GIB, root_exists=True, root_is_authoritative=True,
            package_versions=tuple((package, None) for package in PACKAGES),
            gpu_status="uninspected", gpus=(), torch_cuda=None,
        ))
        self.assertEqual(report.resource_gate, "RESOURCE_CHECKS_PASS_UNATTESTED")
        self.assertEqual(states(report)["trainer_on_controller"], "PASS")
        self.assertNotIn("gpu", states(report))
        self.assertEqual(report.deployment_gate, "HARD_HOLD_NO_AUTHORIZED_DEPLOYMENT")
        self.assertFalse(report.host_identity_attested)

    def test_parser_accepts_bounded_multi_gpu_csv_and_rejects_malformed(self):
        parsed = parse_nvidia_smi_csv("4096, 2048, 552.01\n8192, 5000, 555.00\n")
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[1].memory_free_mib, 5000)
        for raw in (
            "", "4096, 4097, 552.01", "4096, -1, 552.01", "a, b, c",
            "4096,2048", "4096,2048,driver version", "1,2,3,4",
            "4096,2048,552.01\n" * 17, "4096,2048,552.01\x00",
        ):
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                parse_nvidia_smi_csv(raw)

    def test_mutated_frozen_facts_are_revalidated(self):
        facts = worker()
        object.__setattr__(facts, "gpus", (GPUInventory(4096, 2800, "552.01"), "bad"))
        with self.assertRaises(ValueError):
            evaluate_observation(facts)
        with self.assertRaises(ValueError):
            HostObservation(**{**vars(worker()), "role": "vn2-controller"})
        with self.assertRaises(ValueError):
            HostObservation(**{**vars(worker()), "package_versions": (("torch", "1.0"),)})

    def test_local_collection_is_read_only_and_no_gpu_by_default(self):
        with patch("labyrinth.host_topology_probe_v1.subprocess.run") as runner:
            with patch("labyrinth.host_topology_probe_v1.shutil.which") as which:
                observation = collect_local_observation("vn1-controller", ".")
        self.assertEqual(observation.gpu_status, "uninspected")
        self.assertIsNone(observation.torch_cuda)
        runner.assert_not_called()
        which.assert_not_called()
        report = evaluate_observation(observation)
        self.assertFalse(report.physical_training_authorized)

    def test_cli_reports_hard_hold_with_nonzero_status(self):
        with patch("labyrinth.host_topology_probe_v1.collect_local_observation", return_value=worker()):
            with patch("builtins.print") as emit:
                exit_code = main(["--role", "lappy-worker", "--root", "D:\\VERA"])
        self.assertEqual(exit_code, 2)
        data = json.loads(emit.call_args.args[0])
        self.assertEqual(data["deployment_gate"], "HARD_HOLD_NO_AUTHORIZED_DEPLOYMENT")
        self.assertFalse(data["deployment_performed"])


if __name__ == "__main__":
    unittest.main()
