"""Read-only, nonauthorizing local resource probe for VN1/Lappy Labyrinth roles.

No network, model load, trainer launch, deployment, installation or restart.
All observations are untrusted self-reports; the physical trainer is on hold.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from .trainingday_adapter_v1 import TRAINER_ENTRYPOINT, TRAINER_HEAD


SCHEMA = "labyrinth.host-topology-probe.v1"
ROLES = ("vn1-controller", "lappy-worker")
PACKAGES = ("torch", "transformers", "peft", "accelerate")
GIB = 1024 ** 3
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}\Z", re.ASCII)


@dataclass(frozen=True)
class GPUInventory:
    memory_total_mib: int
    memory_free_mib: int
    driver_version: str

    def __post_init__(self) -> None:
        if (type(self.memory_total_mib) is not int
                or not 0 < self.memory_total_mib <= 2**30
                or type(self.memory_free_mib) is not int
                or not 0 <= self.memory_free_mib <= self.memory_total_mib
                or type(self.driver_version) is not str
                or _VERSION.fullmatch(self.driver_version) is None):
            raise ValueError("invalid bounded GPU inventory")


@dataclass(frozen=True)
class HostObservation:
    """Locally collected or test-injected facts; NOT authenticated host identity."""

    role: str
    os_name: str
    machine_arch: str
    python_version: tuple[int, int]
    cpu_count: int | None
    ram_bytes: int | None
    free_disk_bytes: int | None
    root_exists: bool
    root_is_authoritative: bool
    package_versions: tuple[tuple[str, str | None], ...]
    gpu_status: str  # observed, unavailable, uninspected, query_failed
    gpus: tuple[GPUInventory, ...]
    torch_cuda: bool | None

    def __post_init__(self) -> None:
        if type(self.role) is not str or self.role not in ROLES:
            raise ValueError("unsupported host role")
        if (type(self.os_name) is not str or not 1 <= len(self.os_name) <= 32
                or type(self.machine_arch) is not str or not 1 <= len(self.machine_arch) <= 64):
            raise ValueError("invalid OS inventory")
        if (type(self.python_version) is not tuple
                or len(self.python_version) != 2
                or any(type(v) is not int or not 0 <= v <= 99 for v in self.python_version)):
            raise ValueError("invalid Python version")
        for name in ("cpu_count", "ram_bytes", "free_disk_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not 0 <= value <= 2**63 - 1):
                raise ValueError(f"invalid {name}")
        if type(self.root_exists) is not bool or type(self.root_is_authoritative) is not bool:
            raise ValueError("invalid root-path observations")
        if (type(self.package_versions) is not tuple
                or tuple(name for name, _ in self.package_versions) != PACKAGES
                or any(version is not None and (
                    type(version) is not str or _VERSION.fullmatch(version) is None)
                    for _, version in self.package_versions)):
            raise ValueError("invalid package inventory")
        if self.gpu_status not in ("observed", "unavailable", "uninspected", "query_failed"):
            raise ValueError("invalid GPU probe state")
        if (type(self.gpus) is not tuple or len(self.gpus) > 16
                or any(type(gpu) is not GPUInventory for gpu in self.gpus)):
            raise ValueError("invalid GPU roster")
        for gpu in self.gpus:
            GPUInventory.__post_init__(gpu)
        if (self.gpu_status == "observed") != bool(self.gpus):
            raise ValueError("GPU probe state and roster disagree")
        if self.torch_cuda is not None and type(self.torch_cuda) is not bool:
            raise ValueError("invalid PyTorch CUDA observation")


@dataclass(frozen=True)
class ReadinessCheck:
    check_id: str
    state: str  # PASS, FAIL, UNKNOWN
    explanation: str


@dataclass(frozen=True)
class HostReadiness:
    schema: str
    role: str
    observed_os: str
    observed_arch: str
    python_version: str
    cpu_count: int | None
    ram_gib: float | None
    free_disk_gib: float | None
    gpu_count: int | None
    available_gpu_mib: int | None
    package_versions: tuple[tuple[str, str | None], ...]
    checks: tuple[ReadinessCheck, ...]
    resource_gate: str
    deployment_gate: str = "HARD_HOLD_NO_AUTHORIZED_DEPLOYMENT"
    source_contract: str = "TRAININGDAY_SOURCE_ONLY_HARD_HOLD"
    trainingday_head: str = TRAINER_HEAD
    trainingday_entrypoint: str = TRAINER_ENTRYPOINT
    physical_training_authorized: bool = False
    deployment_performed: bool = False
    host_identity_attested: bool = False
    transport_verified: bool = False
    evidence_class: str = "LOCAL_SELF_REPORTED_READ_ONLY_INVENTORY"

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def parse_nvidia_smi_csv(raw: str) -> tuple[GPUInventory, ...]:
    """Parse only `memory.total,memory.free,driver_version` CSV, MiB units."""
    if type(raw) is not str or not 1 <= len(raw) <= 4096 or "\x00" in raw:
        raise ValueError("invalid bounded nvidia-smi response")
    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    if not 1 <= len(rows) <= 16:
        raise ValueError("invalid number of GPUs")
    inventory = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 3 or not all(parts):
            raise ValueError("malformed nvidia-smi CSV")
        try:
            total, free = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError("invalid GPU memory reading") from exc
        inventory.append(GPUInventory(total, free, parts[2]))
    return tuple(inventory)


def _ram_bytes() -> int | None:
    """Bounded local OS memory inventory; no optional psutil dependency."""
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        mem = MemoryStatus()
        mem.dwLength = ctypes.sizeof(MemoryStatus)
        try:
            return int(mem.ullTotalPhys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(mem)
            ) else None
        except (AttributeError, OSError):
            return None
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _package_versions() -> tuple[tuple[str, str | None], ...]:
    versions = []
    for package in PACKAGES:
        try:
            value = metadata.version(package)
        except metadata.PackageNotFoundError:
            value = None
        # Unknown metadata cannot be promoted to a valid pinned package.
        versions.append((package, value if value and _VERSION.fullmatch(value) else None))
    return tuple(versions)


def collect_local_observation(
    role: str, root: str | Path, *, inspect_gpu: bool = False,
    inspect_torch: bool = False,
) -> HostObservation:
    """Explicit local only. Optional NVIDIA/Torch reads never launch training."""
    if type(role) is not str or role not in ROLES:
        raise ValueError("unsupported role")
    if not isinstance(root, (str, Path)) or not str(root):
        raise ValueError("explicit local root required")
    if type(inspect_gpu) is not bool or type(inspect_torch) is not bool:
        raise ValueError("inspect flags must be boolean")
    path = Path(root)
    root_exists = path.is_dir() and not path.is_symlink()
    try:
        free_disk = shutil.disk_usage(path).free if root_exists else None
    except OSError:
        free_disk = None
    # Operator's Lappy root is D:\VERA, but this self-report proves no host ID.
    authoritative = (
        role == "lappy-worker" and platform.system() == "Windows"
        and str(path).rstrip("\\/").casefold() == "d:\\vera"
    ) or (role == "vn1-controller" and root_exists)

    status = "uninspected"
    gpus: tuple[GPUInventory, ...] = ()
    if inspect_gpu:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            status = "unavailable"
        else:
            try:
                process = subprocess.run(
                    [executable,
                     "--query-gpu=memory.total,memory.free,driver_version",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                if process.returncode == 0:
                    gpus = parse_nvidia_smi_csv(process.stdout)
                    status = "observed"
                else:
                    status = "query_failed"
            except (OSError, subprocess.TimeoutExpired, UnicodeError, ValueError):
                status = "query_failed"

    torch_cuda: bool | None = None
    if inspect_torch:
        try:
            import torch  # Read-only opt-in runtime query; no model load.
            torch_cuda = bool(torch.cuda.is_available())
        except (ImportError, OSError, RuntimeError):
            torch_cuda = False
    return HostObservation(
        role=role, os_name=platform.system(), machine_arch=platform.machine(),
        python_version=(sys.version_info.major, sys.version_info.minor),
        cpu_count=os.cpu_count(), ram_bytes=_ram_bytes(),
        free_disk_bytes=free_disk, root_exists=root_exists,
        root_is_authoritative=authoritative, package_versions=_package_versions(),
        gpu_status=status, gpus=gpus, torch_cuda=torch_cuda,
    )


def evaluate_observation(observation: HostObservation) -> HostReadiness:
    """Pure gate; cannot authorize training even when resources appear ample."""
    if type(observation) is not HostObservation:
        raise ValueError("exact host observation required")
    HostObservation.__post_init__(observation)
    checks: list[ReadinessCheck] = []
    def add(key: str, state: str, explanation: str) -> None:
        checks.append(ReadinessCheck(key, state, explanation))
    is_worker = observation.role == "lappy-worker"
    add("host_identity", "UNKNOWN", "Role is an operator claim; no authenticated host identity.")
    add("controller_worker_transport", "UNKNOWN",
        "No authorized authenticated remote handshake was performed.")
    if is_worker:
        add("worker_os", "PASS" if observation.os_name == "Windows" else "FAIL",
            "Lappy worker must report Windows for this target topology.")
    else:
        add("controller_os", "PASS" if observation.os_name in ("Windows", "Linux") else "FAIL",
            "VN1 controller candidate requires supported Windows or Linux Python.")
    add("authoritative_root", "PASS" if observation.root_exists and observation.root_is_authoritative else "FAIL",
        "Lappy requires D:\\VERA; VN1 requires an explicit existing local root.")
    add("python", "PASS" if observation.python_version >= (3, 10) else "FAIL",
        "Local Python 3.10 or newer required for a source candidate.")
    for name, value, minimum in (
        ("cpu", observation.cpu_count, 2),
        ("ram", observation.ram_bytes, (8 if is_worker else 2) * GIB),
        ("free_disk", observation.free_disk_bytes, (10 if is_worker else 2) * GIB),
    ):
        state = "UNKNOWN" if value is None else "PASS" if value >= minimum else "FAIL"
        add(name, state, f"Illustrative resource floor: {minimum} {'bytes' if name != 'cpu' else 'cores'}.")
    versions = dict(observation.package_versions)
    if is_worker:
        for package in PACKAGES:
            add(f"package_{package}", "PASS" if versions[package] else "FAIL",
                f"{package} distribution metadata required; version pin unverified.")
        if observation.gpu_status == "uninspected":
            add("gpu", "UNKNOWN", "NVIDIA GPU was not queried; use explicit --inspect-gpu.")
        elif observation.gpu_status != "observed":
            add("gpu", "FAIL", "NVIDIA tool absent, failed, or returned malformed inventory.")
        else:
            sufficient = any(gpu.memory_total_mib >= 4096 and gpu.memory_free_mib >= 2048
                             for gpu in observation.gpus)
            add("gpu", "PASS" if sufficient else "FAIL",
                "Illustrative worker floor: 4 GiB total and 2 GiB free VRAM on one GPU.")
        add("torch_cuda", "UNKNOWN" if observation.torch_cuda is None
            else "PASS" if observation.torch_cuda else "FAIL",
            "PyTorch CUDA backend must be observed explicitly; metadata alone is insufficient.")
    else:
        add("trainer_on_controller", "PASS", "Controller role does not require local CUDA/trainer.")
    resource_checks = tuple(c for c in checks if c.check_id not in (
        "host_identity", "controller_worker_transport"))
    resource_gate = ("BLOCKED" if any(c.state == "FAIL" for c in resource_checks)
                     else "INDETERMINATE" if any(c.state == "UNKNOWN" for c in resource_checks)
                     else "RESOURCE_CHECKS_PASS_UNATTESTED")
    best_gpu = max((gpu.memory_free_mib for gpu in observation.gpus), default=None)
    return HostReadiness(
        schema=SCHEMA, role=observation.role, observed_os=observation.os_name,
        observed_arch=observation.machine_arch,
        python_version=f"{observation.python_version[0]}.{observation.python_version[1]}",
        cpu_count=observation.cpu_count,
        ram_gib=round(observation.ram_bytes / GIB, 2) if observation.ram_bytes is not None else None,
        free_disk_gib=round(observation.free_disk_bytes / GIB, 2)
        if observation.free_disk_bytes is not None else None,
        gpu_count=len(observation.gpus) if observation.gpu_status == "observed" else None,
        available_gpu_mib=best_gpu, package_versions=observation.package_versions,
        checks=tuple(checks), resource_gate=resource_gate,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Non-deploying VN1/Lappy local readiness inventory")
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--root", required=True,
                        help="Existing local root for disk inventory (Lappy: D:\\VERA)")
    parser.add_argument("--inspect-gpu", action="store_true",
                        help="Opt-in read-only nvidia-smi memory query (5s timeout)")
    parser.add_argument("--inspect-torch", action="store_true",
                        help="Opt-in import torch and query CUDA availability")
    args = parser.parse_args(argv)
    report = evaluate_observation(collect_local_observation(
        args.role, args.root, inspect_gpu=args.inspect_gpu,
        inspect_torch=args.inspect_torch,
    ))
    print(report.as_json())
    # This source adapter CANNOT grant deployment/physical TRAIN authority.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA", "ROLES", "PACKAGES", "GPUInventory", "HostObservation",
    "ReadinessCheck", "HostReadiness", "parse_nvidia_smi_csv",
    "collect_local_observation", "evaluate_observation", "main",
]
