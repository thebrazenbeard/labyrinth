"""Read-only runtime GGUF vs trainable HF W0 interoperability descriptor.

The same filename/model nickname does not establish conversion equivalence.
This checks physical bytes, manifest ancestry and a loopback-only *route*.
It never launches a model, GPU, trainer, converter or private evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from .campaign_full_state_v1 import verify_campaign_state
from .gguf_entrance_fingerprint_v1 import capture_gguf_entrance, verify_gguf_entrance
from .local_full_state_probe_v1 import LocalProbeTarget, _validate_target


SCHEMA = "labyrinth.dual-track-w0-entrance.v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


class DualTrackEntranceError(ValueError):
    """Runtime/trainable W0 identity or operator-supplied pin cannot be checked."""


def describe_dual_track_w0(
    *, runtime_gguf_path: Path,
    runtime_lora_gguf_path: Path | None = None,
    trainable_hf_manifest_path: Path | None = None,
    expected_gguf_manifest_sha256: str | None = None,
    expected_hf_state_sha256: str | None = None,
    local_target: LocalProbeTarget | None = None,
) -> dict[str, Any]:
    """Inspect physically separate inference and training entrances.

    Source-only with at most bounded GGUF parser memory and streaming hashes.
    An operator-supplied expected hash is a *comparison*, not an authenticated
    upstream publisher statement or a demonstrated conversion chain.
    """
    for value, label in ((expected_gguf_manifest_sha256, "GGUF manifest"),
                         (expected_hf_state_sha256, "HF state")):
        if value is not None and (type(value) is not str or _SHA.fullmatch(value) is None):
            raise DualTrackEntranceError(f"invalid {label} expected SHA-256")
    if (expected_hf_state_sha256 is not None and trainable_hf_manifest_path is None
            or local_target is not None and type(local_target) is not LocalProbeTarget):
        raise DualTrackEntranceError("expected HF pin requires HF W0; exact local route required")
    if local_target is not None:
        _validate_target(local_target)
        if local_target.subject_id != "parent":
            raise DualTrackEntranceError("runtime probe route must address W0 parent")
    runtime = capture_gguf_entrance(
        base_path=runtime_gguf_path, lora_adapter_path=runtime_lora_gguf_path,
    )
    if expected_gguf_manifest_sha256 is not None:
        verify_gguf_entrance(runtime,
                             trusted_manifest_sha256=expected_gguf_manifest_sha256)
    hf_state = None
    if trainable_hf_manifest_path is not None:
        if (not isinstance(trainable_hf_manifest_path, Path)
                or not trainable_hf_manifest_path.is_absolute()
                or trainable_hf_manifest_path.is_symlink()
                or trainable_hf_manifest_path != trainable_hf_manifest_path.resolve(strict=True)):
            raise DualTrackEntranceError("trainable HF W0 manifest must be a canonical file")
        evidence = verify_campaign_state(trainable_hf_manifest_path, fresh_process=True)
        manifest = json.loads(trainable_hf_manifest_path.read_bytes())
        if (manifest["generation_index"] != 0
                or manifest["origin_state_sha256"] != manifest["state_sha256"]
                or manifest["mutation_kind"] not in ("full_weight_edit", "lora_adapter")
                or manifest["state_sha256"] != evidence["state_sha256"]):
            raise DualTrackEntranceError("HF candidate is not an actual physical W0 ancestor")
        if (expected_hf_state_sha256 is not None
                and expected_hf_state_sha256 != evidence["state_sha256"]):
            raise DualTrackEntranceError("HF W0 differs from caller-supplied expected state hash")
        hf_state = {
            "manifest_path": str(trainable_hf_manifest_path),
            "state_sha256": evidence["state_sha256"],
            "manifest_sha256": evidence["manifest_sha256"],
            "family": manifest["family"],
            "mutation_kind": manifest["mutation_kind"],
            "generation_index": 0,
            "full_bytes_rehashed_in_fresh_process": True,
            "hf_state_pin_matches": expected_hf_state_sha256 is not None,
            "optimizer_rng_cursor_restoration_proven": False,
        }
    # A local endpoint is *configuration*, not evidence an actual model was
    # loaded. It is refused unless exact loopback syntax was validated above.
    route = (None if local_target is None else {
        "endpoint": local_target.endpoint,
        "backend": local_target.backend,
        "model_alias_claim": local_target.model_alias,
        "model_service_contacted": False,
        "loaded_gguf_sha256_proven": False,
        "public_development_probe_executed": False,
    })
    return {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_SEPARATE_RUNTIME_AND_HF_ENTRANCES",
        "gguf": {
            "base_path": runtime["base"]["path"],
            "base_sha256": runtime["base"]["sha256"],
            "base_bytes": runtime["base"]["bytes"],
            "format_version": runtime["base"]["format_version"],
            "general_architecture_claim": runtime["base"]["metadata"].get(
                "general.architecture", {}).get("value"),
            "file_type_claim": runtime["base"]["metadata"].get(
                "general.file_type", {}).get("value"),
            "lora_adapter_path": (runtime["lora_adapter"]["path"]
                                  if runtime["lora_adapter"] else None),
            "lora_adapter_sha256": (runtime["lora_adapter"]["sha256"]
                                    if runtime["lora_adapter"] else None),
            "pairing_status": runtime["pairing_status"],
            "manifest_sha256": runtime["manifest_sha256"],
            "caller_supplied_gguf_pin_matches": expected_gguf_manifest_sha256 is not None,
        },
        "hf_trainable_w0": hf_state,
        "local_only_probe_route": route,
        "runtime_equals_hf_weights_proven": False,
        "runtime_genuine_publisher_identity_proven": False,
        "gguf_quantization_transform_proven": False,
        "gguf_adapter_applied_proven": False,
        "gguf_itself_restorable_hf_training_state": False,
        "neural_improvement_proven": False,
        "graduation_authorized": False,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only dual-track GGUF/HF W0 inspection; no GPU or network effects")
    parser.add_argument("--runtime-gguf", type=Path, required=True)
    parser.add_argument("--lora-gguf", type=Path)
    parser.add_argument("--hf-full-state-manifest", type=Path)
    parser.add_argument("--expected-gguf-manifest-sha256")
    parser.add_argument("--expected-hf-state-sha256")
    parser.add_argument("--loopback-url")
    parser.add_argument("--backend", choices=("koboldcpp", "llamacpp", "openai_local"))
    args = parser.parse_args(argv)
    if (args.loopback_url is None) != (args.backend is None):
        parser.error("--loopback-url and --backend must be supplied together")
    target = (LocalProbeTarget("parent", args.loopback_url, args.backend)
              if args.loopback_url is not None else None)
    result = describe_dual_track_w0(
        runtime_gguf_path=args.runtime_gguf,
        runtime_lora_gguf_path=args.lora_gguf,
        trainable_hf_manifest_path=args.hf_full_state_manifest,
        expected_gguf_manifest_sha256=args.expected_gguf_manifest_sha256,
        expected_hf_state_sha256=args.expected_hf_state_sha256,
        local_target=target,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["DualTrackEntranceError", "describe_dual_track_w0"]
