"""Two *local* ToyLinearAdapter inference calls bound to R68 source observations.

This is a deliberately small real consumer of the Rezon/Labyrinth source
partition comparator, NOT Rezon distributed inference. Offered source records
and independently governed expected pins are caller inputs; this component
does not fetch Rezon receipts, authenticate owners, transfer grants, verify
consent/permissions, or authorize training, graduation, merge or deployment.

The two toy policies are invoked independently with one public arithmetic
prompt. No output aggregation, shared optimizer, or model-state sharing occurs.
The ToyLinearAdapter model hashes identify local parameter *content*, not a
host-owned model identity or a signed artifact. This code cannot defend against
hostile monkeypatches or model mutation in its own interpreter.
"""

from __future__ import annotations

from typing import NamedTuple

from .evidence_pointer import ExpectedSourcePinsV1
from .rezon_pointer_mapping_r57 import ExpectedRezonPinsV1, RezonSourceClaimsV1
from .rezon_two_adapter_source_partition_r68 import (
    AdapterSourceClaimV1, AdapterSourcePartitionError, ExpectedAdapterSourceV1,
    compare_two_model_adapter_sources,
)
from .toy_backend import ToyLinearAdapter, _PROMPT
from .training_scope import TrainingScopeSpec


QUALIFICATION = "LOCAL_TOY_PROBE_SOURCE_ONLY_UNATTESTED"


class ToyPairProbeV1(NamedTuple):
    """Independent responses and local content digests; no authority fields."""

    source_pair_sha256: str
    scope_sha256: str
    alpha_model_sha256: str
    beta_model_sha256: str
    alpha_response: str
    beta_response: str

    @property
    def qualification_state(self) -> str:
        return QUALIFICATION

    @property
    def model_identity_verified(self) -> bool:
        return False

    @property
    def host_execution_verified(self) -> bool:
        return False

    @property
    def consent_verified(self) -> bool:
        return False

    @property
    def permissions_verified(self) -> bool:
        return False

    @property
    def principal_verified(self) -> bool:
        return False

    @property
    def cross_model_grants_transferred(self) -> bool:
        return False

    @property
    def cross_model_identity_transferred(self) -> bool:
        return False

    @property
    def distributed_inference_verified(self) -> bool:
        return False

    @property
    def can_authorize_training(self) -> bool:
        return False

    @property
    def can_authorize_graduation(self) -> bool:
        return False

    @property
    def can_authorize_merge(self) -> bool:
        return False

    @property
    def can_authorize_deployment(self) -> bool:
        return False


def probe_two_toy_sources(
    *, scope: TrainingScopeSpec, prompt: str,
    alpha: ToyLinearAdapter, beta: ToyLinearAdapter,
    labyrinth_pointer_payload: bytes,
    expected_labyrinth: ExpectedSourcePinsV1,
    observed_rezon: RezonSourceClaimsV1,
    expected_rezon: ExpectedRezonPinsV1,
    observed_alpha: AdapterSourceClaimV1,
    expected_alpha: ExpectedAdapterSourceV1,
    observed_beta: AdapterSourceClaimV1,
    expected_beta: ExpectedAdapterSourceV1,
) -> ToyPairProbeV1:
    """Execute two isolated toy generations after strict source-content checks.

    Expected pins must be separately sourced from their owners: cloning the
    offered pins supplies no independent validation. This makes only a local,
    same-process source observation and cannot invoke any Rezon host worker.
    The declared scope must match the Labyrinth pointer's independently pinned
    scope digest, preventing a source pair replayed under another toy objective.
    """
    if type(scope) is not TrainingScopeSpec:
        raise AdapterSourcePartitionError("exact TrainingScopeSpec required")
    TrainingScopeSpec.__post_init__(scope)
    scope_hash = scope.fingerprint
    if type(expected_labyrinth) is not ExpectedSourcePinsV1:
        raise AdapterSourcePartitionError("exact Labyrinth expected source pins required")
    if scope_hash != expected_labyrinth.scope_sha256:
        raise AdapterSourcePartitionError("Labyrinth source pins do not match toy scope")
    if type(alpha) is not ToyLinearAdapter or type(beta) is not ToyLinearAdapter:
        raise AdapterSourcePartitionError("two exact ToyLinearAdapter instances required")
    if alpha is beta:
        raise AdapterSourcePartitionError("one toy runtime cannot act as two adapters")
    if type(prompt) is not str or _PROMPT.fullmatch(prompt) is None:
        raise AdapterSourcePartitionError("canonical toy arithmetic prompt required")
    if type(observed_alpha) is not AdapterSourceClaimV1 or type(observed_beta) is not AdapterSourceClaimV1:
        raise AdapterSourcePartitionError("exact offered adapter source pins required")
    alpha_hash = alpha.model_hash
    beta_hash = beta.model_hash
    if observed_alpha.model_artifact_sha256 != alpha_hash or observed_beta.model_artifact_sha256 != beta_hash:
        raise AdapterSourcePartitionError("offered toy artifact digests differ from live local parameters")

    source = compare_two_model_adapter_sources(
        labyrinth_pointer_payload=labyrinth_pointer_payload,
        expected_labyrinth=expected_labyrinth,
        observed_rezon=observed_rezon,
        expected_rezon=expected_rezon,
        observed_alpha=observed_alpha, expected_alpha=expected_alpha,
        observed_beta=observed_beta, expected_beta=expected_beta,
    )
    # Generation intentionally occurs *after* every source rejection. No case
    # labels, training objective instructions, identity or rights move between
    # the two local model instances.
    alpha_response = alpha.generate(prompt)
    beta_response = beta.generate(prompt)
    if alpha.model_hash != alpha_hash or beta.model_hash != beta_hash:
        raise AdapterSourcePartitionError("toy parameters changed during local inference")
    if scope.fingerprint != scope_hash or expected_labyrinth.scope_sha256 != scope_hash:
        raise AdapterSourcePartitionError("scope source pins changed during local inference")
    return ToyPairProbeV1(
        source_pair_sha256=source.source_pair_sha256,
        scope_sha256=scope_hash,
        alpha_model_sha256=alpha_hash,
        beta_model_sha256=beta_hash,
        alpha_response=alpha_response,
        beta_response=beta_response,
    )
