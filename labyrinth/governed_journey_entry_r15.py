"""Opt-in provider-bound entry to a single full-state public TRAIN Journey.

This is a *real source call path* joining R13's provider/host reservation
adapter to R12's full-state runner. There is no local authority shortcut: the
caller cannot substitute the three reservation/signature/binding callbacks.

The host service, external monotonic anchor, provider signer, event verifier,
and checkpoint attester still require separately verified operator custody.
Python object construction and green synthetic tests cannot establish that
custody, physical pretrained-model execution, semantic retention, a protected
exam, or release permission. All return receipts remain SOURCE_ONLY.
"""

from __future__ import annotations

from hashlib import sha256
import hmac
from pathlib import Path
from typing import Any, Callable

from .fullstate_successor import DevelopmentStepReceipt
from .hf_development_event_journal import FullStateRunBinding
from .host_authority_reservation_r12 import HostProviderReservationService, ProviderBinding
from .journey import JourneyState
from .journey_execution_r12 import (
    SignedJourneyExecutionReceipt, execute_signed_fullstate_journey,
)
from .measured_weakness_feedback import TrainingExposureRef
from .objective_contract import ObjectiveContract
from .provider_journey_adapter_r13 import ProviderJourneyAdapter, ProviderJourneyPins
from .signed_exam_lineage import SignedExamLineageBinding


class GovernedJourneyEntryError(ValueError):
    """The exact provider, learner, or run identity failed pre-reservation."""


def execute_provider_bound_fullstate_journey(
    *,
    provider_adapter: ProviderJourneyAdapter,
    contract: ObjectiveContract,
    journey: JourneyState,
    binding: SignedExamLineageBinding,
    training_exposures: tuple[TrainingExposureRef, ...],
    expected_training_roster_sha256: str,
    expected_objective_contract_sha256: str,
    expected_journey_sha256: str,
    expected_learner_checkpoint_sha256: str,
    expected_fence: int,
    verification_inputs: dict[str, Any],
    fullstate_binding: FullStateRunBinding,
    expected_source_tree_sha256: str,
    journal_db: Path,
    journal_state_key: bytes,
    initial_checkpoint_dir: Path,
    successor_checkpoint_dir: Path,
    model_factory: Callable[[], Any],
    optimizer_factory: Callable[[Any], Any],
    training_step: Callable[[Any, Any, tuple[tuple[str, str], ...], int], DevelopmentStepReceipt],
    verify_signed_event: Callable[[str, FullStateRunBinding], bool],
    attest_successor_bytes: Callable[[Path, Any], bool],
) -> SignedJourneyExecutionReceipt:
    """Run one pinned development event with no overridable host authority.

    Caller-owned training hooks must not be interpreted as host independence.
    Failure after remote reservation is irreversible: no automatic retries.
    """
    if (type(provider_adapter) is not ProviderJourneyAdapter
            or type(provider_adapter.host) is not HostProviderReservationService
            or type(provider_adapter.binding) is not ProviderBinding
            or type(provider_adapter.pins) is not ProviderJourneyPins
            or provider_adapter.host.binding != provider_adapter.binding):
        raise GovernedJourneyEntryError("real pinned provider/host adapter required")
    pins = provider_adapter.pins
    pins.validate()
    if (not hmac.compare_digest(provider_adapter.binding.sha256,
                                pins.authority_binding_sha256)
            or type(fullstate_binding) is not FullStateRunBinding
            or type(binding) is not SignedExamLineageBinding
            or type(contract) is not ObjectiveContract
            or type(journey) is not JourneyState):
        raise GovernedJourneyEntryError("provider or signed run identity untrusted")
    fullstate_binding.validate()
    if (fullstate_binding.run_id != pins.expected_run_id
            or fullstate_binding.objective_id != pins.expected_objective_id
            or not hmac.compare_digest(fullstate_binding.commitment_sha256(),
                                       pins.fullstate_binding_sha256)
            or expected_source_tree_sha256 != pins.expected_source_tree_sha256
            or expected_training_roster_sha256 != pins.training_roster_sha256
            or expected_objective_contract_sha256 != pins.contract_sha256
            or expected_journey_sha256 != pins.journey_sha256
            or expected_learner_checkpoint_sha256 != pins.learner_checkpoint_sha256
            or expected_fence != pins.expected_host_fence
            or not hmac.compare_digest(sha256(binding.binding_bytes).hexdigest(),
                                       pins.signed_binding_sha256)):
        raise GovernedJourneyEntryError(
            "source, provider, TRAIN roster, full-state or host fence differs from pin"
        )

    # The only reservation path is the validated provider adapter; injection
    # of a caller-owned host callback is intentionally not in this signature.
    return execute_signed_fullstate_journey(
        contract=contract,
        journey=journey,
        binding=binding,
        training_exposures=training_exposures,
        expected_training_roster_sha256=expected_training_roster_sha256,
        expected_objective_contract_sha256=expected_objective_contract_sha256,
        expected_journey_sha256=expected_journey_sha256,
        expected_learner_checkpoint_sha256=expected_learner_checkpoint_sha256,
        expected_fence=expected_fence,
        host_reserve_once=provider_adapter.host_reserve_once,
        verify_host_reservation=provider_adapter.verify_host_reservation,
        verification_inputs=verification_inputs,
        fullstate_binding=fullstate_binding,
        expected_source_tree_sha256=expected_source_tree_sha256,
        verify_host_binding=provider_adapter.verify_host_binding,
        journal_db=journal_db,
        journal_state_key=journal_state_key,
        initial_checkpoint_dir=initial_checkpoint_dir,
        successor_checkpoint_dir=successor_checkpoint_dir,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        training_step=training_step,
        verify_signed_event=verify_signed_event,
        attest_successor_bytes=attest_successor_bytes,
    )
