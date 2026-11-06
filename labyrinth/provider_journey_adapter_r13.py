"""Versioned provider/host reservation adapter for the signed Journey path.

This opt-in adapter joins R12's host-reservation *candidate* to the exact
``prepare_signed_journey_loop`` callback protocol. It consumes a verified
development TRAIN event once; it does not operate the host or independently
attest a signing key, external monotonic authority, model, or final exam.

The administrator must pin the immutable record, configure the external epoch
CAS, and keep all service keys outside the learner. A callback injected by an
untrusted caller is not host independence. The separate full-state journal
cannot be made atomic with the reservation database by this source adapter.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import re
from typing import Callable

from .hf_development_event_journal import FullStateRunBinding
from .host_authority_reservation_r12 import (
    HostAuthorityError, HostProviderReservationService, ProviderBinding,
    ProviderRequest,
)
from .signed_journey_loop import HostLoopReservation, SignedJourneyLoopPlan


SCHEMA = "labyrinth.provider-journey-adapter.v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_FIELDS = {
    "schema", "event_id", "exam_id", "scope_sha256", "contract_sha256",
    "journey_sha256", "learner_checkpoint_sha256", "signed_binding_sha256",
    "training_roster_sha256", "prior_cycle", "regenerated_cycle",
    "expected_host_fence", "weak_skills", "rooms", "answerless_train_pairs",
}


def _is_sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _is_id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HostAuthorityError("duplicate signed TRAIN proposal field")
        result[key] = value
    return result


def training_allocation_sha256(rooms: list[dict[str, object]],
                               views: list[dict[str, object]]) -> str:
    """Domain-separated commitment to the next cycle's TRAIN-only allocation.

    Unlike ``training_roster_sha256``, which commits to *previous* exposures,
    this operator-pinned hash covers newly generated rooms and learner views.
    Callers must compute/pin it from independently trusted allocation inputs.
    """
    body = json.dumps({"rooms": rooms, "answerless_train_pairs": views},
                      sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")
    return sha256(b"labyrinth.train-allocation.v1\n" + body).hexdigest()


@dataclass(frozen=True)
class ProviderJourneyPins:
    """Trusted operator-provided identities, not learner-supplied claims."""

    authority_binding_sha256: str
    event_id: str
    exam_id: str
    scope_sha256: str
    contract_sha256: str
    journey_sha256: str
    learner_checkpoint_sha256: str
    signed_binding_sha256: str
    training_roster_sha256: str
    expected_train_allocation_sha256: str
    previous_cycle: int
    next_cycle: int
    expected_host_fence: int
    weak_skills: tuple[str, ...]
    expected_run_id: str
    expected_objective_id: str
    expected_source_tree_sha256: str
    fullstate_binding_sha256: str

    def validate(self) -> None:
        if (not all(_is_id(v) for v in (
                self.event_id, self.exam_id, self.expected_run_id,
                self.expected_objective_id))
                or not all(_is_sha(v) for v in (
                    self.authority_binding_sha256, self.scope_sha256,
                    self.contract_sha256, self.journey_sha256,
                    self.learner_checkpoint_sha256, self.signed_binding_sha256,
                    self.training_roster_sha256, self.expected_train_allocation_sha256,
                    self.expected_source_tree_sha256,
                    self.fullstate_binding_sha256))
                or type(self.previous_cycle) is not int
                or self.previous_cycle < 0
                or type(self.next_cycle) is not int
                or self.next_cycle != self.previous_cycle + 1
                or type(self.expected_host_fence) is not int
                or not 1 <= self.expected_host_fence <= 4096
                or type(self.weak_skills) is not tuple
                or not self.weak_skills
                or any(not _is_id(skill) for skill in self.weak_skills)
                or len(set(self.weak_skills)) != len(self.weak_skills)):
            raise HostAuthorityError("malformed externally pinned Journey identity")


class ProviderJourneyAdapter:
    """Callback provider restricted to one pinned signed Journey transition."""

    def __init__(
        self,
        host: HostProviderReservationService,
        binding: ProviderBinding,
        pins: ProviderJourneyPins,
        sign_provider_request: Callable[
            [ProviderBinding, str, bytes, int], ProviderRequest],
    ) -> None:
        if (type(host) is not HostProviderReservationService
                or type(binding) is not ProviderBinding
                or type(pins) is not ProviderJourneyPins
                or not callable(sign_provider_request)):
            raise HostAuthorityError("real typed provider/host reservation objects required")
        binding.validate()
        pins.validate()
        if (host.binding != binding
                or not hmac.compare_digest(binding.sha256,
                                           pins.authority_binding_sha256)):
            raise HostAuthorityError("host/provider/policy binding differs from pinned identity")
        self.host = host
        self.binding = binding
        self.pins = pins
        self._sign_request = sign_provider_request
        # Keep an independently valued construction-time authority snapshot.
        # Public Python attributes can be rebound after __init__, even when
        # their dataclass values were originally frozen and validated.
        self._initial_host = host
        self._initial_signer = sign_provider_request
        self._initial_binding = deepcopy(binding)
        self._initial_pins = deepcopy(pins)

    def _require_original_authority(self) -> None:
        if (self.host is not self._initial_host
                or self._sign_request is not self._initial_signer
                or self.binding != self._initial_binding
                or self.pins != self._initial_pins
                or self.host.binding != self._initial_binding):
            raise HostAuthorityError("admitted provider/host/pins changed after construction")

    def _read_pinned_proposal(self, raw: bytes) -> dict[str, object]:
        self._require_original_authority()
        if type(raw) is not bytes or not 0 < len(raw) <= 1_000_000:
            raise HostAuthorityError("bounded canonical TRAIN proposal required")
        try:
            record = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_pairs,
                                parse_constant=lambda _: (_ for _ in ()).throw(
                                    HostAuthorityError("nonfinite TRAIN proposal")))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise HostAuthorityError("invalid signed TRAIN proposal JSON") from exc
        if (type(record) is not dict or record.keys() != _FIELDS
                or (json.dumps(record, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False) + "\n").encode() != raw):
            raise HostAuthorityError("noncanonical or incomplete signed TRAIN proposal")
        p = self.pins
        expected = {
            "schema": "labyrinth.signed-journey-proposal.v1",
            "event_id": p.event_id,
            "exam_id": p.exam_id,
            "scope_sha256": p.scope_sha256,
            "contract_sha256": p.contract_sha256,
            "journey_sha256": p.journey_sha256,
            "learner_checkpoint_sha256": p.learner_checkpoint_sha256,
            "signed_binding_sha256": p.signed_binding_sha256,
            "training_roster_sha256": p.training_roster_sha256,
            "prior_cycle": p.previous_cycle,
            "regenerated_cycle": p.next_cycle,
            "expected_host_fence": p.expected_host_fence,
            "weak_skills": list(p.weak_skills),
        }
        if any(type(record[key]) is not type(value) or record[key] != value
               for key, value in expected.items()):
            raise HostAuthorityError("signed event/objective/learner or host fence differs from pin")
        rooms, views = record["rooms"], record["answerless_train_pairs"]
        if (type(rooms) is not list or type(views) is not list
                or len(rooms) != len(views) or not rooms):
            raise HostAuthorityError("TRAIN rooms and learner views lack paired coverage")
        # A valid provider signature is not sufficient if its own payload
        # routes an approved weakness into a different unapproved task family.
        for room, view in zip(rooms, views, strict=True):
            if (type(room) is not dict or type(view) is not dict
                    or room.keys() != {"room_id", "skill", "case_seed"}
                    or view.keys() != {"challenge_id", "prompt_sha256", "skill"}
                    or type(room["skill"]) is not str
                    or type(view["skill"]) is not str
                    or room["skill"] not in p.weak_skills
                    or view["skill"] != room["skill"]):
                raise HostAuthorityError("signed TRAIN view escaped pinned weakness skill")
        # The prior TRAIN roster digest cannot authenticate the future rooms.
        # Bind the full next-cycle allocation to a distinct operator-owned pin
        # before spending any remote one-shot host reservation.
        if not hmac.compare_digest(training_allocation_sha256(rooms, views),
                                   p.expected_train_allocation_sha256):
            raise HostAuthorityError("signed TRAIN allocation differs from external pin")
        return record

    def host_reserve_once(self, proposal_bytes: bytes) -> HostLoopReservation:
        """Run signed provider admission and exactly one remote-fenced CAS."""
        record = self._read_pinned_proposal(proposal_bytes)
        requested_epoch = self.pins.expected_host_fence - 1
        request = self._sign_request(
            self.binding, record["event_id"], proposal_bytes, requested_epoch,
        )
        expected_sha = sha256(proposal_bytes).hexdigest()
        if (type(request) is not ProviderRequest
                or request.authority_id != self.binding.authority_id
                or request.provider_id != self.binding.provider_id
                or request.binding_sha256 != self.binding.sha256
                or request.event_id != self.pins.event_id
                or request.proposal_sha256 != expected_sha
                or request.expected_epoch != requested_epoch):
            raise HostAuthorityError("provider signer returned foreign or mutable authority request")
        self._require_original_authority()
        receipt = self.host.reserve(request, proposal_bytes)
        if not self.verify_host_reservation(receipt, receipt.signed_bytes):
            raise HostAuthorityError("committed host reservation failed independent readback")
        return receipt

    def verify_host_reservation(self, receipt: HostLoopReservation,
                                payload: bytes) -> bool:
        """Verify pinned host and proposal before trusting the host's MAC."""
        try:
            self._require_original_authority()
            p = self.pins
            return (type(receipt) is HostLoopReservation
                    and receipt.event_id == p.event_id
                    and receipt.host_id == self.binding.host_id
                    and receipt.fence == p.expected_host_fence
                    and payload == receipt.signed_bytes
                    # A signature alone can outlive a rolled-back projection
                    # or an unreadable external monotonic authority.
                    and self.host.reconcile(p.event_id) == receipt
                    and self.host.verify_host_receipt(receipt, payload))
        except (HostAuthorityError, ValueError, TypeError):
            return False

    def verify_host_binding(self, plan: SignedJourneyLoopPlan,
                            run: FullStateRunBinding) -> bool:
        """Match a host-reserved Journey to precommitted run metadata.

        This checks a typed external pin; it is not an attestation that any
        pretrained model loaded, trained or persisted.
        """
        try:
            self._require_original_authority()
            p = self.pins
            if type(plan) is not SignedJourneyLoopPlan or type(run) is not FullStateRunBinding:
                return False
            plan.__post_init__()
            run.validate()
            receipt = self.host.reconcile(p.event_id)
            return (type(receipt) is HostLoopReservation
                    and receipt == plan.reservation
                    and self.verify_host_reservation(plan.reservation,
                                                     plan.reservation.signed_bytes)
                    and plan.journey_sha256 == p.journey_sha256
                    and plan.learner_checkpoint_sha256 == p.learner_checkpoint_sha256
                    and plan.allocation.event_id == p.event_id
                    and plan.allocation.exam_id == p.exam_id
                    and plan.allocation.objective_contract_sha256 == p.contract_sha256
                    and plan.allocation.signed_binding_sha256 == p.signed_binding_sha256
                    and plan.allocation.training_roster_sha256 == p.training_roster_sha256
                    and plan.allocation.weak_skills == p.weak_skills
                    and plan.host_fence == p.expected_host_fence
                    and run.run_id == p.expected_run_id
                    and run.objective_id == p.expected_objective_id
                    and run.source_tree_sha256 == p.expected_source_tree_sha256
                    and run.prior_checkpoint_sha256 == p.learner_checkpoint_sha256
                    and hmac.compare_digest(run.commitment_sha256(),
                                           p.fullstate_binding_sha256))
        except (HostAuthorityError, ValueError, TypeError):
            return False
