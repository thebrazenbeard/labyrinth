"""Bind actual Rezon receipt-bundle *bytes* to a Labyrinth source pointer.

This closes one SOURCE-ONLY substitution gap in ``rezon_pointer_mapping_r57``:
an opaque declared receipt-bundle SHA may now be checked against bytes actually
read by the caller. It does NOT parse or verify Rezon signatures, establish
independent ownership of expectations, attest a host, or grant authority.
"""

from __future__ import annotations

from hashlib import sha256
import hmac
from typing import NamedTuple

from labyrinth.evidence_pointer import ExpectedSourcePinsV1
from labyrinth.rezon_pointer_mapping_r57 import (
    ExpectedRezonPinsV1,
    RezonSourceAlignmentError,
    RezonSourceClaimsV1,
    compare_source_evidence,
)


MAX_BUNDLE_BYTES = 4 * 1024 * 1024


class SourceBundleByteObservationR69(NamedTuple):
    """Inert tuple result; cannot be promoted via ``object.__setattr__``."""

    labyrinth_run_id: str
    rezon_run_id: str
    bundle_sha256: str

    @property
    def disposition(self) -> str:
        return "SOURCE_BUNDLE_BYTES_MATCH_UNATTESTED"

    @property
    def source_bundle_bytes_matched(self) -> bool:
        return True

    @property
    def source_receipts_verified(self) -> bool:
        return False

    @property
    def host_execution_verified(self) -> bool:
        return False

    @property
    def replay_exclusion_verified(self) -> bool:
        return False

    @property
    def model_identity_verified(self) -> bool:
        return False

    @property
    def consent_verified(self) -> bool:
        return False

    @property
    def permissions_verified(self) -> bool:
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


def compare_bundle_bytes(
    labyrinth_pointer_payload: bytes,
    rezon_receipt_bundle: bytes,
    *,
    expected_labyrinth: ExpectedSourcePinsV1,
    observed_rezon: RezonSourceClaimsV1,
    expected_rezon: ExpectedRezonPinsV1,
) -> SourceBundleByteObservationR69:
    """Require matching *actual bytes* as well as all R57 independent pins.

    The bundle is deliberately opaque: its byte identity is asserted, not its
    structure, freshness, signature custody, or semantic truth. No result from
    this function may authorize training, examination, merging or deployment.
    """
    # Check the canonical source and both independently supplied pin sets
    # before hashing a potentially large untrusted opaque blob.
    aligned = compare_source_evidence(
        labyrinth_pointer_payload,
        expected_labyrinth=expected_labyrinth,
        observed_rezon=observed_rezon,
        expected_rezon=expected_rezon,
    )
    if (type(rezon_receipt_bundle) is not bytes
            or not 1 <= len(rezon_receipt_bundle) <= MAX_BUNDLE_BYTES):
        raise RezonSourceAlignmentError("bounded exact Rezon bundle bytes required")

    digest = sha256(rezon_receipt_bundle).hexdigest()
    if not hmac.compare_digest(digest, expected_rezon.receipt_bundle_sha256):
        raise RezonSourceAlignmentError("Rezon bundle bytes differ from source pins")

    return SourceBundleByteObservationR69(
        labyrinth_run_id=aligned.labyrinth_run_id,
        rezon_run_id=aligned.rezon_run_id,
        bundle_sha256=digest,
    )
