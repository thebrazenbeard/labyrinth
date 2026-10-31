"""Read-only, SOURCE-ONLY audit of the legacy R12 consumed-event ledger.

The R12 SQLite ``consumed`` rows retain an event ID, proposal *digest*, epoch,
fence, and HMAC. They do not retain the proposal bytes, the exam ID within
those bytes, or any run ID. It is therefore impossible to reconstruct a
complete global event/exam/run migration witness from this store alone.

This module exports a deterministic inventory of the *actual* retained rows
after verifying the existing source-level seals and epoch readback. It will
never import that inventory as authority or grant a new reservation. The
legacy service and its synthetic anchor are not host custody evidence.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json

from .host_authority_reservation_r12 import (
    HostAuthorityError,
    HostAuthorityHold,
    HostProviderReservationService,
)


SCHEMA = "labyrinth.legacy-consumed-source-inventory.v1"
_DOMAIN = b"labyrinth.legacy-consumed-source-inventory.v1\0"
_LEGACY_COLUMNS = (
    "event_id", "proposal_sha", "epoch", "fence", "host_signature",
)
_ABSENT_IDENTITIES = ("exam_id", "run_id", "proposal_bytes")


class LegacyMigrationHold(ValueError):
    """A legacy source audit is not sufficient to migrate authority."""


@dataclass(frozen=True)
class LegacyConsumedEvent:
    event_id: str
    proposal_sha256: str
    epoch: int
    fence: int
    host_signature_hex: str


@dataclass(frozen=True)
class LegacyMigrationSourceInventory:
    binding_sha256: str
    history_sha256: str
    epoch: int
    events: tuple[LegacyConsumedEvent, ...]
    schema: str = SCHEMA
    missing_identity_fields: tuple[str, ...] = _ABSENT_IDENTITIES
    source_hmac_checked: bool = True
    source_anchor_readback_checked: bool = True
    independent_host_custody_verified: bool = False
    exam_or_run_uniqueness_verified: bool = False
    migration_authorized: bool = False

    def canonical_bytes(self) -> bytes:
        """Exact, deterministic source inventory; not a permission token."""
        record = {
            "binding_sha256": self.binding_sha256,
            "history_sha256": self.history_sha256,
            "epoch": self.epoch,
            "events": [
                {"event_id": row.event_id,
                 "proposal_sha256": row.proposal_sha256,
                 "epoch": row.epoch, "fence": row.fence,
                 "host_signature_hex": row.host_signature_hex}
                for row in self.events
            ],
            "schema": self.schema,
            "missing_identity_fields": list(self.missing_identity_fields),
            "source_hmac_checked": self.source_hmac_checked,
            "source_anchor_readback_checked": self.source_anchor_readback_checked,
            "independent_host_custody_verified": self.independent_host_custody_verified,
            "exam_or_run_uniqueness_verified": self.exam_or_run_uniqueness_verified,
            "migration_authorized": self.migration_authorized,
        }
        return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")

    @property
    def inventory_sha256(self) -> str:
        return sha256(_DOMAIN + self.canonical_bytes()).hexdigest()

    def require_authorized_migration(self) -> None:
        """Always fail: event-only source records lack global consumed identity.

        Neither a caller-supplied run ID nor a claimed external archive may be
        promoted to authority by this inventory. An independently controlled
        service must qualify the complete migrated history separately.
        """
        raise LegacyMigrationHold(
            "legacy R12 records omit exam_id, run_id and proposal bytes; "
            "independent complete-history migration witness required")


def inspect_legacy_source_inventory(
    service: HostProviderReservationService,
) -> LegacyMigrationSourceInventory:
    """Read surviving R12 rows with source MAC and synthetic-anchor checks.

    The service holds Python-accessible signing keys and a replaceable remote
    adapter, so this method cannot attest host ownership, system anti-rollback,
    or the authenticity of a caller-supplied adapter.
    """
    if type(service) is not HostProviderReservationService:
        raise LegacyMigrationHold("exact legacy source service required")
    try:
        service._check_keys()
        with closing(service._connection()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            columns = tuple(row[1] for row in conn.execute(
                'PRAGMA table_info("consumed")').fetchall())
            if columns != _LEGACY_COLUMNS:
                raise LegacyMigrationHold("legacy consumed schema differs from audited format")
            remote_before = service._read_remote()
            epoch, history_sha256 = service._load(conn)
            rows = conn.execute(
                "SELECT event_id, proposal_sha, epoch, fence, host_signature "
                "FROM consumed ORDER BY epoch").fetchall()
            remote_after = service._read_remote()
            if (remote_before != remote_after or remote_after.epoch != epoch
                    or not hmac.compare_digest(remote_after.history_sha256,
                                               history_sha256)):
                raise LegacyMigrationHold("source ledger and epoch adapter disagree")
            # The source's _load already checks every row and local HMAC,
            # reconstructs the event-history digest, and bounds event count.
            events = tuple(LegacyConsumedEvent(*row) for row in rows)
            return LegacyMigrationSourceInventory(
                service.binding.sha256, history_sha256, epoch, events)
    except LegacyMigrationHold:
        raise
    except (HostAuthorityError, HostAuthorityHold, TypeError, ValueError,
            OSError) as exc:
        raise LegacyMigrationHold("legacy source state or epoch readback failed") from exc
