"""Read-only Windows ACL negative-evidence probe for private examiner material.

An operator-controlled ACL snapshot CANNOT prove independent custody.
This probe detects broad allow-ACE exposures, including writable ancestors.
It never changes permissions, creates accounts, or upgrades qualification.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .live_model_probe import _write_once, _receipt

SCHEMA = "labyrinth.examiner-local-acl-custody-negative-v1"
SID = re.compile(r"S-1-\d+(?:-\d+)+\Z")
BROAD_SIDS = {
    "S-1-1-0",       # Everyone
    "S-1-5-2",       # Network
    "S-1-5-4",       # Interactive
    "S-1-5-11",      # Authenticated Users
    "S-1-5-32-545",  # Builtin Users
    "S-1-5-32-546",  # Builtin Guests
    "S-1-5-32-547",  # Power Users
}
READ_DATA = 0x00000001
WRITE_DANGEROUS = (
    0x00000002  # WriteData/CreateFiles
    | 0x00000004  # AppendData/CreateDirectories
    | 0x00000010  # WriteExtendedAttributes
    | 0x00000040  # DeleteSubdirectoriesAndFiles
    | 0x00000100  # WriteAttributes
    | 0x00010000  # Delete
    | 0x00040000  # ChangePermissions
    | 0x00080000  # TakeOwnership
)

# This is intentionally read-only. Path supplied via an environment variable
# rather than interpolated into PowerShell source or a shell command.
_ACL_PS = r"""
$ErrorActionPreference = 'Stop'
$path = $env:LABYRINTH_ACL_PROBE_PATH
if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw 'not an existing examiner-private file'
}
$initial = (Get-Item -LiteralPath $path -ErrorAction Stop).FullName
$current = $initial
$layers = @()
while ($true) {
    $acl = Get-Acl -LiteralPath $current -ErrorAction Stop
    $aces = @()
    foreach($ace in $acl.Access) {
        $sid = $null
        try {
            $sid = $ace.IdentityReference.Translate(
                [System.Security.Principal.SecurityIdentifier]).Value
        } catch { }
        $aces += [pscustomobject]@{
            sid = $sid
            mask = [int64]$ace.FileSystemRights
            access = $ace.AccessControlType.ToString()
            inherited = [bool]$ace.IsInherited
        }
    }
    $layers += [pscustomobject]@{
        is_target = ($current -eq $initial)
        aces = @($aces)
    }
    $parent = Split-Path -Path $current -Parent
    if ([string]::IsNullOrEmpty($parent) -or $parent -eq $current) { break }
    $current = $parent
}
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
[pscustomobject]@{
    platform = 'windows'
    identity_sid = $identity.User.Value
    targets = @([pscustomobject]@{
        role = $env:LABYRINTH_ACL_PROBE_ROLE
        layers = @($layers)
    })
} | ConvertTo-Json -Depth 12 -Compress
"""


def classify_acl_snapshot(snapshot):
    """Conservatively classify ACL allows, never certify an independent actor.

    An Allow ACE may be superseded by a Deny, so findings describe
    *potential exposure* rather than asserting effective access. Only a
    separate security principal executing a denied-read probe can qualify
    an actual boundary; such a probe is not present in this source.
    """
    findings = set()
    incomplete = False
    if (type(snapshot) is not dict
            or snapshot.get("platform") != "windows"
            or type(snapshot.get("identity_sid")) is not str
            or SID.fullmatch(snapshot["identity_sid"]) is None
            or type(snapshot.get("targets")) is not list
            or not 1 <= len(snapshot["targets"]) <= 2):
        incomplete = True
        targets = []
        identity = ""
    else:
        targets = snapshot["targets"]
        identity = snapshot["identity_sid"]

    seen_roles = set()
    for target in targets:
        if (type(target) is not dict
                or set(target) != {"role", "layers"}
                or target["role"] not in ("private_cases", "signer_private_key")
                or target["role"] in seen_roles
                or type(target["layers"]) is not list
                or not 1 <= len(target["layers"]) <= 64):
            incomplete = True
            continue
        role = target["role"]
        seen_roles.add(role)
        layers = target["layers"]
        if (type(layers[0]) is not dict
                or layers[0].get("is_target") is not True):
            incomplete = True
        for i, layer in enumerate(layers):
            if (type(layer) is not dict or set(layer) != {"is_target", "aces"}
                    or type(layer["is_target"]) is not bool
                    or layer["is_target"] is not (i == 0)
                    or type(layer["aces"]) is not list
                    or len(layer["aces"]) > 256):
                incomplete = True
                continue
            for ace in layer["aces"]:
                if (type(ace) is not dict
                        or set(ace) != {"sid", "mask", "access", "inherited"}
                        or type(ace["sid"]) is not str
                        or SID.fullmatch(ace["sid"]) is None
                        or type(ace["mask"]) is not int
                        or type(ace["access"]) is not str
                        or ace["access"] not in ("Allow", "Deny")
                        or type(ace["inherited"]) is not bool):
                    incomplete = True
                    continue
                if ace["access"] != "Allow" or ace["sid"] not in BROAD_SIDS:
                    continue
                mask = ace["mask"] & 0xFFFFFFFF
                suffix = role.upper()
                if i == 0 and mask & READ_DATA:
                    findings.add("BROAD_READ_" + suffix)
                if mask & WRITE_DANGEROUS:
                    findings.add(("BROAD_WRITE_" + suffix if i == 0
                                  else "BROAD_WRITE_ANCESTOR_OF_" + suffix))
    if "private_cases" not in seen_roles:
        incomplete = True
    status = ("INSECURE_BROAD_ALLOW" if findings else
              "ACL_INSPECTION_INCOMPLETE" if incomplete else
              "NO_BROAD_ALLOW_DETECTED_NOT_ATTESTED")
    return {
        "schema": SCHEMA,
        "status": status,
        "finding_semantics": "POTENTIAL_BROAD_ALLOW_NOT_EFFECTIVE_TOKEN_ACCESS",
        "findings": sorted(findings),
        "inspected_roles": sorted(seen_roles),
        "identity_sid_sha256": sha256(identity.encode("utf-8")).hexdigest()
            if identity else None,
        "acl_inspection_complete": not incomplete,
        "acl_isolation_verified": False,
        "independent_custody_verified": False,
        "candidate_principal_read_denial_verified": False,
        "evidence_class": "LOCAL_OPERATOR_ACL_OBSERVATION_ONLY",
        "limitations": [
            "Host-controlled ACLs and file snapshots do not establish a separate examiner.",
            "Allow ACEs are potential exposures, not resolved effective-token access.",
            "No attempt was made to log in as a separate candidate or test its denied reads.",
            "A same-operator examiner cannot graduate to independent custody.",
        ],
    }


def _collect_windows_acl(path, role):
    """One read-only Get-Acl process, no credentials or security changes."""
    if os.name != "nt":
        raise RuntimeError("local Windows NTFS examiner ACL probe requires Windows")
    item = Path(path)
    if not item.is_file() or item.is_symlink():
        raise ValueError("examiner-private input must be existing regular file")
    env = dict(os.environ)
    env["LABYRINTH_ACL_PROBE_PATH"] = str(item.absolute())
    env["LABYRINTH_ACL_PROBE_ROLE"] = role
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _ACL_PS],
        shell=False, text=True, capture_output=True, timeout=15,
        env=env, check=False)
    if result.returncode != 0 or len(result.stdout) > 100000:
        raise RuntimeError("examiner ACL read-only inspection failed")
    try:
        snapshot = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("examiner ACL inspection returned invalid data") from exc
    if (type(snapshot) is not dict
            or type(snapshot.get("targets")) is not list
            or len(snapshot["targets"]) != 1
            or snapshot["targets"][0].get("role") != role):
        raise RuntimeError("examiner ACL inspection lost target identity")
    return snapshot


def audit_local_examiner_custody(private_cases_path, out_path, *,
                                 signing_private_key_path=None):
    """Persist an immutable local *negative-evidence* record, never a PASS."""
    out = Path(out_path)
    paths = [Path(private_cases_path)]
    if signing_private_key_path is not None:
        paths.append(Path(signing_private_key_path))
    if out.exists():
        raise FileExistsError("examiner ACL report output must be fresh")
    if len({str(p.absolute()).lower() for p in paths + [out]}) != len(paths) + 1:
        raise ValueError("examiner input and report paths must be distinct")
    snapshots = [
        _collect_windows_acl(path, role)
        for path, role in zip(paths, ("private_cases", "signer_private_key"))]
    if len({s["identity_sid"] for s in snapshots}) != 1:
        raise RuntimeError("ACL audit process identity changed across files")
    merged = {
        "platform": "windows",
        "identity_sid": snapshots[0]["identity_sid"],
        "targets": [s["targets"][0] for s in snapshots],
    }
    report = classify_acl_snapshot(merged)
    # The output itself is a local, unkeyed receipt, not a custody attestation.
    return _write_once(out, _receipt(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", required=True)
    parser.add_argument("--signer-private-key")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = audit_local_examiner_custody(
        args.private, args.out, signing_private_key_path=args.signer_private_key)
    print(json.dumps({
        "status": report["status"],
        "findings": report["findings"],
        "acl_inspection_complete": report["acl_inspection_complete"],
        "independent_custody_verified": False,
        "receipt_sha256": report["receipt_sha256"],
    }, sort_keys=True))
    # Deliberately no exit status meaning 'independent custody verified'.
    return 2 if report["status"] == "INSECURE_BROAD_ALLOW" else 3


if __name__ == "__main__":
    sys.exit(main())
