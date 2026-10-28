"""Offline public study CLI. Does not invoke independent gate or activation."""
import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--development-smoke",action="store_true",
                        help="Use exposed dev seeds; never consume prospective pairs")
    parser.add_argument("--expect-source-manifest-sha256",
                        help="Externally pinned SHA256 of the canonical source manifest")
    args=parser.parse_args(argv)
    if not args.development_smoke and not args.expect_source_manifest_sha256:
        parser.error("full study requires an externally pinned source-manifest SHA256")
    if args.output.exists():
        parser.error("output already exists; no overwrite permitted")
    try:
        import torch
        from labyrinth.minotaur_prospective_study import (
            run_development_smoke,run_study,
        )
        if torch.__version__!="2.5.1+cpu" or torch.version.cuda is not None:
            raise RuntimeError("full public study requires pinned CPU torch 2.5.1+cpu")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if args.development_smoke:
            if args.expect_source_manifest_sha256:
                parser.error("development smoke does not accept a full-study source pin")
            report=run_development_smoke(torch)
        else:
            report=run_study(torch,expected_source_manifest_sha256=args.expect_source_manifest_sha256)
        payload=(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf8")
        with args.output.open("xb") as destination:
            destination.write(payload)
    except (RuntimeError,ValueError,FileExistsError,OSError) as error:
        parser.error(str(error))
    print(json.dumps({"output_sha256":hashlib.sha256(payload).hexdigest(),
                      "public_primary_gate":report["public_primary_gate"],
                      "qualification":"NOT_ESTABLISHED"},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
