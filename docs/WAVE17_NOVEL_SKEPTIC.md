# Wave17 Novel Router process trial hostile review

**Frozen source:** `343ec2fadc5b0149e60d45a66b86a4d3508c3383`
(`scripts/novel_router_model_trial.py`), read-only production audit.

**Isolated test-only artifact:** `tests/test_wave17_novel_skeptic.py`.
Only a fixed local Python child fixture is executed. The command-mode
adapter is not invoked, and no neural model, paid compute, private host
resource or production service is touched.

## NR17-S01 — forged typed custody metadata survives report replay

**FAILS (four independent mutations).** A valid fixed-child receipt reports
the Git commit, executable-file metadata, local source files, and an optional
model artifact separately. Change any one to a structurally invalid value:

- `source_commit='not-a-git-record'`
- `executable='not-an-executable-record'`
- `source_files={'runner':'fake'}`
- `model_artifact={'sha256':'faked'}`

Then recompute the unkeyed trial receipt SHA. Every forged report still
returns `verify_trial_receipt=True`. This is schema integrity failure:
downstream tools cannot trust those fields' *shape*, even while model
identity remains correctly classified unqualified. A syntactically plausible
hash would still be forgeable without an independent signer; exact shape
checks must not be misrepresented as verified external custody.

## NR17-S02 — private transcript and seed printed by default

**FAILS (two fields).** Invoking the ordinary default CLI with explicit seed
produces one JSON report on stdout. The report explicitly declares itself
`HOST_PRIVATE_CASE_SEED_AND_MODEL_OUTPUT`, yet stdout includes both the
exact private `configuration.case_seed` and `turns[].child.raw_stdout_base64`.
Redirected stdout, process supervisors, CI logs, or chat automation can
therefore unintentionally disclose the active case and the unredacted
child response. A clean design should either require an explicit private
output destination/opt-in, or print only a public redacted projection by
default. This is a disclosure risk, **not a proven unauthorized recipient**.

## Reproduction

```bash
PYTHONPATH=. python3 -B -m unittest discover -s tests \
  -p test_wave17_novel_skeptic.py -q
PYTHONPATH=. python3 -O -B -m unittest discover -s tests \
  -p test_wave17_novel_skeptic.py -q
```

These receipts establish source-level falsifiers and a real local fixture
run. They do not demonstrate trained neural reasoning, independent model
identity, protected examination custody, signed source provenance or an OS
sandbox. The runner's scoring logic may be stronger than its output-
publication and metadata provenance contracts; these must be evaluated
separately.

## Post-fix independent reattack

**SURVIVES the two reproduced classes** on author clean exact Git SHA
`a6ebeccf06e692c1b714c42ae6a1d31a09cd780e`. The independent
test-only `f921c26` suite, without changing any tests, returns
**2/2 PASS normal** and **2/2 PASS under `python -O`**, covering the
same six formerly red provenance/private-stdout subcases.

The author test suite `test_novel_router_model_trial.py` was also
independently rerun: **23/23 PASS normal and 23/23 optimized**. This is a
source-test result; it does not turn unkeyed SHA consistency into signed
file/model custody or establish learned reasoning.
