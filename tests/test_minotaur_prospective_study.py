"""Protocol integrity tests, never a private or full 100-pair evaluation."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch
except ImportError:
    torch=None


@unittest.skipUnless(torch is not None,"torch optional")
class ProspectivePublicStudyTests(unittest.TestCase):
    def test_seed_manifest_is_unique_domain_separated_and_development_disjoint(self):
        from labyrinth import minotaur_prospective_study as study
        manifest=study.seed_manifest()
        self.assertEqual(manifest,study.seed_manifest())
        self.assertEqual(len(manifest),100)
        for name,excluded in (("dataset",study.EXPOSED_DATA_SEEDS),
                              ("initialization",study.EXPOSED_INIT_SEEDS),
                              ("program",study.EXPOSED_PROGRAM_SEEDS),
                              ("corruption",frozenset())):
            values=[item[name+"_seed"] for item in manifest]
            self.assertEqual(len(set(values)),100)
            self.assertTrue(all(type(v) is int and 0<=v<2**31 for v in values))
            self.assertFalse(set(values)&excluded)

    def test_programs_are_unique_six_step_and_atom_disjoint_from_train_domain(self):
        from labyrinth import minotaur_prospective_study as study
        programs=study.make_programs(study.seed_manifest()[0]["program_seed"],128)
        self.assertEqual(len(programs),128)
        self.assertEqual(len(set(programs)),128)
        self.assertEqual(programs,study.make_programs(study.seed_manifest()[0]["program_seed"],128))
        for start,commands in programs:
            self.assertEqual(len(commands),6)
            state=list(start)
            for op,target,is_register,value in commands:
                x=state[target]
                y=state[value] if is_register else value
                self.assertFalse(x<=8 and y<=8)
                state[target]=study._oracle(op,x,y)

    def test_primary_decision_never_uses_partial_data_or_replaces_failures(self):
        from labyrinth import minotaur_prospective_study as study
        source_manifest=study.seed_manifest()
        expected_source=study._digest(study._source_manifest())
        def fake_pair(entry,win):
            def scored(n):
                return {"status":"SCORED","programs":{"exact":n,"total":128,
                        "states_correct":6*n,"states_total":768},
                        "withheld":{"correct":0,"total":28191},
                        "training":{},"coefficients_sha256":"c"*64}
            return {**entry,"status":"PROCESSED","win":win,
                    "training_atom_identity_sha256":"a"*64,
                    "program_suite_sha256":"b"*64,
                    "arms":{"bilinear":scored(13 if win else 12),
                            "no_xy":scored(0),"frozen":scored(0),"corrupted":scored(0)}}
        for wins,decision in ((0,"FAIL"),(58,"FAIL"),(59,"PASS"),(100,"PASS")):
            with self.subTest(wins=wins):
                outcomes=iter([fake_pair(entry,i<wins) for i,entry in enumerate(source_manifest)])
                with patch.object(study,"_pair",side_effect=lambda _entry,_torch:next(outcomes)):
                    with patch.object(torch,"get_num_threads",return_value=1):
                        with patch.object(torch,"are_deterministic_algorithms_enabled",return_value=True):
                            report=study.run_study(torch,expected_source_manifest_sha256=expected_source)
                self.assertEqual(report["pair_count"],100)
                self.assertEqual(report["wins"],wins)
                self.assertEqual(report["seed_manifest_sha256"],study._digest(source_manifest))
                self.assertEqual(report["public_primary_gate"],decision)
                self.assertEqual(report["qualification"],"NOT_ESTABLISHED")
                if wins==0:
                    self.assertEqual(report["exact_binomial_p_one_sided"],1.0)
                if wins==59:
                    self.assertGreater(report["exact_binomial_p_one_sided"],0.04)
                    self.assertLess(report["exact_binomial_p_one_sided"],0.05)
                if wins==100:
                    self.assertEqual(report["exact_binomial_p_one_sided"],2**-100)
        with patch.object(study,"_pair",return_value={"win":False}) as executor:
            partial=study.run_development_smoke(torch)
        self.assertEqual(executor.call_args.args[0]["dataset_seed"],47)
        self.assertEqual(partial["public_primary_gate"],"INCOMPLETE")
        self.assertIsNone(partial["exact_binomial_p_one_sided"])
        with self.assertRaises(ValueError):
            study._audit_records([{"win":True}],source_manifest)
        self.assertEqual(study._digest(study._source_manifest()),expected_source)

    def test_pair_failure_is_not_a_candidate_win(self):
        from labyrinth import minotaur_prospective_study as study
        program_metrics={"exact":128,"total":128}
        def scored(n):
            return {"status":"SCORED","programs":{**program_metrics,"exact":n}}
        attempts=iter([scored(128),{"status":"FAILED"},scored(0),scored(0)])
        with patch.object(study,"_arm",side_effect=lambda *_args,**_kw:next(attempts)):
            record=study._pair(study.seed_manifest()[0],torch)
        self.assertFalse(record["win"])

    def test_pair_setup_and_individual_control_failures_are_attributed(self):
        from labyrinth import minotaur_prospective_study as study
        entry=study.seed_manifest()[0]
        with patch.object(study,"make_programs",side_effect=ValueError("bad setup")):
            failed=study._pair(entry,torch)
        self.assertEqual(failed["status"],"SETUP_FAILED")
        self.assertFalse(failed["win"])
        self.assertEqual(set(failed["arms"]),{"bilinear","no_xy","frozen","corrupted"})
        self.assertTrue(all(x["status"]=="NOT_RUN" for x in failed["arms"].values()))

        def fake_arm(head,*args,**kwargs):
            if not kwargs["optimize"]:
                raise ValueError("frozen failure")
            return {"status":"SCORED","programs":{"exact":128,"total":128}}
        with patch.object(study,"_arm",side_effect=fake_arm):
            result=study._pair(entry,torch)
        self.assertEqual(result["arms"]["frozen"]["reason"],"ValueError")
        self.assertEqual(result["arms"]["corrupted"]["status"],"SCORED")
        self.assertFalse(result["win"])

    def test_postrun_source_drift_is_invalid_not_a_pass(self):
        from labyrinth import minotaur_prospective_study as study
        manifest=study._source_manifest()
        digest=study._digest(manifest)
        with patch.object(study,"_source_manifest",side_effect=[manifest,{"changed":"x"}]):
            with patch.object(study,"_pair",return_value={"win":True}):
                with patch.object(torch,"get_num_threads",return_value=1):
                    with patch.object(torch,"are_deterministic_algorithms_enabled",return_value=True):
                        with self.assertRaisesRegex(RuntimeError,"drifted during execution"):
                            study.run_study(torch,expected_source_manifest_sha256=digest)

    def test_full_study_enforces_pinned_numeric_environment(self):
        from labyrinth import minotaur_prospective_study as study
        digest=study._digest(study._source_manifest())
        with patch.object(torch,"get_num_threads",return_value=2):
            with self.assertRaisesRegex(RuntimeError,"numerical execution policy"):
                study.run_study(torch,expected_source_manifest_sha256=digest)
        with patch.object(torch,"get_num_threads",return_value=1):
            with patch.object(torch,"are_deterministic_algorithms_enabled",return_value=False):
                with self.assertRaisesRegex(RuntimeError,"numerical execution policy"):
                    study.run_study(torch,expected_source_manifest_sha256=digest)

    def test_cli_requires_pin_and_writes_development_smoke_exclusively(self):
        import io
        import json
        from contextlib import redirect_stdout,redirect_stderr
        from scripts.minotaur_prospective_public import main
        with TemporaryDirectory() as folder:
            path=Path(folder)/"public-smoke.json"
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["--output",str(path)])
            self.assertFalse(path.exists())
            smoke={"public_primary_gate":"INCOMPLETE",
                   "qualification":"NOT_ESTABLISHED"}
            with patch("labyrinth.minotaur_prospective_study.run_development_smoke",
                       return_value=smoke):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--output",str(path),"--development-smoke"]),0)
            self.assertEqual(json.loads(path.read_text()),smoke)
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["--output",str(path),"--development-smoke"])


if __name__=="__main__":
    unittest.main()
