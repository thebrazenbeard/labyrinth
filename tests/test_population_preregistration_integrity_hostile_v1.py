"""A prepared population cannot silently rewrite its examiner's sealed cohort."""

import unittest

from labyrinth.evolutionary_population_v1 import PopulationError, select_generation


class PopulationPreregistrationIntegrityTests(unittest.TestCase):
    def test_mutated_frozen_plan_alias_cannot_replace_preregistered_seal(self):
        from tests.test_evolutionary_population_v1 import PopulationTests

        fixtures = PopulationTests()
        fixtures.setUp()
        try:
            population, _ = fixtures._generation()
            original_commitment = population.plan.sealed_case_commitment
            object.__setattr__(population.plan, "sealed_case_commitment", "e" * 64)
            self.assertNotEqual(population.plan.sealed_case_commitment,
                                original_commitment)
            # Examiner records agree with the tampered plan, but not with the
            # originally prepared population's precommitment.
            examinations = fixtures._exams(population)
            with self.assertRaisesRegex(PopulationError, "plan|preregister|changed|integrity"):
                select_generation(population, examinations)
        finally:
            fixtures.doCleanups()


if __name__ == "__main__":
    unittest.main()
