"""Prediction snapshot integrity contract, independent of neural training."""
import unittest
import json
from labyrinth.minotaur_v7_intermediate_audit import canonical_prediction_json

class IntermediatePredictionArtifactTests(unittest.TestCase):
    def test_canonical_order_and_no_hidden_gold_labels(self):
        a={2:{"mod101":{(2,1):8,(1,2):7}}}
        b={2:{"mod101":{(1,2):7,(2,1):8}}}
        raw=canonical_prediction_json(a)
        self.assertEqual(raw,canonical_prediction_json(b))
        obj=json.loads(raw)
        self.assertEqual(obj["records"][0]["cases"]["mod101"],[[1,2,7],[2,1,8]])
        self.assertNotIn("target",raw)
    def test_refuses_boolean_or_unknown_prediction_values(self):
        for bad in ({1:{"mod101":{(0,1):True}}}, {1:{"mod101":{(0,1):-1}}},
                    {1:{"mod101":{(0,1):101}}}, {1:{"mod101":{(0,1):"6"}}}):
            with self.assertRaises(ValueError):canonical_prediction_json(bad)

