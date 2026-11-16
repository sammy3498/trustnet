import unittest
from backend.app.scoring.evidence_verdict import reconcile
from backend.app.scoring.calculator import calculate_overall_score as weighted, get_label
KEYS = ('evidence_strength','factual_consistency','source_provenance','bias_framing','incentive_conflict')
def rows(vals):
    return {k:dict(score=v, justification='fixture') for k,v in zip(KEYS, vals)}

class Replay(unittest.TestCase):
    def test_normal_conflicting_factors(self):
        f = rows([0,85,60,90,100])
        self.assertEqual(weighted(**{k:r['score'] for k,r in f.items()}),50)
        score,reason = reconcile('contradicts', f, weighted, get_label)
        self.assertIsNone(reason)
        self.assertEqual(score,0)
        self.assertIn('85/100', f['factual_consistency']['justification'])
    def test_deep_mixed(self):
        f = rows([50,90,69,90,100])
        self.assertEqual(weighted(**{k:r['score'] for k,r in f.items()}),73)
        self.assertIsNone(reconcile('mixed', f, weighted, get_label)[0])
    def test_supported_and_refuted_controls(self):
        for direction,values in [('supports',[95,90,80,70,60]),('contradicts',[10,15,80,90,90])]:
            self.assertEqual(reconcile(direction,rows(values),weighted,get_label),
                             (values[0],None))
    def test_weak_support_and_disagreement(self):
        self.assertEqual(reconcile('supports',rows([90,10,90,90,90]),weighted,get_label)[0], 90)
        self.assertIsNone(reconcile('insufficient',rows([50]*5),weighted,get_label)[0])
