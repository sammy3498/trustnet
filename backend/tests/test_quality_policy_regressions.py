from types import SimpleNamespace as NS
import unittest
from backend.app.evidence.quality_policy import directional_gate, prefer_comparable_sources
from backend.app.evidence.provenance import classify_source_quality


def doc(name, quality, semantic=.8):
    return NS(url='https://'+name, source=name, quality=quality, semantic_score=semantic,
              bm25_score=1., reranker_score=1., evidence_origin='webpage')


class QualityTests(unittest.TestCase):
    def test_publisher_boundary(self):
        self.assertEqual(classify_source_quality('https://www.unicef.org/research'), 'high')
        self.assertEqual(classify_source_quality('https://unicef.org.example.com/x'), 'low')

    def test_selection(self):
        a,b=doc('a.org','low'),doc('b.gov','high')
        def choose(ranks):
            return prefer_comparable_sources([a], [a,b], ranks, str)
        self.assertEqual(choose({}), [b])
        self.assertEqual(choose({a.url:{'exception':1}}), [b])
        b.semantic_score=.45
        self.assertEqual(choose({}), [a])

    def test_gate(self):
        a,b=doc('a.gov','high'),doc('b.org','low')
        def row(d,s):
            return dict(url=d.url, directness='direct', relevance='high', stance=s)
        def gate(rows):
            return directional_gate('supports', rows, [a,b], str, 2)
        self.assertTrue(gate([row(a,'neutral'),row(b,'supports')]))
        self.assertTrue(gate([row(a,'contradicts'),row(b,'supports')]))
        self.assertIsNone(gate([row(a,'supports')]))
        a.quality=b.quality='medium'
        self.assertIsNone(gate([row(a,'supports')]*2))
        self.assertIsNone(gate([row(a,'supports'),row(b,'supports')]))


if __name__ == '__main__':
    unittest.main()
