import unittest
from search import SearchEngine, preprocess

class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = SearchEngine([
            {'id':1,'text':'football team football win','category':'sport'},
            {'id':2,'text':'mobile software phone','category':'tech'},
            {'id':3,'text':'football software','category':'tech'}])
    def test_preprocessing(self):
        self.assertEqual(preprocess('The MOBILE phones! 2026'), ['mobile','phones'])
    def test_boolean_logic(self):
        self.assertEqual(self.e.boolean('football AND NOT software'), [0])
        self.assertEqual(self.e.boolean('(football OR mobile) AND software'), [1,2])
        self.assertEqual(self.e.boolean('football OR mobile AND software'), [0,1,2])
    def test_invalid_query(self):
        for q in ['football AND', '(football', 'football mobile']:
            with self.assertRaises(ValueError): self.e.boolean(q)
    def test_ranked_and_unknown(self):
        for model in ['tfidf','bm25']:
            self.assertEqual(self.e.search('mobile',model)[0]['id'],2)
            self.assertEqual(self.e.search('zzzzunknown',model),[])
            self.assertEqual(self.e.search('the and',model),[])
    def test_bm25_formula(self):
        import math
        score=self.e.search('mobile')[0]['score']
        expected=math.log(1+(3-1+.5)/(1+.5))*2.5/(1+1.5*(.25+.75*3/3))
        self.assertAlmostEqual(score,expected)

if __name__=='__main__': unittest.main()
