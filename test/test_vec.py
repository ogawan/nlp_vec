import unittest
from nlp_vec import vec

class Test_nlp(unittest.TestCase):
    
    def test_word2vect(self):
        self.assertEqual(vec.word2vect("hello"), "h")    
        
    def test_word(self):
        self.assertEqual(vec.word(), "hello") 

if __name__ == '__main__':
    unittest.main()