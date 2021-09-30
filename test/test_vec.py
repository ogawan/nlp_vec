import os 
import unittest
import numpy as np
from nlp_vec import vec

class Test_nlp(unittest.TestCase):
    
    def test_word2vect(self):
        self.assertEqual(vec.word2vect("hello"), "h")    
        
    def test_word(self):
        self.assertEqual(vec.word(), "hello") 
        
    def test_Vectorize_GloVe(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + "/glove_test_data.npy"
        glove_vectorizer = vec.Vectorize_GloVe()
        vecs = glove_vectorizer.fit_transform(["This is an apple", "I am orange boy with a pinapple", "Fat elephant"])
        
        self.assertEqual((vecs == np.load(path)).all(), True) 
        
if __name__ == '__main__':
    unittest.main()