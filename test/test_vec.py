import os 
import unittest
import numpy as np
from nlp_vec import vec
from nlp_vet.utils import utils

dir_path = os.path.dirname(os.path.realpath(__file__))

class Test_nlp(unittest.TestCase):
        
    def test_Vectorize_GloVe(self):
        path = dir_path + "/glove_test_data.npy"
        glove_vectorizer = vec.Vectorize_GloVe()
        vecs = glove_vectorizer.fit_transform(["This is an apple", "I am orange boy with a pinapple", "Fat elephant"])
        
        self.assertEqual((vecs == np.load(path)).all(), True) 
        
    def test_Vectorize_word2vec(self):
        path = dir_path + "/word2vec_test_data.npy"
        word2vec_vectorizer = vec.Vectorize_word2vec()
        vecs =  word2vec_vectorizer.fit_transform(["This is an apple", "I am orange boy with a pineapple", "Fat elephant", "organic chemistry", "Fuji mountain", "Megane TRPV1", "FGHSRHG$%&YEHDRT"])
        self.assertEqual((vecs == np.load(path)).all(), True) 
    
    def test_utils(self):
        x = np.array([0.3, -1, 0.4, 0.3])
        
        self.assertEqual(utils(x), np.array([0.29605921,0.08068555,0.32719603, 0.29605921]))
                    
        
if __name__ == '__main__':
    unittest.main()