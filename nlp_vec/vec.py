import numpy as np

class Vectorize_GloVe:
    
    def __init__(self):  
        word2vec = {}
        word_embedding = []
        words = []

        # directory to the pretrain weights downloaded from https://nlp.stanford.edu/projects/glove/
        with open('/work/data/glove/glove.6B.50d.txt') as file:
            for entry in file:
                word_and_vec = entry.split()
                word = word_and_vec[0]
                vec = np.asarray(word_and_vec[1:], dtype ="float32")
                word2vec[word] = vec
                word_embedding.append(vec)
                words.append(word)
            print("This pretrain word vectors include {} word embedding".format(len(words)))
        
        self.word2vec = word2vec
        self.word_embedding = np.array(word_embedding)
        self.word2index = {v:k for k,v in enumerate(words)}
        self.V, self.D = self.word_embedding.shape
        
    def fit(self, data):
        pass
    
    def transform(self, data):
        """
        This is a function to transform data into Glove vectors. 
        
        Parameter:
        -----------
            data: list, numpy array
                List or numpy array where each element is a sentence. Perhaps, you need to split the text by period(0). Also, this vectorization is insensitive to case. 
                
        Returns:
        ----------
            
        
        """
        
        
        
        X = np.zeros(len(data), self.D)
        n = 0
        empty = 0
        
        for sentence in data:
            token = sentence.lower().split()
            vecs = []
            # word in token:
            #   if word in 


def word2vect(sentences):
    return sentences[0]

def word():
    return "hello"