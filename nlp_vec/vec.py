import numpy as np
from gensim.models import KeyedVectors

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
            The average of word vectors in sentences. If no corresponding vectors are found, the sentence is ignores. 
        
        """
        
        
        X = np.zeros((len(data), self.D))
        n = 0
        empty = 0
        
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0) # average in vertical direction. 
            else:
                empty += 1
            n += 1
        print("No. no words found {}/{}".format(empty, len(data)))
        return X
                    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def word2vect(sentences):
    return sentences[0]

def word():
    return "hello"

class Vectorize_word2vec:
    
    def __init__(self):
        self.word_vectors = KeyedVectors.load_word2vec_format(
              '/work/data/word2vec/GoogleNews-vectors-negative300.bin',
            binary=True
        ) 
        self.word_not_found = []
        
    def fit(self, data):
        pass
    
    def transform(self, data):
        v = self.word_vectors.get_vector('king')
        self.D = v.shape[0]
        
        
        X = np.zeros((len(data), self.D))
        n = 0
        empty = 0
        for sentence in data:
            tokens = sentence.split()
            vecs = []
            m = 0
            for word in tokens:
                try:
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    # Store words that are not found
                    self.word_not_found.append(word)
                    pass
                                               
            if len(vecs) > 0:
                vecs =  np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                empty += 1
            n += 1
        return X           
    
    def fit_tansform(self, data):
        self.fit(data)
        return self.transform(data)
        
                                           
                                               
                
            
                    
                    
                    
            
        