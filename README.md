# nlp_vec

## About this repo

This repo has codes for following functions:
1) Vectorize sentences using standard word embedding such as GloVe and word2vec. 
2) Build word2vec from scratch using negative sampling and skip-gram
3) Fine-tune large pre-trained language models using scientific data. 
4) Fine-tune BERT model. 
5) Text similarity search engine based on cosine similarity and euclidean distances. 

## References

The implmentation for using pre-trained word embedding is minor modification from [Natural language processing 2](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/bow_classifier.py) by [Lazyprogrammer](https://lazyprogrammer.me/)

The implmentation of word2vec (skip-gram with negative sampling) is minor modification from [deep_learning_NLP](https://github.com/Tixierae/deep_learning_NLP/blob/master/skipgram/sg_d2v_numpy.ipynb) by 
[Tixierae](https://github.com/Tixierae)

## How to run this app

First, clone this repository and open a terminal inside the folder. 


Install pretrained vectors:

word2vec

```bash
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gunzip GoogleNews-vectors-negative300.bin.gz 
```

GloVe

```bash
wget -c https://nlp.stanford.edu/data/glove.6B.zip
```
Install dependencies:

```bash

pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```
