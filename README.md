# nlp_vec

## About this repo

This repo has codes for following functions:
1) Vectorize sentences using standard word embedding such as GloVe and word2vec. 
2) Fine-tune large pre-trained language models using scientific data. 
3) Representative app using 1) and 2). 

## How to run this app

First, clone this repository and open a terminal inside the folder. 


Install pretrained vectors:

word2vec

```bash
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```

GloVe

```bash
https://nlp.stanford.edu/data/glove.6B.zip
```
Install dependencies:

```bash

pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

## References

The implmentation for word embedding is minor modification from [Natural language processing 2](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/bow_classifier.py) by [Lazyprogrammer](https://lazyprogrammer.me/)