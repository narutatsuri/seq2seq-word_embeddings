# Sequence to Sequence with Word Embeddings

## Requirements
Use requirements.txt to install all requirements.
```
keras==2.9.0
numpy==1.22.3
pandas==1.4.2
scipy==1.8.0
tqdm==4.64.0
```

## Setup
Add dataset ```dataset.csv``` to ```data/```. 
```dataset.csv``` must contain two columns, 
the first column corresponding to the training input to the model and the 
second corresponding to the training target of the model.

Add GloVe embeddings ```glove.txt``` to ```embeddings/```.

Embedding dimensions can be adjusted in ```util/__init__.py``` under 
```embeddings_dim```.

## Running the Code
```
python main.py
```