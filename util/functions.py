from util import *
import numpy as np
import pickle as pkl
from pathlib import Path
import pandas as pd
from string import punctuation
import openai
from tqdm import tqdm
import time
from scipy.spatial import KDTree


def load_word_to_glove_embeddings():
    """
    Load word -> embeddings mapping as dictionary.
    """
    if not Path(embeddings_dic_dir).is_file():
        embeddings = {}
        with open(embeddings_dir, "r") as f:
            for line in f.readlines():
                embeddings[line.split()[0]] = np.array([float(i) for i in line.split()[1:]])
        with open(embeddings_dic_dir, "wb") as dic:
            pkl.dump(embeddings, dic)
    else:
        with open(embeddings_dic_dir, "rb") as dic:
            embeddings = pkl.load(dic)
            
    return embeddings

def load_glove_embeddings_to_word():
    """
    Load embeddings -> word mapping and construct KD Tree to search for closest
    embedding during inference.
    """
    if not Path(reverse_embeddings_dir).is_file():
        embeddings = []
        words = []
        with open(embeddings_dir, "r") as f:
            for line in f.readlines():
                embeddings.append(np.array([float(i) for i in line.split()[1:]]))
                words.append(line.split()[0])
        # Add EOL token to dictionary
        embeddings.append(np.array([[[0] * embeddings_dim]]))
        words.append("[EOL]")
        pkl.dump(embeddings, open(reverse_embeddings_dir, "wb"))
        pkl.dump(words, open(reverse_words_dir, "wb"))
    else:
        embeddings = pkl.load(open(reverse_embeddings_dir, "rb"))
        words = pkl.load(open(reverse_words_dir, "rb"))

    kdtree = KDTree(embeddings)
    
    return words, kdtree

def load_dataset(vectorizer,
                 dataset_dir):
    """
    Load dataset, with words converted to vectors and saves to a npy file.
    """
    if not Path(dataset_vectorized_dir).is_file():
        df = pd.read_csv(dataset_dir)
        vectorized_dataset = []
        
        for index, row in df.iterrows():        
            vectorized_dataset.append([vectorize_sentence(row[1], vectorizer),
                                       vectorize_sentence(row[2], vectorizer)])
        # Save to .npy file
        with open(dataset_vectorized_dir, "wb") as f:
            pkl.dump(vectorized_dataset, f)
    else:
        with open(dataset_vectorized_dir, "rb") as f:
            vectorized_dataset = pkl.load(f)
            
    return vectorized_dataset

def vectorize_sentence(sentence,
                       vectorizer):
    """
    """
    vectorized_sentence = []
    for word in clean_text(sentence):
        try:
            vectorized_sentence.append(vectorizer[word])
        except KeyError:
            pass
        
    return vectorized_sentence

def clean_text(text):
    """
    Clean sentence and return split sentence.
    """
    for item in punctuation:
        text = text.replace(item, " " + item + " ")
        
    return text.lower().split()

def embedding_to_word(embedding,
                      words, 
                      kdtree):
    """
    Find closest word embedding using KD Tree and returns word corresponding
    to word embedding.
    """
    _, i = kdtree.query(embedding, k=1)
    
    return words[i]
    
def column(array, index):
    """
    Return column in position "index" of array.
    """
    return [row[index] for row in array]