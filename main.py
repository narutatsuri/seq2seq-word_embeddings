from util import *
from util.functions import *
from util.models import *
import sys


# Load vectorizer (currently only has GloVe)
glove_dict = load_word_to_glove_embeddings()
words, kdtree = load_glove_embeddings_to_word()

# Load dataset converted to vectors
dataset = load_dataset(glove_dict,
                       dataset_1_dir)
input_train = column(dataset, 0)
target_train = column(dataset, 1)

# Create seq2seq model
model = seq2seq(glove_dict,
                words,
                kdtree)

# Train model
model.train(input_train, 
            target_train,
            500,
            load_model="")