from util import *
from util.functions import *
from keras.models import Sequential, model_from_json
from keras.layers import LSTM
from keras.optimizers import Adam
import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path


class seq2seq():
    def __init__(self, 
                 vectorizer,
                 words,
                 kdtree):
        """
        Constructor for sequence to sequence model.
        """
        self.model = Sequential()

        self.model.add(LSTM(latent_dim, 
                            return_sequences=True, 
                            input_shape=(None, embeddings_dim)))
        self.model.add(LSTM(embeddings_dim, 
                            return_sequences=True))
        
        self.model.compile(loss="mse",
                           optimizer="adam")
        
        self.vectorizer = vectorizer
        self.words = words
        self.kdtree = kdtree
            
    def train(self, 
              input_data, 
              target_data,
              epochs,
              load_model=""):
        """
        Train seq2seq model with SGD. 
        """
        # If load previously trained model to continue training, pass str of 
        # epoch number:
        if load_model != "":
            self.load(load_model)
        
        # Pad training data
        training_data = []
        for index in range(len(input_data)):
            input_instance = np.array([input_data[index]])
            target_instance = np.array([target_data[index]])
                            
            if input_instance.shape[1] > target_instance.shape[1]:
                target_instance = np.append(target_instance, 
                                            np.array([[[0] * embeddings_dim] * (input_instance.shape[1] - target_instance.shape[1])]), 
                                            axis=1)
            elif input_instance.shape[1] < target_instance.shape[1]:
                input_instance = np.append(input_instance, 
                                            np.array([[[0] * embeddings_dim] * (target_instance.shape[1] - input_instance.shape[1])]), 
                                            axis=1) 
            training_data.append([input_instance, target_instance])
        
        # Generator for training data; we need this because we pass training 
        # data one by one to fit():
        def train_generator(training_data):
            for index in range(len(training_data)):                
                yield training_data[index][0], training_data[index][1]

        # Train for number of epochs:
        for epoch in tqdm(range(epochs+1), position=0):
            history = self.model.fit(train_generator(training_data), verbose=0)
            tqdm.write("Epoch "+ str(epoch)+ ", Loss: "+ str(history.history["loss"][0]))
            
            if epoch % save_interval == 0:
                self.save(epoch)
            
    def save(self, epoch):
        """
        Save model at epoch number.
        """
        self.model.save_weights(save_model_dir + "checkpoint-" + str(epoch) + ".h5")
        
    def load(self, epoch):
        """
        Load model at epoch number.
        """
        self.model.load_weights(save_model_dir + "checkpoint-" + str(epoch) + ".h5")
        
    def inference(self,
                  input_text):
        """
        """
        vectorized_input = np.array([vectorize_sentence(input_text, self.vectorizer)])
        sequence = self.model.predict(vectorized_input, verbose=0)[0]
        sentence = ""
        
        for vector in sequence:
            sentence += embedding_to_word(vector,
                                          self.words,
                                          self.kdtree)
            sentence += " "
            
        return sentence