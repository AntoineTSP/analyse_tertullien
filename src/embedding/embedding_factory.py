import os

import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from embedding.callback import TrainingLogger


class EmbeddingFactory:

    def __init__(self, paragraphs: list[list[str]], model: Model = None):
        self.min_word_frequency = 5
        self.tokenizer = Tokenizer()
        self.embedding_dim = 10
        self.first_dim_output_lstm = 32
        self.second_dim_output_lstm = 8
        self.tokenizer.fit_on_texts(paragraphs)

        # Filter out less frequent words
        word_counts = self.tokenizer.word_counts
        self.tokenizer.word_index = {
            word: index
            for word, index in self.tokenizer.word_index.items()
            if word_counts[word] >= self.min_word_frequency
        }
        self.tokenizer.word_index[self.tokenizer.oov_token] = (
            len(self.tokenizer.word_index) + 1
        )  # Add "unknown" token

        self.sequences = self.tokenizer.texts_to_sequences(paragraphs)

        self.padded_sequences = sequence.pad_sequences(
            self.sequences,  # Use our train dataset
            maxlen=max(len(seq) for seq in self.sequences),  # Max length
            dtype="int32",
            padding="post",  # Add zeros if needed to the max length
            truncating="post",  # Truncate sequence that are too long
            value=0.0,  # Value to add at the end of the sequence if it is too short
        )
        self.labels = np.random.randint(2, size=(len(self.padded_sequences),))
        self.nb_subjected_to_embedding = len(self.tokenizer.word_index) + 1
        self.model = model

    def divide_train_test(self, padded_sequences: np.ndarray, labels: np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, labels, test_size=0.33, random_state=46
        )
        return X_train, X_test, y_train, y_test

    def define_lstm_model(self) -> Model:
        # --- Using the Sequential API
        model = Sequential()
        model.add(Embedding(self.nb_subjected_to_embedding, self.embedding_dim))  # To embed the inputs
        model.add(
            LSTM(self.first_dim_output_lstm, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
        )  # Use a LSTM with dropout
        model.add(
            LSTM(self.second_dim_output_lstm, dropout=0.2, recurrent_dropout=0.2)
        )  # Use a LSTM with dropout
        model.add(Dense(1, activation="sigmoid"))  # Perform a binary classification

        # Loss = BCE | Optimizer = Adam | Optimisation selon l'accuracy
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.array,
        y_test: np.array,
        model_name: str,
        epochs: int = 5,
        batch_size: int = 64,
    ):
        if self.model is None:
            raise ValueError("Model is not defined. Please call define_lstm_model() first.")

        logger = TrainingLogger()

        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[logger],
        )
        # Sauvegarde le model
        self.model.save(os.path.join("./data/models", f"{model_name}.keras"))

        training_logs = logger.logs

        return self.model, training_logs

    def load_model(self, model_name: str) -> Model:
        model_path = os.path.join("./data/models", f"{model_name}.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
        self.model = load_model("test_lemma.h5")
        return self.model

    def get_accuracy(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        if self.model is None:
            raise ValueError("Model is not defined. Please call load_model() first.")
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Accuracy: {scores[1] * 100:.2f}%")
        return scores[1]

    def get_word_at_index(self, index: int) -> str:
        # Get the word_index dictionary
        word_index = self.tokenizer.word_index
        word_at_index = next(word for word, index in word_index.items() if index == index)

        print(f"The word at index {index} is: {word_at_index}")
        return word_at_index

    def extract_closest_word(self, word):
        word_index = self.tokenizer.word_index
        index_of_word = word_index.get(word, None)

        if index_of_word is not None:
            print(f"The index of '{word}' is: {index_of_word}")
        else:
            print(f"'{word}' not found in the vocabulary.")

        # Get the embedding matrix
        embedding_matrix = self.model.layers[0].get_weights()[0]
        # word_embedding = embedding_matrix[index_of_word, :]  # Get the embedding of the word

        dist_v = tf.norm(
            embedding_matrix - embedding_matrix[index_of_word], axis=1
        )  # euclidean norm by default

        closest_words = [
            (next(word for word, index in word_index.items() if index == i), dist_v[i].numpy())
            for i in np.argsort(dist_v)[1:20]
        ]
        return closest_words

    def get_new_embedding(self, positive_words: list[str], negative_words: list[str]) -> np.ndarray:
        word_index = self.tokenizer.word_index
        embedding_matrix = self.model.layers[0].get_weights()[0]

        # Initialize the resulting embedding as a zero vector
        result_embedding = np.zeros(self.embedding_dim)

        # Add embeddings of positive words
        for word in positive_words:
            index = word_index.get(word)
            if index is not None:
                result_embedding += embedding_matrix[index]
            else:
                print(f"'{word}' not found in the vocabulary.")

        # Subtract embeddings of negative words
        for word in negative_words:
            index = word_index.get(word)
            if index is not None:
                result_embedding -= embedding_matrix[index]
            else:
                print(f"'{word}' not found in the vocabulary.")

        return result_embedding

    def find_top_10_nearest_words(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        word_index = self.tokenizer.word_index
        embedding_matrix = self.model.layers[0].get_weights()[0]

        # Compute distances between the given embedding and all embeddings in the matrix
        distances = tf.norm(embedding_matrix - embedding, axis=1)

        # Get the indices of the 10 smallest distances (excluding the first one if it's the same word)
        nearest_indices = np.argsort(distances.numpy())[:10]

        # Map indices back to words and return the top-10 nearest words with their distances
        nearest_words = [
            (word, distances[i].numpy())
            for i in nearest_indices
            for word, index in word_index.items()
            if index == i
        ]
        return nearest_words
