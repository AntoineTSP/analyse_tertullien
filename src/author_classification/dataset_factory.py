import re
import string
from collections import Counter
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from wordcloud import WordCloud


class DatasetFactory:
    def __init__(self, clf_type: str = "LinearSVC"):
        if not clf_type in ["SVC", "LinearSVC"]:
            raise ValueError("clf_type must be 'SVM' or 'LinearSVC'")
        with open("data/Stopwords_latin.txt", "r", encoding="utf-8") as stop_words_file:
            self.stop_words = stop_words_file.read().splitlines()
        self.text_apu = self.get_text_and_clean("data/author_classification/Apuleius_Flo.txt")
        self.text_ciceron = self.get_text_and_clean("data/author_classification/Cic_Pro_Milone.txt")
        self.text_de_pallio = self.get_text_and_clean("data/author_classification/De_Pallio.txt")
        self.text_apol = self.get_text_and_clean("data/author_classification/Tert_Apol.txt")
        self.nb_sequences = 100
        # Avec un modèle SVM linéaire
        if clf_type == "SVC":
            self.clf = SVC(kernel="linear", probability=True)
        if clf_type == "LinearSVC":
            base_clf = LinearSVC(max_iter=10000, C=0.1, dual="auto")
            self.clf = CalibratedClassifierCV(base_clf)

    def get_text_and_clean(self, text_path: str) -> list[str]:
        """
        Reads a text file, removes punctuation, converts to lowercase,
        removes digits, tokenizes the text, and removes stop words.

        Args:
            text_path (str): The path to the text file to be processed.

        Returns:
            str: The cleaned text as a single string.
        """

        with open(text_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Suppression de la ponctuation et réduction à la minuscule
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()

        # Suppression des nombres
        text = re.sub(r"\d+", "", text)

        # Tokenisation du texte
        words = text.split()

        words = [word for word in words if word not in self.stop_words]

        # Création d'une variable texte pour la lisibilité humaine
        text = " ".join(words)

        return text

    def create_wordcloud(self, text: str, title: str = "Word Cloud"):
        """
        Create a word cloud from the given text and display it.

        Args:
            text (str): The text to generate the word cloud from.
            title (str): The title of the word cloud.
        """
        book_mask = np.array(Image.open("./static/img/book.png"))

        wordcloud = WordCloud(
            background_color="white",
            max_words=100,
            mask=book_mask,
            contour_width=3,
            contour_color="steelblue",
            width=400,  # Further reduce width for smaller resolution
            height=400,  # Further reduce height for smaller resolution
        ).generate(text)

        plt.figure(figsize=(12, 6))  # Adjust figure size for better display
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.savefig(
            f"./plot/{title.replace(' ', '_')}.png", dpi=300
        )  # Save with higher DPI for better quality

    def get_dataframe_dataset(self) -> pd.DataFrame:
        """
        Creates a DataFrame containing text excerpts from different authors.
        Each excerpt is wrapped to ensure it fits within a specified number of sequences.

        Returns:
            pd.DataFrame: A DataFrame with two columns: 'text' and 'author'.
        """

        # On crée des listes contenant des extraits de texte
        phrases_apu = wrap(self.text_apu, len(self.text_apu) // self.nb_sequences)
        phrases_ciceron = wrap(self.text_ciceron, len(self.text_ciceron) // self.nb_sequences)
        phrases_de_pallio = wrap(self.text_de_pallio, len(self.text_de_pallio) // self.nb_sequences)
        phrases_apol = wrap(self.text_apol, len(self.text_apol) // self.nb_sequences)

        # On crée une liste qui contient le nom de l'auteur des extraits en question
        auteur = np.array(["Apulée"] * len(phrases_apu))
        auteur = np.append(auteur, np.array(["Cicéron"] * len(phrases_ciceron)))
        auteur = np.append(auteur, np.array(["Tertullien_pallio"] * len(phrases_de_pallio)))
        auteur = np.append(auteur, np.array(["Tertullien_apologeticum"] * len(phrases_apol)))
        d = {"text": phrases_apu + phrases_ciceron + phrases_de_pallio + phrases_apol, "author": auteur}
        spooky_df = pd.DataFrame(d)
        return spooky_df

    def encode_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Encode the 'author' column in the DataFrame using Label Encoding.

        Args:
            df (pd.DataFrame): The DataFrame containing the 'author' column.

        Returns:
            pd.DataFrame: The DataFrame with an additional 'author_encoded' column.
            LabelEncoder: The fitted LabelEncoder instance.
        """

        le = LabelEncoder()
        df["author_encoded"] = le.fit_transform(df["author"])
        return df, le

    def get_labels_classes(self, le: LabelEncoder) -> np.array:
        return le.classes_

    def split_train_test(self, df: pd.DataFrame, test_size: float = 0.2) -> tuple:
        """
        Split the DataFrame into training and testing sets.

        Args:
            df (pd.DataFrame): The DataFrame to split.
            test_size (float): The proportion of the dataset to include in the test split.

        Returns:
            tuple: A tuple containing the training and testing sets for both features and labels.
        """

        X_train, X_test, y_train, y_test = train_test_split(
            df["text"].values,
            df["author_encoded"].values,
            test_size=test_size,
            random_state=33,
            stratify=df["author_encoded"].values,
        )
        return X_train, X_test, y_train, y_test

    def get_label_to_author(self, le: LabelEncoder, y_train: np.array) -> dict:
        """
        Create a mapping from encoded labels to author names.

        Args:
            le (LabelEncoder): The fitted LabelEncoder instance.
            y_train (np.ndarray): The training labels.

        Returns:
            dict: A dictionary mapping encoded labels to author names.
        """

        label_to_author = {y_train[i]: le.inverse_transform([y_train[i]])[0] for i in range(len(y_train))}
        return label_to_author

    def get_proportion_label_in_sets(self, y_train: np.ndarray, y_test: np.ndarray, le: LabelEncoder) -> dict:
        """
        Calculate the proportion of each author in the training and testing sets.

        Args:
            y_train (np.ndarray): The training labels.
            y_test (np.ndarray): The testing labels.
            le (LabelEncoder): The fitted LabelEncoder instance.

        Returns:
            dict: A dictionary containing the proportion of each author in both training and testing sets.
        """

        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        proportion_author = {
            le.inverse_transform([label])[0]: {
                "train": 100 * np.sum(y_train == label) / len(y_train),
                "test": 100 * np.sum(y_test == label) / len(y_test),
            }
            for label in unique_labels
        }
        return proportion_author

    def plot_top_words(self, initials, ax, X_train, y_train, le, n_words=12):
        """
        Plot the top n most frequent words used by a specific author.

        Args:
            initials (str): The initials or label of the author.
            ax (matplotlib.axes.Axes): The axis to plot on.
            X_train (np.ndarray): Training text data.
            y_train (np.ndarray): Training labels.
            le (LabelEncoder): Label encoder for decoding labels.
            n_words (int): Number of top words to display.
        """
        # Map initials to author names
        initials_to_author = {
            "Apulée": "Apulée",
            "Cicéron": "Cicéron",
            "Tertullien_pallio": "Tertullien (De Pallio)",
            "Tertullien_apologeticum": "Tertullien (Apologeticum)",
        }

        # Filter texts for the given author
        author_texts = X_train[y_train == le.transform([initials])[0]]
        all_tokens = " ".join(author_texts).split()
        counts = Counter(all_tokens)

        # Get the top n words and their counts
        top_words = [word[0] for word in counts.most_common(n_words)]
        top_words_counts = [word[1] for word in counts.most_common(n_words)]

        # Generate a color palette
        colors = sns.color_palette("husl", n_colors=n_words)

        # Plot the bar chart
        sns.barplot(ax=ax, x=top_words, y=top_words_counts, palette=colors, hue=top_words, legend=False)
        ax.set_title(f"Mots les plus fréquents chez {initials_to_author[initials]}")
        ax.set_xlabel("Mots")
        ax.set_ylabel("Fréquence")

    def plot_all_authors_top_words(self, X_train, y_train, le, n_words=12):
        """
        Plot the top n most frequent words for all authors.

        Args:
            X_train (np.ndarray): Training text data.
            y_train (np.ndarray): Training labels.
            le (LabelEncoder): Label encoder for decoding labels.
            n_words (int): Number of top words to display for each author.
        """
        authors = ["Apulée", "Cicéron", "Tertullien_pallio", "Tertullien_apologeticum"]
        fig, axs = plt.subplots(len(authors), 1, figsize=(12, 12))

        for i, author in enumerate(authors):
            self.plot_top_words(author, ax=axs[i], X_train=X_train, y_train=y_train, le=le, n_words=n_words)

        plt.tight_layout()
        plt.savefig("./plot/Distribution_mots.png")
        plt.show()

    def fit_vectorizers(self, vectorizer, x_train, y_train) -> GridSearchCV:
        """
        Fit a vectorizer and train a linear SVM model using GridSearchCV.

        Args:
            vectorizer (callable): A callable that returns a vectorizer instance.
            x_train (np.ndarray): Training text data.
            y_train (np.ndarray): Training labels.

        Returns:
            GridSearchCV: The fitted GridSearchCV object.
        """
        print(f"Fitting vectorizer {vectorizer.__name__}...")
        pipeline = Pipeline(
            [
                ("vect", vectorizer()),
                ("scaling", StandardScaler(with_mean=False)),
                ("clf", self.clf),
            ]
        )

        parameters = {
            "vect__ngram_range": (
                (1, 1),
                (1, 2),
            ),  # unigrams or bigrams
            "vect__stop_words": (
                "english",
                None,
            ),
        }

        grid_search = GridSearchCV(
            pipeline,
            parameters,
            scoring="f1_micro",
            cv=4,
            n_jobs=4,
            verbose=1,
        )
        grid_search.fit(x_train, y_train)

        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print(f"\t{param_name}: {best_parameters[param_name]!r}")

        print(f"CV scores {grid_search.cv_results_['mean_test_score']}")
        print(f"Mean F1 {np.mean(grid_search.cv_results_['mean_test_score'])}")

        return grid_search

    def fit_w2v_avg(self, w2v_vectors, x_train_tokens, y_train):
        def get_mean_vector(w2v_vectors, words):
            words = [word for word in words if word in w2v_vectors]
            if words:
                avg_vector = np.mean([w2v_vectors[word] for word in words], axis=0)
            else:
                try:
                    vector_size = w2v_vectors.vector_size
                except AttributeError:
                    vector_size = w2v_vectors.shape[1]
                avg_vector = np.zeros(vector_size)
            return avg_vector

        X_train_vectors = np.array([get_mean_vector(w2v_vectors, words) for words in x_train_tokens])

        scores = cross_val_score(self.clf, X_train_vectors, y_train, cv=4, scoring="f1_micro", n_jobs=4)
        print("Fitting Word2Vec average vectors...")
        print(f"CV scores {scores}")
        print(f"Mean F1 {np.mean(scores)}")
        self.clf.fit(X_train_vectors, y_train)
        return self.clf, scores

    def predict_author_from_text_with_bag(self, text: str, cv_bow: GridSearchCV, le: LabelEncoder) -> str:
        """
        Predict the author of a given text using a fitted Bag of Words model.

        Args:
            text (str): The text to classify.
            cv_bow (GridSearchCV): The fitted Bag of Words model.
            le (LabelEncoder): The label encoder for decoding labels.

        Returns:
            str: The predicted author.
        """
        # Directly use the full pipeline:
        prediction = cv_bow.predict([text])
        probs = cv_bow.predict_proba([text])
        predicted_author = le.inverse_transform(prediction)[0]
        return predicted_author, {le.inverse_transform([i])[0]: prob for i, prob in enumerate(probs[0])}

    def predict_author_from_text_w2v(self, text: str, w2v_vectors, le: LabelEncoder, clf: LinearSVC) -> str:
        """
        Predict the author of a given text using averaged Word2Vec vectors.

        Args:
            text (str): The text to classify.
            w2v_vectors: Trained Word2Vec KeyedVectors.
            le (LabelEncoder): The label encoder for decoding labels.

        Returns:
            str: The predicted author.
        """
        tokens = text.split()
        words = [word for word in tokens if word in w2v_vectors]
        if words:
            avg_vector = np.mean([w2v_vectors[word] for word in words], axis=0)
        else:
            avg_vector = np.zeros(w2v_vectors.vector_size)

        # Predict
        prediction = clf.predict(avg_vector.reshape(1, -1))
        probs = clf.predict_proba(avg_vector.reshape(1, -1))
        predicted_author = le.inverse_transform(prediction)[0]
        return predicted_author, {le.inverse_transform([i])[0]: prob for i, prob in enumerate(probs[0])}
