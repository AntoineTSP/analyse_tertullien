import re
import string
from textwrap import wrap

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DatasetFactory:
    def __init__(self):
        with open("data/Stopwords_latin.txt", "r", encoding="utf-8") as stop_words_file:
            self.stop_words = stop_words_file.read().splitlines()
        print(f"Stop words loaded: {len(self.stop_words)}")
        self.text_apu = self.get_text_and_clean("data/author_classification/Apuleius_Flo.txt")
        self.text_ciceron = self.get_text_and_clean("data/author_classification/Cic_Pro_Milone.txt")
        self.text_de_pallio = self.get_text_and_clean("data/author_classification/De_Pallio.txt")
        self.text_apol = self.get_text_and_clean("data/author_classification/Tert_Apol.txt")
        self.nb_sequences = 100

    def get_text_and_clean(self, text_path: str) -> list[str]:
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

    def get_dataframe_dataset(self):

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
        Encode a categorical variable into numerical values.
        """
        le = LabelEncoder()
        df["author_encoded"] = le.fit_transform(df["author"])
        return df, le

    def get_labels_classes(self, le: LabelEncoder) -> np.array:
        return le.classes_
