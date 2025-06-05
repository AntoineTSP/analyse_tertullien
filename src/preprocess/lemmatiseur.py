import csv
import os
import re
import string


class PyrrhaLemmatiseur:
    def __init__(self, tsv_folder_path: str):
        self.tsv_folder_path = tsv_folder_path
        self.text = ""
        self.nb_words = 0
        self.length_sequence = 20
        self.folder_output = "data/Pyrrha_output"
        with open("./data/Stopwords_latin.txt", "r", encoding="utf-8") as stop_words_file:
            self.stop_words = stop_words_file.read().splitlines()

    def obtain_corpus_lemma(self, output_file: str) -> str:
        for filename in sorted(os.listdir(self.tsv_folder_path)):
            if filename.endswith(".tsv"):
                file_path = os.path.join(self.tsv_folder_path, filename)
                print(file_path)
                with open(file_path, "r", newline="", encoding="utf-8") as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter="\t")
                    # I go to the next line to skip the header
                    next(csv_reader, None)
                    for row in csv_reader:
                        if self.nb_words % self.length_sequence == 0:
                            self.text += "\n"
                            self.text += row[1]
                            self.text += " "
                            self.nb_words += 1
                        else:
                            self.text += row[1]
                            self.text += " "
                            self.nb_words += 1
            else:
                raise ValueError(f"Le fichier {filename} n'est pas au format .tsv")

        if not os.path.isdir(self.folder_output):
            os.makedirs(self.folder_output)

        output_file_path = os.path.join(self.folder_output, f"{output_file}.txt")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(self.text + "\n")

    def obtain_list_sequences(self, raw_text_lemma_file: str) -> list[list[str]]:
        paragraphs = []
        with open(
            os.path.join(self.folder_output, f"{raw_text_lemma_file}.txt"), "r", encoding="utf-8"
        ) as file:
            paragraphs = [[paragraph.strip()] for paragraph in file.readlines()]

        paragraphs = [paragraph[0] for paragraph in paragraphs if paragraph[0]]
        paragraphs = [[paragraph.strip()] for paragraph in paragraphs]

        data = []
        for i, paragraph in enumerate(paragraphs):
            for j, sentence in enumerate(paragraph):
                text = sentence.translate(str.maketrans("", "", string.punctuation))
                text = text.lower()
                text = re.sub(r"\d+", "", text)
                words = text.split()
                words = [word for word in words if word not in self.stop_words]
                data.append(words)

        with open(os.path.join(self.folder_output, "corpus_clean.txt"), "w", encoding="utf-8") as file:
            for inner_list in data:
                file.write(" ".join(inner_list) + "\n")

        return data
