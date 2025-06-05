from fastapi import FastAPI, Request

from embedding.embedding_factory import EmbeddingFactory
from preprocess.lemmatiseur import PyrrhaLemmatiseur

app = FastAPI()


@app.get("/")
def dashboard(request: Request):
    # -----Preprocess-----
    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path="data/Pyrrha")
    pyrrha_lemmatiseur.obtain_corpus_lemma(output_file="raw_text_lemma")
    print("Corpus lemmatized successfully and saved to test_corpus_lemma.txt")
    data = pyrrha_lemmatiseur.obtain_list_sequences(raw_text_lemma_file="raw_text_lemma")
    # -----Embedding-----
    embedding_factory = EmbeddingFactory()
    padded_sequences = embedding_factory.convert_paragraphs_to_sequences(paragraphs=data)
    # return {"padded_sequences": [seq.tolist() for seq in padded_sequences[:3]]}
    labels = embedding_factory.generate_random_labels(padded_sequences=padded_sequences)
    X_train, X_test, y_train, y_test = embedding_factory.divide_train_test(
        padded_sequences=padded_sequences, labels=labels
    )
    embedding_factory.model = embedding_factory.define_lstm_model(
        nb_words_subjected_to_embedding=embedding_factory._get_number_of_words_subjected_to_embedding()
    )

    embedding_factory.model = embedding_factory.fit_model(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_name="test"
    )
    return {
        "accuracy": embedding_factory.get_accuracy(X_test=X_test, y_test=y_test),
        "next_word_tomarcion": embedding_factory.extract_closest_word("marcion"),
    }
