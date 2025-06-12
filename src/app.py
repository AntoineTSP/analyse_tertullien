import os
from pathlib import Path

import gensim.downloader
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from author_classification.dataset_factory import DatasetFactory
from embedding.embedding_factory import EmbeddingFactory
from preprocess.lemmatiseur import PyrrhaLemmatiseur
from utils.converter import (
    add_title_to_spans,
    get_all_annotations,
    get_model_summary,
    parse_full_tei,
)

app = FastAPI()
app.mount("/img", StaticFiles(directory="static/img"), name="img")

# Serve the HTML template
templates = Jinja2Templates(directory="templates")

# ----------------------------------------------------------------------------------------
# Welcome page
# ----------------------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def welcome_page(request: Request):
    context = {
        "request": request,
        "title": "Explore Tertullien like never before",
        "description": "This is a simple welcome page that points to every resources.",
        "credit_images": [
            {"src": "/img/tertullien.jpeg", "alt": "Image 1"},
            {"src": "/img/scorpiace.jpg", "alt": "Image 2"},
        ],
        "buttons": [
            {"name": "Go to Route Health", "url": "/health"},
            {"name": "Go to Route Embedding Section", "url": "/embedding"},
            {"name": "Go to Route Text Annotation Section", "url": "/text_annotation"},
            {"name": "Go to Route Author Classification", "url": "/author_classification"},
        ],
    }
    return templates.TemplateResponse("welcome.html", context)


# ----------------------------------------------------------------------------------------
# Embedding Section
# ----------------------------------------------------------------------------------------


@app.get("/embedding", response_class=HTMLResponse)
async def embedding_page(request: Request):
    context = {
        "request": request,
        "title": "Explore Tertullien like never before",
        "description": "Embedding part: the goal is to explore a new way to manipulate Tertulien's semantic.",
        "credit_images": [
            {"src": "/img/tertullien.jpeg", "alt": "Image 1"},
            {"src": "/img/scorpiace.jpg", "alt": "Image 2"},
        ],
        "buttons": [
            {"name": "Go to Route Welcome", "url": "/"},
            {"name": "Go to Route Get Corpus", "url": "/embedding/get_corpus"},
            {"name": "Go to Route Fit Model", "url": "/embedding/fit"},
            {"name": "Go to Route Model JSON", "url": "/embedding/model_json"},
            {"name": "Go to Route Model Summary", "url": "/embedding/model_summary"},
            {"name": "Go to Route New Embedding", "url": "/embedding/nem_embedding"},
        ],
    }
    return templates.TemplateResponse("welcome.html", context)


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running successfully."}


@app.get("/embedding/get_corpus")
async def get_corpus(request: Request):
    # -----Preprocess-----
    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path="data/Pyrrha")
    pyrrha_lemmatiseur.obtain_corpus_lemma(output_file="raw_text_lemma")
    data = pyrrha_lemmatiseur.obtain_list_sequences(raw_text_lemma_file="raw_text_lemma")
    return {"corpus": data}


@app.get("/embedding/fit")
async def dashboard(request: Request):
    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path="data/Pyrrha")
    data = pyrrha_lemmatiseur.obtain_list_sequences(raw_text_lemma_file="raw_text_lemma")
    # -----Embedding-----
    embedding_factory = EmbeddingFactory(paragraphs=data)
    labels = embedding_factory.labels
    padded_sequences = embedding_factory.padded_sequences
    X_train, X_test, y_train, y_test = embedding_factory.divide_train_test(
        padded_sequences=padded_sequences, labels=labels
    )
    embedding_factory.model = embedding_factory.define_lstm_model()

    embedding_factory.model, logs = embedding_factory.fit_model(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_name="test"
    )
    return {
        "accuracy": embedding_factory.get_accuracy(X_test=X_test, y_test=y_test),
        "training_logs": logs,
    }


@app.get("/embedding/model_json")
async def get_model_json(request: Request):
    if os.path.exists("data/models/test.keras"):
        model = load_model("data/models/test.keras")
        model_json = model.to_json()
        return JSONResponse(content=model_json)
    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path="data/Pyrrha")
    data = pyrrha_lemmatiseur.obtain_list_sequences(raw_text_lemma_file="raw_text_lemma")
    # -----Embedding-----
    embedding_factory = EmbeddingFactory(paragraphs=data)
    # return {"padded_sequences": [seq.tolist() for seq in padded_sequences[:3]]}
    labels = embedding_factory.labels
    padded_sequences = embedding_factory.padded_sequences
    X_train, X_test, y_train, y_test = embedding_factory.divide_train_test(
        padded_sequences=padded_sequences, labels=labels
    )
    embedding_factory.model = embedding_factory.define_lstm_model()

    embedding_factory.model, _ = embedding_factory.fit_model(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_name="test"
    )
    model_json = embedding_factory.model.to_json()
    return JSONResponse(content=model_json)


@app.get("/embedding/model_summary", response_class=HTMLResponse)
async def model_summary(request: Request):
    if os.path.exists("data/models/test.keras"):
        model = load_model("data/models/test.keras")
        summary = get_model_summary(model)
        return templates.TemplateResponse("summary.html", {"request": request, "summary": summary})
    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path="data/Pyrrha")
    data = pyrrha_lemmatiseur.obtain_list_sequences(raw_text_lemma_file="raw_text_lemma")
    # -----Embedding-----
    embedding_factory = EmbeddingFactory(paragraphs=data)
    labels = embedding_factory.labels
    padded_sequences = embedding_factory.padded_sequences
    X_train, X_test, y_train, y_test = embedding_factory.divide_train_test(
        padded_sequences=padded_sequences, labels=labels
    )
    embedding_factory.model = embedding_factory.define_lstm_model()

    embedding_factory.model, _ = embedding_factory.fit_model(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_name="test"
    )
    summary = get_model_summary(embedding_factory.model)
    # return f"<pre>{summary}</pre>"
    return templates.TemplateResponse("summary.html", {"request": request, "summary": summary})


@app.get("/embedding/nem_embedding")
async def new_embedding(request: Request):
    if os.path.exists("data/models/test.keras"):
        model = load_model("data/models/test.keras")
    else:
        raise FileNotFoundError("Model not found. Please train the model first.")

    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path="data/Pyrrha")
    data = pyrrha_lemmatiseur.obtain_list_sequences(raw_text_lemma_file="raw_text_lemma")
    # -----Embedding-----
    embedding_factory = EmbeddingFactory(paragraphs=data, model=model)

    new_embedding = embedding_factory.get_new_embedding(
        positive_words=["marcion", "cupiditas"], negative_words=["apocalypsis"]
    )
    return {
        "new_embedding": new_embedding.tolist(),
        "top_10words": embedding_factory.find_top_10_nearest_words(embedding=new_embedding),
    }


# ----------------------------------------------------------------------------------------
# Text Annotation Section
# ----------------------------------------------------------------------------------------


@app.get("/text_annotation", response_class=HTMLResponse)
async def text_annotation_page(request: Request):
    context = {
        "request": request,
        "title": "Explore Tertullien like never before",
        "description": "Text Annotation section: find all texts annotated.",
        "credit_images": [
            {"src": "/img/tertullien.jpeg", "alt": "Image 1"},
            {"src": "/img/scorpiace.jpg", "alt": "Image 2"},
        ],
        "buttons": [
            {"name": "Go to Welcome page", "url": "/"},
            {"name": "Go to Route XML", "url": "/text_annotation/xml"},
            {"name": "Go to Route XML to HTML", "url": "/text_annotation/xml_to_html"},
            {"name": "Go to Route Get Annotation", "url": "/text_annotation/get_annotation"},
            {"name": "Go to Route Get Some Annotation", "url": "/text_annotation/get_some_annotation"},
        ],
    }
    return templates.TemplateResponse("welcome.html", context)


@app.get("/text_annotation/xml", response_class=Response)
async def get_xml():
    xml_path = Path("./data/xml/Pall2.xml")
    if not xml_path.exists():
        return Response(content="File not found", status_code=404)

    content = xml_path.read_text(encoding="utf-8")
    return Response(content=content, media_type="application/xml")


@app.get("/text_annotation/xml_to_html", response_class=HTMLResponse)
async def render_full2_tei(request: Request):
    data = parse_full_tei("./data/xml/Pall2.xml")  # your function generating html string
    data["body_html"] = add_title_to_spans(data["body_html"])
    return templates.TemplateResponse("xml.html", {"request": request, **data})


@app.get("/text_annotation/get_annotation")
async def get_annotation(request: Request):
    annotations = get_all_annotations(xml_path="./data/xml/Pall2")
    return JSONResponse(content=annotations)


@app.get("/text_annotation/get_some_annotation")
async def get_some_annotation(request: Request):
    annotations = get_all_annotations(xml_path="./data/xml/Pall2")
    annotations = annotations.get("metaphore", [])
    return JSONResponse(content=annotations)


# ----------------------------------------------------------------------------------------
# Author Classification Section
# ----------------------------------------------------------------------------------------


@app.get("/author_classification", response_class=HTMLResponse)
async def author_classification(request: Request):
    context = {
        "request": request,
        "title": "Explore Tertullien like never before",
        "description": "Text Annotation section: find all texts annotated.",
        "credit_images": [
            {"src": "/img/tertullien.jpeg", "alt": "Image 1"},
            {"src": "/img/scorpiace.jpg", "alt": "Image 2"},
        ],
        "buttons": [
            {"name": "Go to Welcome page", "url": "/"},
            {"name": "Get Stopwords", "url": "/author_classification/get_stopwords"},
            {"name": "Get Author Clean Text", "url": "/author_classification/get_clean_author_text"},
            # {"name": "Get Wordcloud", "url": "/author_classification/get_wordcloud"},
            {"name": "Plot Wordcloud", "url": "/author_classification/plot_wordclouds"},
            {"name": "Get Dataframe", "url": "/author_classification/get_df"},
            {"name": "Get Labels Classes", "url": "/author_classification/get_labels_classes"},
            {"name": "Get Author Repartition", "url": "/author_classification/get_author_repartition"},
            {"name": "Plot Top words", "url": "/author_classification/plot_top_words"},
            {"name": "Get Accuracy for each model", "url": "/author_classification/get_accuracy"},
            {"name": "Predict author from text", "url": "/author_classification/predict_author"},
        ],
    }
    return templates.TemplateResponse("welcome.html", context)


@app.get("/author_classification/get_stopwords")
async def get_stopwords(request: Request):
    dataset_factory = DatasetFactory()
    stop_words = dataset_factory.stop_words
    return JSONResponse(
        content={
            "Nb_stopwords": len(stop_words),
            "Stopwords list": stop_words,
        }
    )


@app.get("/author_classification/get_clean_author_text")
async def get_clean_author_text(request: Request):
    dataset_factory = DatasetFactory()
    return JSONResponse(
        content={
            "Text Apulée": dataset_factory.text_apu,
            "Text Cicéron": dataset_factory.text_ciceron,
            "Text De Pallio": dataset_factory.text_de_pallio,
            "Text Apologeticum": dataset_factory.text_apol,
        }
    )


@app.get("/author_classification/get_wordcloud")
async def get_wordcloud(request: Request):
    dataset_factory = DatasetFactory()
    dataset_factory.create_wordcloud(text=dataset_factory.text_apu, title="Wordcloud of Apulée")
    dataset_factory.create_wordcloud(text=dataset_factory.text_ciceron, title="Wordcloud of Cicéron")
    dataset_factory.create_wordcloud(text=dataset_factory.text_de_pallio, title="Wordcloud of De Pallio")
    dataset_factory.create_wordcloud(text=dataset_factory.text_apol, title="Wordcloud of Apologeticum")
    return JSONResponse(
        content={
            "Success": "Wordclouds created for each author.",
        }
    )


@app.get("/author_classification/plot_wordclouds")
async def plot_wordclouds(request: Request):
    return templates.TemplateResponse("display_wordclouds.html", {"request": request})


@app.get("/author_classification/get_df")
async def get_df(request: Request):
    dataset_factory = DatasetFactory()
    df = dataset_factory.get_dataframe_dataset()
    df, le = dataset_factory.encode_variable(df)
    return JSONResponse(content=df.to_dict(orient="records"))


@app.get("/author_classification/get_labels_classes")
async def get_labels_classes(request: Request):
    dataset_factory = DatasetFactory()
    df = dataset_factory.get_dataframe_dataset()
    df, le = dataset_factory.encode_variable(df)
    print(df[98:110].head())
    print(dataset_factory.get_labels_classes(le))
    X_train, X_test, y_train, y_test = dataset_factory.split_train_test(df)
    label_to_author = dataset_factory.get_label_to_author(le=le, y_train=y_train)
    label_to_author = {int(k): v for k, v in label_to_author.items()}
    return JSONResponse(content=label_to_author)


@app.get("/author_classification/get_author_repartition")
async def get_author_repartition(request: Request):
    dataset_factory = DatasetFactory()
    df = dataset_factory.get_dataframe_dataset()
    df, le = dataset_factory.encode_variable(df)
    X_train, X_test, y_train, y_test = dataset_factory.split_train_test(df)
    proportion = dataset_factory.get_proportion_label_in_sets(y_train=y_train, y_test=y_test, le=le)
    return JSONResponse(content=proportion)


@app.get("/author_classification/plot_top_words")
async def plot_top_words(request: Request):
    return templates.TemplateResponse("display.html", {"request": request})


@app.get("/author_classification/get_accuracy")
async def get_accuracy(request: Request):
    dataset_factory = DatasetFactory(clf_type="LinearSVC")
    df = dataset_factory.get_dataframe_dataset()
    df, le = dataset_factory.encode_variable(df)
    X_train, X_test, y_train, y_test = dataset_factory.split_train_test(df)
    proportion = dataset_factory.get_proportion_label_in_sets(y_train=y_train, y_test=y_test, le=le)
    dataset_factory.plot_all_authors_top_words(X_train=X_train, y_train=y_train, le=le, n_words=12)
    # Bag-of-words
    cv_bow = dataset_factory.fit_vectorizers(CountVectorizer, x_train=X_train, y_train=y_train)
    # TF-IDF
    cv_tfidf = dataset_factory.fit_vectorizers(TfidfVectorizer, x_train=X_train, y_train=y_train)
    # Word2Vec
    X_train_tokens = [text.split() for text in X_train]
    w2v_model = Word2Vec(X_train_tokens, vector_size=200, window=5, min_count=1, workers=4)
    clf_w2vec, cv_w2vec = dataset_factory.fit_w2v_avg(
        w2v_model.wv, x_train_tokens=X_train_tokens, y_train=y_train
    )
    # Word2Vec pre-trained
    glove_model_path = "data/models/glove-wiki-gigaword-200.model"
    if not os.path.exists(glove_model_path):
        glove_model = gensim.downloader.load("glove-wiki-gigaword-200")
        glove_model.save(glove_model_path)
    else:
        glove_model = gensim.models.KeyedVectors.load(glove_model_path)
    clf_w2vec_transfert, cv_w2vec_transfert = dataset_factory.fit_w2v_avg(
        glove_model, x_train_tokens=X_train_tokens, y_train=y_train
    )

    return JSONResponse(
        content={
            "Bag-of-Words": np.mean(cv_bow.cv_results_["mean_test_score"]),
            "TF-IDF": np.mean(cv_tfidf.cv_results_["mean_test_score"]),
            "Word2Vec non pré-entraîné": np.mean(cv_w2vec),
            "Word2Vec pré-entraîné": np.mean(cv_w2vec_transfert),
            "Hyperparameters": {
                "Classifier": str(dataset_factory.clf),
            },
        }
    )


@app.get("/author_classification/predict_author")
async def predict_author(request: Request):
    dataset_factory = DatasetFactory(clf_type="LinearSVC")
    df = dataset_factory.get_dataframe_dataset()
    df, le = dataset_factory.encode_variable(df)
    X_train, X_test, y_train, y_test = dataset_factory.split_train_test(df)
    proportion = dataset_factory.get_proportion_label_in_sets(y_train=y_train, y_test=y_test, le=le)
    dataset_factory.plot_all_authors_top_words(X_train=X_train, y_train=y_train, le=le, n_words=12)
    # Bag-of-words
    cv_bow = dataset_factory.fit_vectorizers(CountVectorizer, x_train=X_train, y_train=y_train)
    # TF-IDF
    cv_tfidf = dataset_factory.fit_vectorizers(TfidfVectorizer, x_train=X_train, y_train=y_train)
    # Word2Vec
    X_train_tokens = [text.split() for text in X_train]
    w2v_model = Word2Vec(X_train_tokens, vector_size=200, window=5, min_count=1, workers=4)
    clf_w2vec, cv_w2vec = dataset_factory.fit_w2v_avg(
        w2v_model.wv, x_train_tokens=X_train_tokens, y_train=y_train
    )
    # Word2Vec pre-trained
    glove_model_path = "data/models/glove-wiki-gigaword-200.model"
    if not os.path.exists(glove_model_path):
        glove_model = gensim.downloader.load("glove-wiki-gigaword-200")
        glove_model.save(glove_model_path)
    else:
        glove_model = gensim.models.KeyedVectors.load(glove_model_path)
    clf_w2vec_transfer, cv_w2vec_transfert = dataset_factory.fit_w2v_avg(
        glove_model, x_train_tokens=X_train_tokens, y_train=y_train
    )

    text_to_predict = "crura prodigae nec intra genua inverecundae nec brachiis parcae"
    # text_to_predict = "nihil causa sua deprecatur condicione miratur scit se peregrinam terris agere inter extraneos facile inimicos invenire ceterum genus sedem spem gratiam dignitatem caelis habere unum gestit interdum ne ignorata damnetur quid hic deperit legibus suo regno dominantibus audiatur an magis gloriabitur potestas eorum quo auditam damnabunt veritatem ceterum inauditam damnent praeter invidiam iniquitatis suspicionem merebuntur alicuius conscientiae nolentes audire auditum damnare possint hanc itaque primam causam apud vos collocamus iniquitatis odii erga nomen christianorum iniquitatem idem"
    # text_to_predict = "j'aime les pommes et les poires, mais je préfère les pommes."

    predic_bow, probs_bow = dataset_factory.predict_author_from_text_with_bag(
        text=text_to_predict, cv_bow=cv_bow, le=le
    )
    predic_tf_idf, probs_tf_idf = dataset_factory.predict_author_from_text_with_bag(
        text=text_to_predict, cv_bow=cv_tfidf, le=le
    )

    predic_w2c, probs_w2vec = dataset_factory.predict_author_from_text_w2v(
        text=text_to_predict, w2v_vectors=w2v_model.wv, le=le, clf=clf_w2vec
    )

    predic_w2c_transfer, probs_w2vec_transfer = dataset_factory.predict_author_from_text_w2v(
        text=text_to_predict, w2v_vectors=glove_model, le=le, clf=clf_w2vec_transfer
    )

    return JSONResponse(
        content={
            "Text to predict": text_to_predict,
            "Prediction Bag-of-words": {"expected_author": predic_bow, "probabilities": probs_bow},
            "Prediction TF-IDF": {"expected_author": predic_tf_idf, "probabilities": probs_tf_idf},
            "Prediction My W2vec": {"expected_author": predic_w2c, "probabilities": probs_w2vec},
            "Prediction Pre-Trained W2vec": {
                "expected_author": predic_w2c_transfer,
                "probabilities": probs_w2vec_transfer,
            },
            "Hyperparameters": {
                "Classifier": str(dataset_factory.clf),
            },
        }
    )
