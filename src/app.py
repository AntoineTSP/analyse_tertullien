import os
from pathlib import Path

import gensim.downloader
import numpy as np
from fastapi import FastAPI, Query, Request, Response
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
    get_all_annotations_categories,
    get_model_summary,
    parse_full_tei,
)

app = FastAPI()
app.mount("/img", StaticFiles(directory="static/img"), name="img")

# Serve the HTML template
templates = Jinja2Templates(directory="templates")
templates.env.filters["tojson"] = lambda value: value  # fallback, not needed if Jinja2 >=3.1

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
        # Working
        # "form_routes": [
        #     {
        #         "name": "Go to predict_author",
        #         "url": "/author_classification/predict_author/",
        #         "parameters": [
        #             {"label": "Classifier Type", "param_name": "clf_type"},
        #             {"label": "Text on which we want a predicted author", "param_name": "text_to_predict"},
        #         ],
        #     },
        # ],
        # Fail
        # "form_routes": [
        #     {
        #         "name": "Go to predict_author",
        #         "url": "/author_classification/predict_author/",
        #         "parameters": [
        #             {
        #                 "label": "Classifier Type",
        #                 "param_name": "clf_type",
        #                 "type": "select",
        #                 "choices": ["SVM", "LinearSVC"],
        #             },
        #             {"label": "Other Param", "param_name": "option", "type": "text"},
        #         ],
        #     },
        #     {
        #         "name": "Embedding Route",
        #         "url": "/embedding",
        #         "parameters": [
        #             {"label": "Text to embed", "param_name": "text", "type": "text"},
        #         ],
        #     },
        # ],
    }
    return templates.TemplateResponse("welcome.html", context)


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running successfully."}


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
        ],
        "form_routes": [
            {
                "name": "Get corpus from tsv",
                "url": "/embedding/get_corpus",
                "parameters": [
                    {
                        "label": "TSV folder to read from",
                        "param_name": "tsv_folder_path",
                        "type": "select",
                        "options": ["Pyrrha"],
                    },
                ],
            },
            {
                "name": "Fit a LSTM model",
                "url": "/embedding/fit",
                "parameters": [
                    {
                        "label": "TSV folder to read from",
                        "param_name": "tsv_folder_path",
                        "type": "select",
                        "options": ["Pyrrha"],
                    },
                ],
            },
            {
                "name": "Get JSON model description",
                "url": "/embedding/model_json",
                "parameters": [
                    {
                        "label": "TSV folder to read from",
                        "param_name": "tsv_folder_path",
                        "type": "select",
                        "options": ["Pyrrha"],
                    },
                    {
                        "label": "Output name of your model",
                        "param_name": "model_name",
                        "type": "select",
                        "options": ["test"],
                    },
                ],
            },
            {
                "name": "Get model summary",
                "url": "/embedding/model_summary",
                "parameters": [
                    {
                        "label": "TSV folder to read from",
                        "param_name": "tsv_folder_path",
                        "type": "select",
                        "options": ["Pyrrha"],
                    },
                    {
                        "label": "Output name of your model",
                        "param_name": "model_name",
                        "type": "select",
                        "options": ["test"],
                    },
                ],
            },
            {
                "name": "Create a new embedding from words then return the top-10 nearest words",
                "url": "/embedding/nem_embedding",
                "parameters": [
                    {
                        "label": "TSV folder to read from",
                        "param_name": "tsv_folder_path",
                        "type": "select",
                        "options": ["Pyrrha"],
                    },
                    {
                        "label": "Output name of your model",
                        "param_name": "model_name",
                        "type": "select",
                        "options": ["test"],
                    },
                    {
                        "label": "Words separated with a whitespace to be added for the new embedding",
                        "param_name": "positive_words",
                    },
                    {
                        "label": "Words separated with a whitespace to be subtracted for the new embedding",
                        "param_name": "negative_words",
                    },
                ],
            },
        ],
        # Add modularity here!TODO
    }
    return templates.TemplateResponse("welcome.html", context)


@app.get("/embedding/get_corpus")
async def get_corpus(request: Request, tsv_folder_path: str = Query(description="Path to the TSV folder")):
    if tsv_folder_path == "":
        tsv_folder_path = "Pyrrha"
    # -----Preprocess-----
    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path=f"data/{tsv_folder_path}")
    pyrrha_lemmatiseur.obtain_corpus_lemma(output_file="raw_text_lemma")
    data = pyrrha_lemmatiseur.obtain_list_sequences(raw_text_lemma_file="raw_text_lemma")
    return {"corpus": data}


@app.get("/embedding/fit")
async def dashboard(request: Request, tsv_folder_path: str = Query(description="Path to the TSV folder")):
    if tsv_folder_path == "":
        tsv_folder_path = "Pyrrha"
    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path=f"data/{tsv_folder_path}")
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
        "corpus_name": tsv_folder_path,
        "nb_sequences": len(data),
        "unique_labels": np.unique(labels).tolist(),
        "accuracy": embedding_factory.get_accuracy(X_test=X_test, y_test=y_test),
        "training_logs": logs,
    }


@app.get("/embedding/model_json")
async def get_model_json(
    request: Request,
    tsv_folder_path: str = Query(description="Path to the TSV folder"),
    model_name: str = Query(description="Name of the keras model to load"),
):
    if tsv_folder_path == "":
        tsv_folder_path = "Pyrrha"
    if model_name == "":
        model_name = "test"
    if os.path.exists(f"data/models/{model_name}.keras"):
        model = load_model(f"data/models/{model_name}.keras")
        model_json = model.to_json()
        return JSONResponse(content=model_json)
    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path=f"data/{tsv_folder_path}")
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
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_name=model_name
    )
    model_json = embedding_factory.model.to_json()
    return JSONResponse(content=model_json)


@app.get("/embedding/model_summary", response_class=HTMLResponse)
async def model_summary(
    request: Request,
    tsv_folder_path: str = Query(description="Path to the TSV folder"),
    model_name: str = Query(description="Name of the keras model to load"),
):
    if tsv_folder_path == "":
        tsv_folder_path = "Pyrrha"
    if model_name == "":
        model_name = "test"
    if os.path.exists(f"data/models/{model_name}.keras"):
        model = load_model(f"data/models/{model_name}.keras")
        summary = get_model_summary(model)
        return templates.TemplateResponse("summary.html", {"request": request, "summary": summary})
    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path=f"data/{tsv_folder_path}")
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
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_name=model_name
    )
    summary = get_model_summary(embedding_factory.model)
    return templates.TemplateResponse("summary.html", {"request": request, "summary": summary})


@app.get("/embedding/nem_embedding")
async def new_embedding(
    request: Request,
    tsv_folder_path: str = Query(description="Path to the TSV folder"),
    model_name: str = Query(description="Name of the keras model to load"),
    positive_words: str = Query(description="List of words to be added for the new embedding"),
    negative_words: str = Query(description="List of words to be subtracted for the new embedding"),
):
    if positive_words == "" and negative_words == "":
        positive_words = ["marcion", "cupiditas"]
        negative_words = ["apocalypsis"]
    else:
        positive_words = positive_words.split()
        negative_words = negative_words.split()

    if not isinstance(positive_words, list) or not isinstance(negative_words, list):
        return JSONResponse(
            content={"error": "positive_words and negative_words should be lists of strings."},
            status_code=400,
        )

    if tsv_folder_path == "":
        tsv_folder_path = "Pyrrha"
    if model_name == "":
        model_name = "test"
    if os.path.exists(f"data/models/{model_name}.keras"):
        model = load_model(f"data/models/{model_name}.keras")
    else:
        raise FileNotFoundError("Model not found. Please train the model first.")

    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path=f"data/{tsv_folder_path}")
    data = pyrrha_lemmatiseur.obtain_list_sequences(raw_text_lemma_file="raw_text_lemma")
    # -----Embedding-----
    embedding_factory = EmbeddingFactory(paragraphs=data, model=model)

    new_embedding = embedding_factory.get_new_embedding(
        positive_words=positive_words, negative_words=negative_words
    )
    return {
        "corpus_name": tsv_folder_path,
        "model_name": model_name,
        "positive_words": positive_words,
        "negative_words": negative_words,
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
            {
                "name": "Get all figure of speech categories",
                "url": "/text_annotation/get_annotations_categories",
            },
        ],
        "form_routes": [
            {
                "name": "Get the XML of a text",
                "url": "/text_annotation/xml/",
                "parameters": [
                    {
                        "label": "XML file to be displayed",
                        "param_name": "xml_path",
                        "type": "select",
                        "options": ["Idol.xml", "Pall2.xml", "Mart2.xml", "Cult2.xml", "Val_2.xml"],
                    },
                ],
            },
            {
                "name": "Get the Interactive HTML of a XML",
                "url": "/text_annotation/xml_to_html/",
                "parameters": [
                    {
                        "label": "XML file to be displayed",
                        "param_name": "xml_path",
                        "type": "select",
                        "options": ["Idol.xml", "Pall2.xml", "Mart2.xml", "Cult2.xml", "Val_2.xml"],
                    },
                ],
            },
            {
                "name": "Retrieve all/some figure of speech annotations on the provided XML",
                "url": "/text_annotation/get_annotation/",
                "parameters": [
                    {
                        "label": "XML file to be displayed",
                        "param_name": "xml_path",
                        "type": "select",
                        "options": ["Idol.xml", "Pall2.xml", "Mart2.xml", "Cult2.xml", "Val_2.xml"],
                    },
                    {
                        "label": "Figure of speech to select",
                        "param_name": "annotation",
                        "type": "select",
                        "options": [
                            "",
                            "metaphore",
                            "comparaison",
                            "personnification",
                            "enargeia",
                            "dialogisme",
                            "deductio_ad_absurdum",
                            "symbolisme",
                            "exemple",
                            "militaire",
                            "retorsion_inversée",
                            "irreel",
                            "epiphoneme",
                            "ab_absurdo",
                            "ab_origine",
                            "calembour",
                            "dicton",
                            "prosopopeia",
                            "scene",
                            "corps",
                        ],
                    },
                ],
            },
        ],
    }
    return templates.TemplateResponse("welcome.html", context)


@app.get("/text_annotation/xml", response_class=Response)
async def get_xml(xml_path: str = Query(description="Path to the XML file")):
    if xml_path == "":
        xml_path = "Pall2.xml"
    xml_path = Path(f"./data/xml/{xml_path}")
    if not xml_path.exists():
        return Response(
            content="File not found. It only supports Cult2.xml | Idol.xml | Mart2.xml | Pall2.xml | Val_2.xml",
            status_code=404,
        )

    content = xml_path.read_text(encoding="utf-8")
    return Response(content=content, media_type="application/xml")


@app.get("/text_annotation/get_annotations_categories")
async def get_annotations_categories(
    request: Request,
):
    annotations_categories = get_all_annotations_categories()
    return JSONResponse(content=annotations_categories)


@app.get("/text_annotation/xml_to_html", response_class=HTMLResponse)
async def xml_to_html(request: Request, xml_path: str = Query(description="Path to the XML file")):
    if xml_path == "":
        xml_path = "Pall2.xml"
    xml_path = Path(f"./data/xml/{xml_path}")
    if not xml_path.exists():
        return Response(
            content="File not found. It only supports Cult2.xml | Idol.xml | Mart2.xml | Pall2.xml | Val_2.xml",
            status_code=404,
        )
    data = parse_full_tei(xml_path)  # your function generating html string
    data["body_html"] = add_title_to_spans(data["body_html"])
    return templates.TemplateResponse("xml.html", {"request": request, **data})


@app.get("/text_annotation/get_annotation")
async def get_annotation(
    request: Request,
    xml_path: str = Query(description="Path to the XML file"),
    annotation: str = Query(description="Which figure of speech to select"),
):
    if xml_path == "":
        xml_path = "Pall2.xml"
    xml_path = Path(f"./data/xml/{xml_path}")
    if not xml_path.exists():
        return Response(
            content="File not found. It only supports Cult2.xml | Idol.xml | Mart2.xml | Pall2.xml | Val_2.xml",
            status_code=404,
        )
    annotations = get_all_annotations(xml_path=xml_path)
    if annotation != "":
        if annotation in get_all_annotations_categories():
            annotations = annotations[annotation]
        else:
            return Response(
                content=f"Figure of speech {annotation} has not been annotated. Supported ones are {get_all_annotations_categories()}",
                status_code=404,
            )
    annotations = {"xml_path": str(xml_path), **annotations}
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
            # {"name": "Get Wordcloud", "url": "/author_classification/get_wordcloud"},
            {"name": "Plot Wordcloud", "url": "/author_classification/plot_wordclouds"},
            {"name": "Get Dataframe", "url": "/author_classification/get_df"},
            {"name": "Get Labels Classes", "url": "/author_classification/get_labels_classes"},
            {"name": "Get Author Repartition", "url": "/author_classification/get_author_repartition"},
            {"name": "Plot Top words", "url": "/author_classification/plot_top_words"},
        ],
        "form_routes": [
            {
                "name": "Get Author Clean Text",
                "url": "/author_classification/get_clean_author_text",
                "parameters": [
                    {
                        "label": "Which author text to retrieve",
                        "param_name": "author",
                        "type": "select",
                        "options": ["", "Apulée", "Cicéron", "De Pallio", "Apologeticum"],
                    },
                ],
            },
            {
                "name": "Get Accuracy for each model",
                "url": "/author_classification/get_accuracy",
                "parameters": [
                    {
                        "label": "Classifier Type",
                        "param_name": "clf_type",
                        "type": "select",
                        "options": ["LinearSVC", "SVC"],
                    },
                ],
            },
            {
                "name": "Go to predict_author",
                "url": "/author_classification/predict_author/",
                "parameters": [
                    {
                        "label": "Classifier Type",
                        "param_name": "clf_type",
                        "type": "select",
                        "options": ["LinearSVC", "SVC"],
                    },
                    {"label": "Text on which we want a predicted author", "param_name": "text_to_predict"},
                ],
            },
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
async def get_clean_author_text(author: str = Query(description="Author name")):
    dataset_factory = DatasetFactory()
    print(f"Author requested: {author}")
    if author == "":
        return JSONResponse(
            content={
                "Text Apulée": dataset_factory.text_apu,
                "Text Cicéron": dataset_factory.text_ciceron,
                "Text De Pallio": dataset_factory.text_de_pallio,
                "Text Apologeticum": dataset_factory.text_apol,
            }
        )
    elif author in ["Apulée", "Cicéron", "De Pallio", "Apologeticum"]:
        if author == "Apulée":
            text = dataset_factory.text_apu
        elif author == "Cicéron":
            text = dataset_factory.text_ciceron
        elif author == "De Pallio":
            text = dataset_factory.text_de_pallio
        elif author == "Apologeticum":
            text = dataset_factory.text_apol
        else:
            return JSONResponse(
                content={"error": f"Author {author} not recognized."},
                status_code=404,
            )
        return JSONResponse(content={"Author": author, "Text": text})
    else:
        return JSONResponse(
            content={
                "error": (
                    f"Author {author} not recognized. Supported authors are: "
                    "Apulée, Cicéron, De Pallio, Apologeticum."
                )
            },
            status_code=404,
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
async def get_accuracy(
    request: Request,
    clf_type: str = Query(description="Type of classifier to use (e.g., LinearSVC, SVM)"),
):
    if clf_type == "":
        clf_type = "LinearSVC"
    dataset_factory = DatasetFactory(clf_type=clf_type)
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
async def predict_author(
    clf_type: str = Query(description="Type of classifier to use (e.g., LinearSVC, SVM)"),
    text_to_predict: str = Query(description="Text to predict author for"),
):
    if text_to_predict == "":
        text_to_predict = "Quid bene amat bene castigat"
    if clf_type == "":
        clf_type = "LinearSVC"
    dataset_factory = DatasetFactory(clf_type=clf_type)
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
