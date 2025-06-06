import os

from fastapi import FastAPI, Request
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from keras.models import load_model

from embedding.embedding_factory import EmbeddingFactory
from preprocess.lemmatiseur import PyrrhaLemmatiseur
from utils.converter import get_model_summary

app = FastAPI()
app.mount("/img", StaticFiles(directory="static/img"), name="img")

# Serve the HTML template
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def welcome_page(request: Request):
    context = {
        "request": request,
        "title": "Explore Tertullien like never before",
        "description": "This is a simple welcome page using FastAPI and Jinja2 templates.",
        "credit_images": [
            {"src": "/img/tertullien.jpeg", "alt": "Image 1"},
            {"src": "/img/scorpiace.jpg", "alt": "Image 2"},
        ],
        "buttons": [
            {"name": "Go to Route A", "url": "/health"},
            {"name": "Go to Route B", "url": "/model_summary"},
        ],
    }
    return templates.TemplateResponse("welcome.html", context)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running successfully."}


@app.get("/get_corpus")
def get_corpus(request: Request):
    # -----Preprocess-----
    pyrrha_lemmatiseur = PyrrhaLemmatiseur(tsv_folder_path="data/Pyrrha")
    pyrrha_lemmatiseur.obtain_corpus_lemma(output_file="raw_text_lemma")
    data = pyrrha_lemmatiseur.obtain_list_sequences(raw_text_lemma_file="raw_text_lemma")
    return {"corpus": data}


@app.get("/model_json")
def get_model_json(request: Request):
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


@app.get("/model_summary", response_class=HTMLResponse)
def model_summary(request: Request):
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


@app.get("/nem_embedding")
def new_embedding(request: Request):
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


@app.get("/fit")
def dashboard(request: Request):
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
