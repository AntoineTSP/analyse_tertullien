from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/")
def dashboard(request: Request):
    return "Hello, World! This is the dashboard."
