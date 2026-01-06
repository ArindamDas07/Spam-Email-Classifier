from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.tasks import enqueue_email
from app.redis_conn import redis_client

app = FastAPI(title="ML Inference System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify", response_class=HTMLResponse)
def classify(request: Request, email_text: str = Form(...)):
    task_id = enqueue_email(email_text)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "email_text": email_text,
            "task_id": task_id
        }
    )

@app.get("/result/{task_id}", response_class=JSONResponse)
def get_result(task_id: str):
    result = redis_client.get(task_id)
    if result:
        return {"status": "done", "prediction": result}
    return {"status": "processing"}
