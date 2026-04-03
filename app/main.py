from fastapi import FastAPI, Request, Form,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from loguru import logger
import time

from app.tasks import enqueue_email
from app.redis_conn import redis_client
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Spam Email Classification API")
# -------------------- Prometheus --------------------
# instrument(app) → attaches metrics collection to FastAPI
# expose(app) → creates /metrics endpoint
Instrumentator().instrument(app).expose(app)

# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------- Routes --------------------
@app.get("/home")
async  def health():
    return {"status": "ok"}


@app.post("/classify")
async def classify(request: Request):
    start_time = time.time()
    try:
        data = await request.json()
        email_text = data.get("email", "")
        logger.info(f"Received email | Preview: {email_text[:50]}")
    
        task_id = enqueue_email(email_text)
        latency = round(time.time()-start_time,3)
        logger.info(
            f"Task queued | TaskID={task_id} | Latency={latency}s"
        )
        return {"task_id": task_id}
        
    except Exception as e:
        logger.exception(f"Failed to enqueue task: {e}")

        raise HTTPException(status_code=400, detail="process failed")

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    try:
        result = redis_client.get(task_id)
        if result:
            logger.info(f"Result fetched | TaskID={task_id} | Prediction={result}")
            return {"status": "done", "prediction": result}
        return {"status": "processing"}
    except Exception as e:
        logger.exception(f"Error fetching result for TaskID={task_id}: {e}")
        return {"status": "error", "message": "Internal server error"}





# 🔥 ALWAYS KEEP THIS AT THE VERY END
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")