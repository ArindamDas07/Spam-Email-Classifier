# worker/worker.py
import os
import time
import torch
import socket
from celery import Celery
from celery.utils.log import get_task_logger
from loguru import logger
from prometheus_client import CollectorRegistry, Counter, Histogram, push_to_gateway
import mlflow
from dotenv import load_dotenv
from app.model import SpamModel
from app.preprocess import clean_text
from app.redis_conn import redis_client

# -------------------- Load .env --------------------
load_dotenv()



REDIS_HOST = os.getenv("REDIS_HOST","redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB_TASKS = int(os.getenv("REDIS_DB_TASKS",0))
REDIS_DB_RESULTS = int(os.getenv("REDIS_DB_RESULTS", 1))
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")
# maximum retries for a failed task.
CELERY_MAX_RETRIES = int(os.getenv("CELERY_MAX_RETRIES",3))
# time before retrying a failed task.
CELERY_RETRY_DELAY = int(os.getenv("CELERY_RETRY_DELAY", 10))  # seconds
PUSHGATEWAY = os.getenv("PUSHGATEWAY", "http://pushgateway:9091")

logger.info("Starting Celery worker with Redis backend, MLflow, and Pushgateway metrics.")

# -------------------- Celery Setup --------------------
celery = Celery(
    "worker",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_TASKS}",    # task queue
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_RESULTS}"    # store task results
)
# Configure Celery task logger
# Celery-specific logger for task-level logging.
# Useful for Celery monitoring tools.
task_logger = get_task_logger(__name__)

# -------------------- MLflow Setup --------------------
mlflow.set_tracking_uri(MLFLOW_URI)
experiment_name = "SpamEmailClassification"

def get_or_create_experiment(name):
    try:
        # This is the safest way to handle concurrent creation
        exp = mlflow.get_experiment_by_name(name)
        if exp:
            return exp.experiment_id
        return mlflow.create_experiment(name)
    except Exception:
        # If another worker created it between the 'get' and 'create'
        return mlflow.get_experiment_by_name(name).experiment_id

# Run this once at startup
EXPERIMENT_ID = get_or_create_experiment(experiment_name)


# -------------------- Prometheus (Pushgateway) Setup --------------------
registry = CollectorRegistry()
# --- COUNTERS: Use for things that ONLY go UP (Total, Success, Failure) ---
# Prometheus 'rate()' function works best with Counters.
TASK_COUNTER = Counter(
    "spam_tasks_total", 
    "Total tasks processed", 
    registry=registry
)

TASK_SUCCESS = Counter(
    "spam_task_success_total", 
    "Total successful tasks", 
    registry=registry
)

TASK_FAILURES = Counter(
    "spam_task_failures_total", 
    "Total failed tasks", 
    registry=registry
)

# --- HISTOGRAM: Use for TIMINGS (Latency) ---
# It automatically tracks Count, Sum, and Buckets.
TASK_LATENCY = Histogram(
    "spam_task_latency_seconds", 
    "Task latency in seconds", 
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], # BERT takes time, so these buckets are good
    registry=registry
)
def push_metrics(worker_name: str):
    try:
        # Use a unique grouping key per worker instance
        push_to_gateway(
            PUSHGATEWAY, 
            job="celery_worker_tasks", 
            grouping_key={"worker_instance": worker_name}, 
            registry=registry
        )
    except Exception as e:
        logger.error(f"Could not push to gateway: {e}")

# -------------------- Celery Task --------------------
@celery.task(
    bind=True,
    autoretry_for=(Exception,),       # retry on any Exception
    retry_backoff=True,               # exponential backoff
    retry_backoff_max=60,             # max wait 60 seconds
    retry_kwargs={"max_retries": CELERY_MAX_RETRIES}
)
def classify_email(self, text: str):
    # track latency.
    start_time =time.time()
    # multiple workers
    worker_name = socket.gethostname()
    # Count total tasks (including retries)
    TASK_COUNTER.inc()

    try:
        #load the model Loads the singleton model and tokenizer.
        model, tokenizer = SpamModel.load()
        logger.info(f"Worker={worker_name} | Task={self.request.id} | Model loaded successfully.")
        # Preprocess input
        text = clean_text(text)

        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        # Moves tensors to GPU or CPU depending on where the model is.
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # -------------------- Inference --------------------
        with torch.no_grad():
            outputs=model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0,1].item()

        label = "Spam" if prob >= 0.7 else "Not Spam"

        # -------------------- Latency --------------------
        latency = time.time() - start_time
        TASK_LATENCY.observe(latency)

        # -------------------- Store Result --------------------

        redis_client.set(self.request.id,label)
        TASK_SUCCESS.inc()
        logger.info(
        f"Worker={worker_name} | Task={self.request.id} | Label={label} | Latency={latency:.3f}s"
        )
        # Push metrics to Pushgateway
        push_metrics(worker_name)
     
        # -------------------- MLflow Logging --------------------
        try:
            # Explicitly use the experiment_id to ensure logs go to the right place
            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=f"task_{self.request.id}"):
                mlflow.log_param("prediction", label)
                mlflow.log_metric("spam_probability", prob)
                mlflow.log_metric("latency_ms", latency * 1000)
                logger.info(f"MLflow logged for task {self.request.id}")
        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")
        
        return label
    except Exception as e:
        TASK_FAILURES.inc()
        push_metrics(worker_name)
        logger.exception(f"Task {self.request.id}: Failed with error: {e}. Retrying...")
        raise e  # Celery will handle retry automatically
           

    