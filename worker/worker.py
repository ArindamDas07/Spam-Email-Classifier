import os
import time
import torch
import socket
import json
from celery import Celery
from celery.utils.log import get_task_logger
from loguru import logger
from prometheus_client import CollectorRegistry, Counter, Histogram, push_to_gateway
import mlflow
from dotenv import load_dotenv

# Internal imports from the isolated worker directory
from worker.model import SpamModel
from worker.preprocess import clean_text
from app.redis_conn import redis_client

# -------------------- Load Configuration --------------------
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB_TASKS = int(os.getenv("REDIS_DB_TASKS", 0))
REDIS_DB_RESULTS = int(os.getenv("REDIS_DB_RESULTS", 1))
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")
PUSHGATEWAY = os.getenv("PUSHGATEWAY", "http://pushgateway:9091")

CELERY_MAX_RETRIES = int(os.getenv("CELERY_MAX_RETRIES", 3))

logger.info("Initializing Spam Worker: Decoupled BERT Inference Engine started.")

# -------------------- Celery Setup --------------------
# Standardized App Name: Must match the Producer (API) side for task routing
celery = Celery(
    "worker",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_TASKS}",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_RESULTS}"
)

# -------------------- MLflow Observability Setup --------------------
mlflow.set_tracking_uri(MLFLOW_URI)
experiment_name = "SpamEmailClassification"

def get_or_create_experiment(name):
    """Safely initializes the MLflow experiment for concurrent worker scaling."""
    try:
        exp = mlflow.get_experiment_by_name(name)
        if exp:
            return exp.experiment_id
        return mlflow.create_experiment(name)
    except Exception:
        # Prevent race condition if multiple replicas create experiment simultaneously
        return mlflow.get_experiment_by_name(name).experiment_id

EXPERIMENT_ID = get_or_create_experiment(experiment_name)

# -------------------- Prometheus Monitoring Setup --------------------
# Using a local registry for atomic pushes to the Pushgateway
registry = CollectorRegistry()

TASK_COUNTER = Counter(
    "spam_tasks_total", 
    "Total classification tasks processed", 
    registry=registry
)

TASK_SUCCESS = Counter(
    "spam_task_success_total", 
    "Total successful classifications", 
    registry=registry
)

TASK_FAILURES = Counter(
    "spam_task_failures_total", 
    "Total failed worker tasks", 
    registry=registry
)

# Labeled Histogram to track inference distribution (P95 Latency)
TASK_LATENCY = Histogram(
    "spam_task_latency_seconds", 
    "Inference latency in seconds", 
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=registry
)

def push_metrics(worker_name: str):
    """Reports worker telemetry to the centralized Prometheus stack."""
    try:
        push_to_gateway(
            PUSHGATEWAY, 
            job="spam_worker_nodes", 
            grouping_key={"worker_instance": worker_name}, 
            registry=registry
        )
    except Exception as e:
        logger.error(f"Pushgateway sync failed: {e}")

# -------------------- Core Celery Task --------------------
@celery.task(
    bind=True,
    name="worker.worker.classify_email", # Explicit naming for string-based routing
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": CELERY_MAX_RETRIES}
)
def classify_email(self, text: str):
    """
    Asynchronous Inference Task:
    1. Preprocesses raw text.
    2. Executes DistilBERT inference.
    3. Synchronizes telemetry (MLflow + Prometheus).
    4. Persists results to Redis DB 2 for API polling.
    """
    start_time = time.time()
    worker_name = socket.gethostname()
    TASK_COUNTER.inc()

    try:
        # SINGLETON LOAD: Weights loaded into memory exactly once per container
        model, tokenizer = SpamModel.load()
        
        # Data Sanitization
        clean_input = clean_text(text)

        # Tokenization & Tensor mapping
        inputs = tokenizer(
            clean_input,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        
        # Device Agnostic Execution (Handles CUDA/CPU fallback)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # INFERENCE (torch.no_grad used inside SpamModel.load() logic)
        with torch.no_grad():
            outputs = model(**inputs)
            # Binary Classification: Index [1] represents the Spam probability
            prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

        # Decision Threshold (Senior logic: Conservative 0.7 for precision)
        label = "Spam" if prob >= 0.7 else "Not Spam"

        # Record Performance Metrics
        latency = time.time() - start_time
        TASK_LATENCY.observe(latency)
        TASK_SUCCESS.inc()

        # --- DATA SYNC ---
        # 1. Redis: High-speed result storage for stateless API polling
        redis_client.set(self.request.id, label)
        
        # 2. MLflow: Comprehensive Audit Trail
        try:
            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=f"task_{self.request.id}"):
                mlflow.log_param("prediction", label)
                mlflow.log_param("text_length", len(text))
                mlflow.log_metric("spam_probability", prob)
                mlflow.log_metric("latency_ms", latency * 1000)
        except Exception as e:
            logger.error(f"MLflow sync failed: {e}")

        # 3. Prometheus: Infrastructure telemetry
        push_metrics(worker_name)
        
        logger.success(f"Task {self.request.id} complete | Label: {label} | Worker: {worker_name}")
        return label

    except Exception as e:
        TASK_FAILURES.inc()
        push_metrics(worker_name)
        logger.exception(f"Classification failed for Task {self.request.id}: {e}")
        raise e  # Triggers Celery Auto-retry