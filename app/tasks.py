# app/tasks.py
import os
from celery import Celery
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# 1. Get Redis details from .env
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB_TASKS = os.getenv("REDIS_DB_TASKS", "0")

# 2. Create a "Light" Celery app (It only needs the Broker to send messages)
celery_app = Celery(
    "worker", 
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_TASKS}"
)

def enqueue_email(text: str):
    try:
        # 3. Use send_task with the STRING NAME of the function
        # This sends the message to Redis without importing the worker code.
        task = celery_app.send_task(
            "worker.worker.classify_email",  # The task name must match the worker file structure
            args=[text]
        )
        
        logger.info(f"Task queued successfully | TaskID={task.id}")
        return task.id
    except Exception as e:
        logger.exception(f"Failed to enqueue task to Redis: {e}")
        raise e