from worker.worker import classify_email

def enqueue_email(text: str):
    task = classify_email.delay(text)
    return task.id
