# 📧 Scalable Spam Email Classifier: Training & Production-Grade Inference

**Author:** Arindam Das\
Full Project Bundle (Ready-to-Run with Model): 🚀 Download from Google
Drive:https://drive.google.com/file/d/1Jkbx4COiwly2zdOU60UU62Iq8glaXdTa/view?usp=drive_link

## 1. 📝 Project Overview

-   This project is a comprehensive MLOps end-to-end solution. It
    bridges the gap between high-accuracy Machine Learning research and
    a distributed, horizontally scalable production architecture.
-   🧠 Research Phase: Fine-tuned DistilBERT on 100,000+ emails for
    state-of-the-art spam detection.
-   🏗️ Production Phase: A decoupled, multi-container system designed
    for high availability, traffic shielding, and full observability.

## 2. 🧬 Machine Learning Lifecycle

### 📊 Dataset & Training

-   Source: Kaggle (Merged 7 public spam datasets).
-   Cleaned Samples: 100,177 emails.
-   Model: distilbert-base-uncased (66M parameters).
-   Strategy: Selected DistilBERT for its Production-Accuracy
    Trade-off---offering 60% faster inference than BERT-Base while
    maintaining 99%+ accuracy.
-   Notebooks: See training.ipynb and evaluation.ipynb for full training
    logic and loss curves.

### 📈 Test Set Performance

-   Accuracy: 99.29%
-   Precision (Spam): 99.27%
-   F1-Score: 99.26%

Confusion Matrix:

    [[5202   35]]  <-- Only 35 False Positives (Legitimate emails marked as Spam)
    [  36 4745]]

## 3. 🏗️ Advanced Production Architecture

### 🗺️ System Flow Diagram

    User (Browser / Client)
            ↓
    Nginx (Reverse Proxy, Load Balancer & Rate Limiter) 🛡️
            ↓
    FastAPI API Replicas (Stateless Ingestion - Lightweight) ⚡
            ↓
    Redis (Task Queue - DB 0) 📩
            ↓
    Celery Worker Pool (Async DistilBERT Inference) 🧠
            ↓
    MLflow (Experiment & Metric Tracking) 🧪
            ↓
    Redis (Result Storage - DB 2) 📥
            ↓
    FastAPI API
            ↓
    User Receives Result (Spam / Not Spam) ✅

### 🛡️ Production-Grade Features 

-   Strict Architectural Decoupling: API containers are stripped of
    heavy ML libraries (No Torch/Transformers), reducing RAM footprint
    from 1.2GB to \~60MB per replica.
-   Nginx Traffic Shield: Acts as a Reverse Proxy and Rate Limiter
    (10r/s) to protect BERT workers from bot attacks and CPU exhaustion.
-   Smart Load Balancing: Uses the least_conn algorithm to route traffic
    to the container with the lowest active workload.
-   Dynamic Service Discovery: Implemented a DNS Resolver (127.0.0.11)
    in Nginx to prevent "502 Bad Gateway" errors during container
    restarts or auto-scaling.
-   Asynchronous Processing: Heavy model inference is handled by Celery
    workers, ensuring the frontend UI remains responsive and never
    "freezes."

## 4. 👁️ Full-Stack Observability

-   We monitor the system from two perspectives: AI Performance and
    Infrastructure Health.
-   🧪 MLflow: Logs exact probabilities, model confidence, and inference
    parameters for every request.
-   📈 Prometheus: Scrapes real-time system metrics (Requests per
    second, Error rates, RAM usage).
-   📮 Pushgateway: Captures metrics from ephemeral Celery workers to
    track BERT latency and success counts.
-   🎨 Grafana: Visualizes the entire ecosystem, including P95 Latency
    and worker throughput, in a single dashboard.

## 5. 💻 How to Run

### 📦 Option A: GitHub Setup (Source Code)

-   Clone this repository.
-   Download the model weights (\~256 MB) from the Google Drive link
    above.
-   Place the weights in the models/bert-spam/ directory.
-   Deploy:

```
    docker-compose up --build -d --scale api=3 --scale worker=2
```
### 📥 Option B: One-Click Setup (Google Drive Bundle)

-   Download and unzip the bundle from the Google Drive link.
-   Open a terminal in the project folder.
-   Deploy:

```
    docker-compose up --build -d --scale api=3 --scale worker=2
```
## 🔗 Monitoring Endpoints

-   🌍 Main Application: http://localhost
-   📊 API Telemetry: http://localhost/metrics
-   📮 Pushgateway UI: http://localhost:9091
-   🧪 MLflow Dashboard: http://localhost:5000
-   📡 Prometheus UI: http://localhost:9090
-   🎨 Grafana Dashboard: http://localhost:3000

## 🚀 Scalability & Roadmap

-   The architecture is Cloud-Ready. By using Redis and Celery, the
    inference workers can be moved to a dedicated GPU cluster while the
    API remains on standard web servers, ensuring maximum cost
    efficiency and performance at any scale.

------------------------------------------------------------------------

Arindam Das\
Master's in Electronics & Telecommunication\
ML / AI / MLOps Engineer
