# Spam Email Classification System – Training & Production-Grade Inference

**Author:** Arindam Das

Full Project Link: https://drive.google.com/file/d/1SnenF1up1-7cnIqHAY_Z5mxscH9JDV5Z/view?usp=sharing
---

## 1. Problem Statement

Detecting spam emails is a binary classification problem where false positives are costly. This project prioritizes high precision to minimize mislabeling legitimate emails while maintaining strong recall and overall robustness.

---

## 2. Dataset Description

- **Source:** Kaggle
- **Construction:** Merged 7 publicly available spam datasets
- **Total cleaned dataset:** 100,177 emails

### Dataset Split
| Split | Rows |
|-------|------|
| Training | 80,142 |
| Validation | 10,017 |
| Test | 10,018 |

### Classes
- 0 → Not Spam
- 1 → Spam

---

## 3. Model & Training Setup

- **Base Model:** distilbert-base-uncased
- **Max Sequence Length:** 256
- **Optimizer:** AdamW (via HuggingFace Trainer)
- **Learning Rate:** 2e-5
- **Batch Size:** 16
- **Epochs:** 3
- **Loss:** Cross-Entropy Loss
- **Mixed Precision:** FP16 enabled (GPU)

---

## 4. Validation Performance

| Epoch | Train Loss | Val Loss | Accuracy | Precision | Recall | F1 |
|-------|------------|----------|----------|-----------|--------|----|
| 1 | 0.0342 | 0.0473 | 0.9878 | 0.9922 | 0.9822 | 0.9872 |
| 2 | 0.0243 | 0.0328 | 0.9921 | 0.9892 | 0.9944 | 0.9918 |
| 3 | 0.0012 | 0.0416 | 0.9930 | 0.9927 | 0.9927 | 0.9927 |

---

## 5. Test Set Performance

- **Accuracy:** 0.9929
- **Precision:** 0.9927
- **Recall:** 0.9925
- **F1-score:** 0.9926

### Classification Report
```
              precision    recall  f1-score   support

    Not Spam     0.9931    0.9933    0.9932      5237
        Spam     0.9927    0.9925    0.9926      4781
```

### Confusion Matrix
```
[[5202   35]
 [  36 4745]]
```

✅ High precision ensures minimal false positives  
✅ Balanced recall captures most spam emails  
✅ Stable validation vs test metrics → low overfitting

---

## 6. Production Inference System Architecture

**Tech Stack:**
- API: FastAPI
- Async Task Queue: Celery
- Broker / Result Store: Redis
- Model Serving: PyTorch + Transformers
- Reverse Proxy & Load Balancer: Nginx
- Deployment: Docker & Docker Compose

**Key Features:**
- Asynchronous inference
- Horizontal scaling of API & workers
- Centralized model loading
- Threshold-based decision: Spam ≥ 0.7 probability
- Frontend UI for real-time inference

**Project Flow Diagram :**
```
User (Browser / Client)
        ↓
Nginx (Reverse Proxy & Load Balancer)
        ↓
FastAPI API Replicas (Stateless)
        ↓
Redis (Task Queue & Result Store)
        ↓
Celery Worker Pool (Async Inference)
        ↓
Spam Classification Model (DistilBERT)
        ↓
Redis (Prediction Stored)
        ↓
FastAPI API
        ↓
User Receives Result (Spam / Not Spam)

```

---

## 7. Scalability & Concurrency

- Multiple FastAPI replicas handle concurrent requests
- Celery workers scale independently
- Redis decouples API latency from model execution
- Nginx load balances API requests

Example deployment:
```
docker-compose up --build --scale api=3 --scale worker=2
```

---

## 8. Why This Project is Production-Grade

- Clean separation: training vs inference
- Versioned model & tokenizer
- Asynchronous & horizontally scalable
- Business-aware metric choice: precision-first
- Dockerized and cloud-ready
- Realistic ML system architecture suitable for industry

---

## 9. Author

**Arindam Das**  
Master’s in Electronics & Telecommunication  
ML / AI Engineer

